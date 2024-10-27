import torch
import torch.nn as nn

from timm.layers.helpers import to_2tuple

import pathlib
import numpy as np
from transformers.models.switch_transformers.modeling_switch_transformers import *

class ConvMlpExpert(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.ReLU,
            norm_layer=None,
            bias=True,
            drop=0.,
            expert_scale=4,  # number of experts
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        hidden_features = hidden_features // expert_scale
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class ConvMoE(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.ReLU,
            norm_layer=None,
            bias=True,
            drop=0.,
            num_experts=4,
            top_k=2,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.gate = nn.Sequential(
            nn.Linear(in_features, num_experts),
            nn.Softmax(dim=-1)
        )
        
        self.experts = nn.ModuleList([
            ConvMlpExpert(
                in_features=in_features,
                hidden_features=hidden_features,
                out_features=out_features,
                act_layer=act_layer,
                norm_layer=norm_layer,
                bias=bias,
                drop=drop,
                expert_scale=num_experts
            ) for _ in range(num_experts)
        ])
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.out_features = out_features
        self.routing_logits = None
        self.routing_index = None

    def get_routing_logits(self):
        if self.routing_logits is None:
            return 0
        ret_logits = self.routing_logits
        self.routing_logits = None
        return ret_logits, self.routing_index

    def forward(self, x):
        # x shape: (batch, in_features, H, W)
        batch_size, in_features, H, W = x.shape
        
        x_reshaped = x.permute(0, 2, 3, 1)  # (batch, H, W, in_features)
        x_flattened = x_reshaped.reshape(-1, in_features)  # (batch*H*W, in_features)
        
        gate_scores = self.gate(x_flattened)  # (batch*H*W, num_experts)
        if self.training:
            self.routing_logits = gate_scores.reshape(batch_size, -1, self.num_experts)
        
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)  # (batch*H*W, top_k)
        top_k_scores = torch.softmax(top_k_scores, dim=-1)  
        
        output = torch.zeros((batch_size, self.out_features, H, W), device=x.device)
        
        expert_index_list = []
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]  # (batch*H*W,)
            expert_weights = top_k_scores[:, k]  # (batch*H*W,)
            expert_index_list.append(expert_indices.reshape(batch_size, -1, 1))
            for i in range(self.num_experts):
                mask = (expert_indices == i)  # (batch*H*W,)
                if not mask.any():
                    continue
                    
                expert_output = self.experts[i](x)  # (batch, out_features, H, W)
                expert_weights_reshaped = expert_weights.reshape(batch_size, 1, H, W)  # (batch, 1, H, W)
                output = output + expert_output * expert_weights_reshaped
        self.routing_index = expert_index_list
        # print(output.shape)
        return output
    
    
def mobilevit2sparse(model: nn.Module, sparse_label_path):
    label_path = pathlib.Path(sparse_label_path)
    module_map = {mm: m for mm, m in model.named_modules()}
    for file_path in label_path.iterdir():
        # print(file_path, file_path.name)
        # print(torch.load(file_path))
        module_name, layer_idx = file_path.name.split("transformer")
        layer_idx = layer_idx[1:]
        layer_idx = int(layer_idx.split('.')[0])
        module_name = module_name + 'transformer'
        
        in_features = module_map[module_name][layer_idx].mlp.fc1.weight.shape[1]
        hidden_feature = module_map[module_name][layer_idx].mlp.fc1.weight.shape[0]
        out_features = module_map[module_name][layer_idx].mlp.fc2.weight.shape[0]
        splitting_idxs = torch.load(file_path)
        # print(np.unique(np.stack(splitting_idxs)))
        n_experts = len(np.unique(np.stack(splitting_idxs)))
        # print(hidden_feature, hidden_feature // n_experts)
        moe_mlp = ConvMoE(in_features, hidden_feature, out_features, num_experts=n_experts)
        dense_mlp = module_map[module_name][layer_idx].mlp
        expert_idx = np.stack(splitting_idxs)
        with torch.no_grad():
            for i in range(n_experts):
                # print(moe_mlp.experts[i].fc1.weight.shape, dense_mlp.fc1.weight.shape)
                moe_mlp.experts[i].fc1.weight.copy_(dense_mlp.fc1.weight[expert_idx==i])
                if dense_mlp.fc1.bias is not None:
                    moe_mlp.experts[i].fc1.bias.copy_(dense_mlp.fc1.bias[expert_idx==i])
                
                # print(moe_mlp.experts[i].fc2.weight.shape, dense_mlp.fc2.weight.shape, dense_mlp.fc2.weight[:,expert_idx==i].shape)
                moe_mlp.experts[i].fc2.weight.copy_(dense_mlp.fc2.weight[:,expert_idx==i])
                if dense_mlp.fc2.bias is not None:
                    moe_mlp.experts[i].fc2.bias.copy_(dense_mlp.fc2.bias)
                
                # setting
                module_map[module_name][layer_idx].mlp = moe_mlp
        # print(type(module_map[module_name][layer_idx]))
        # for mm, m in model.name
        
    return model

def get_routing_logits(model: nn.Module):
    routing_probs = []
    for mm, m in model.named_modules():
        if hasattr(m, 'get_routing_logits'):
            routing_probs.append(m.get_routing_logits())

    return routing_probs
    
def load_balance_loss(router_probs):
    b_loss = None
    n_expert = 4
    for layer in router_probs:
        if len(layer) < 2:
            continue
        
        if type(layer[1]) is tuple or type(layer[1]) is list:
            n_expert = len(layer[1])
            break
    
    b_loss = None
    z_loss = None
    
    for i in range(n_expert):
        total_router_logits = {}
        total_expert_indexes = {}
        for layer in router_probs:
            if len(layer) > 1:
                router_logits, expert_indexes = layer
                expert_indexes = expert_indexes[i]
                if router_logits.shape[0] not in total_router_logits:
                    total_router_logits[router_logits.shape[0]] = []
                    total_expert_indexes[router_logits.shape[0]] = []
                total_router_logits[router_logits.shape[0]].append(router_logits)
                total_expert_indexes[router_logits.shape[0]].append(expert_indexes)
                # print(expert_indexes)
                # print(router_logits.shape)
        for kk in total_router_logits:
            sub_router_logits = total_router_logits[kk]
            sub_expert_indexes = total_expert_indexes[kk]
            sub_router_logits = torch.cat(sub_router_logits, dim=1)
            sub_expert_indexes = torch.cat(sub_expert_indexes, dim=1)
            if z_loss is None:
                z_loss = router_z_loss_func(sub_router_logits)

            router_probs_i = nn.Softmax(dim=-1)(sub_router_logits)
            if b_loss is None:
                b_loss = load_balancing_loss_func(router_probs_i, sub_expert_indexes)
            else:
                b_loss += load_balancing_loss_func(router_probs_i, sub_expert_indexes)
    # print(b_loss, z_loss)
    # tensor(1.8232, device='cuda:0', grad_fn=<AddBackward0>) tensor(673.1109, device='cuda:0', grad_fn=<DivBackward0>)
    return b_loss, z_loss


def z_balance_loss(router_probs):
    z_loss = None
    norm_scale = 1
    for layer in router_probs:
        if len(layer) < 2:
            continue
        # logits = nn.functional.softmax(layer[0], dim=-1)
        logits = layer[0]
        # router_z_loss_func(layer[0])
        # if type(layer[1]) is tuple or type(layer[1]) is list:
        norm_scale = 1
        if z_loss is None:
            z_loss = router_z_loss_func(logits) * norm_scale
        else:
            z_loss += router_z_loss_func(logits) * norm_scale
    return z_loss