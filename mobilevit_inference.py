import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchmetrics.classification import (
    Accuracy,
    Precision,
    Recall,
    F1Score
)
from torchmetrics import MetricCollection

import timm
from tqdm import tqdm
import sys

import argparse

sys.path.append('..')

from src.dense_to_sparse import *

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument("--balance_loss_weight", type=float, default=0.1)
parser.add_argument("--z_loss_weight", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--timm_model", type=str, default="mobilevitv2_100")
parser.add_argument("--model_splitting_dir", type=str, default="model/mobilevitv2_100_s/param_split")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--store_dir", type=str, default='model')

args = parser.parse_args()

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define transformations for the training and validation sets
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to the size expected by MobileViT
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the dataset
])

metrics = MetricCollection({
    'accuracy': Accuracy(task='multiclass', num_classes=10),
    'precision': Precision(task='multiclass', num_classes=10, average='macro'),
    'recall': Recall(task='multiclass', num_classes=10, average='macro'),
    'f1': F1Score(task='multiclass', num_classes=10, average='macro'),
})

# Load the Fashion MNIST dataset
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# Load a pre-trained MobileViT model from timm
model = timm.create_model(args.timm_model, pretrained=True)

# Modify the final classification layer for 10 classes (Fashion MNIST)
model.head.fc = nn.Linear(model.get_classifier().in_features, 10)

# transfer model to Sparse MoE model
model = mobilevit2sparse(model, args.model_splitting_dir)

# Move the model to the correct device
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Optionally, define a learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# Evaluate the model
model.load_state_dict(torch.load(pathlib.Path(args.store_dir).joinpath(f"{args.timm_model}_smoe_smnist.pth")))

# for pn, p in model.named_parameters():
#     print(pn, p.shape)

model.eval()
metrics.reset()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        metrics.update(preds=predicted.cpu(), target=labels.cpu())

print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')
print(metrics.compute())

# Training loop
# num_epochs = args.num_epochs
# best_f1 = 0
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     metrics.reset()
#     for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
#         images, labels = images.to(device), labels.to(device)

#         # Zero the parameter gradients
#         optimizer.zero_grad()

#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         routing_logits = get_routing_logits(model)
#         routing_loss = load_balance_loss(routing_logits)
#         loss += args.balance_loss_weight * routing_loss[0] + args.z_loss_weight * routing_loss[1] 
#         # Backward pass and optimize
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         with torch.no_grad():
#             # print(outputs.data.shape)
#             _, predicted = torch.max(outputs.data, 1)
#             # print(predicted)
#             metrics.update(preds=predicted.cpu(), target=labels.cpu())
#             # if i % 10 == 0:
                
#     epoch_metrics = metrics.compute()
#     if epoch_metrics['f1'] > best_f1:
#         best_f1 = epoch_metrics['f1']
#         torch.save(model.state_dict(), pathlib.Path(args.store_dir).joinpath(f"{args.timm_model}_smoe_smnist.pth"))
#     # Adjust the learning rate
#     scheduler.step()

#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

