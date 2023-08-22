"""
https://pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html#torchvision.models.ConvNeXt_Tiny_Weights
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# from data_loader.cifar import get_cifar

BATCH_SIZE = 16
CIFAR = Path("/home/fergus/data/cifar")


def get_torchvision_cifar() -> tuple[torch.utils.data.dataloader.DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize images
        ]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root=str(CIFAR), transform=transform, train=True, download=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=str(CIFAR), transform=transform, train=False, download=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    return train_loader, test_loader


class SimClr(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet101()

    def forward(self, x):
        x = self.resnet(x)
        return x


def evals():
    # Disable gradient computation and reduce memory consumption.
    running_vloss = 0.0
    model.eval()
    with torch.no_grad():
        for i, (vinputs, vlabels) in enumerate(test_ds):
            if i > len(test_ds):
                break
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            voutputs = model(vinputs)
            vloss = ce(voutputs, vlabels)
            running_vloss += vloss
    return running_vloss / (i + 1)


def step():
    model.train(True)
    images, labels = next(iter_train_ds)
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = ce(outputs, labels)
    loss.backward()
    optimizer.step()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimClr()
model.to(device)


ce = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


train_ds, test_ds = get_torchvision_cifar()
iter_train_ds = iter(train_ds)
iter_test_ds = iter(test_ds)

for step_index in range(5000):
    current_loss = evals()
    print(f"Step {step_index + 1}, loss: {current_loss}")
    step()
