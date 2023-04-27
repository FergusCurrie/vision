import os
import torch
import torch.nn as nn
from pascal_torch import get_pascal_torch
from transformers import ViTForImageClassification


class SETR(nn.Module):
    def __init__(self, vit, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.vit_model = vit.vit
        self.decoder_conv2d_1 = nn.Conv2d(in_channels=768, out_channels=256, kernel_size=(1, 1))
        self.decoder_conv2d_2 = nn.Conv2d(in_channels=256, out_channels=n_classes, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(num_features=256)
        self.relu = nn.ReLU()
        self.upsample = nn.UpsamplingBilinear2d(size=(224, 224))

    def decoder(self, encoder_outputs):
        # print(f"encoder out put shape = {(encoder_outputs.shape)}")  # torch.Size([16, 197, 768])
        encoder_outputs = encoder_outputs[:, :196, :]
        x = torch.reshape(encoder_outputs, shape=(batch_size, 768, 14, 14))
        x = self.decoder_conv2d_1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.decoder_conv2d_2(x)
        x = self.upsample(x)
        return x

    def forward(self, x):
        hidden_states = self.vit_model(x).last_hidden_state
        x = self.decoder(hidden_states)
        return x


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    # print(f"test size = {size}")
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            if X.shape[0] != batch_size:
                break
            X, y = X.to(device), y.to(device)
            y = torch.flatten(y, start_dim=1)
            pred = model(X)
            # print(pred.shape)
            pred = torch.flatten(pred, start_dim=2)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size * 16 * 224 * 224  # pixels per image, batch size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        if X.shape[0] != batch_size:
            break
        X, y = X.to(device), y.to(device)
        # y = y.view(batch_size, 21, image_size, image_size)
        y = torch.flatten(y, start_dim=1)

        # Compute prediction error
        pred = model(X)
        pred = torch.flatten(pred, start_dim=2)
        loss = loss_fn(input=pred, target=y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 20 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"batch={batch} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.set_visible_devices(physical_devices[0], "CPU")

# Get cpu, gpu or mps device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Load pretrained model
model_name_or_path = "google/vit-base-patch16-224-in21k"
vit_model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=21,  # model creates classification head with correct number of classes
    # id2label={str(i): c for i, c in enumerate(labels)},  # mapping between human readable labels and model's internal labels
    # label2id={c: str(i) for i, c in enumerate(labels)},
)

# Create setr model
setr = SETR(vit_model, 21)

# put it on gpu
setr = setr.to(device)


image_size = 224
batch_size = 16


train_ds, val_ds = get_pascal_torch(image_size=image_size, batch_size=batch_size)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(setr.parameters(), lr=1e-3)

test(val_ds, setr, loss_fn)
for epoch in range(5):
    print(f"epoch={epoch}")
    train(train_ds, setr, loss_fn, optimizer)
    test(val_ds, setr, loss_fn)
