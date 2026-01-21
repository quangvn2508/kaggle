import torch
from cnn import DigitCNN
import torch.nn as nn
from dataloader import train_loader, val_loader, test_loader
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

model = DigitCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(model, loader):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def validate(model, loader):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / len(loader.dataset)

    return avg_loss, accuracy

def test(model, loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            predictions.extend(preds.cpu().numpy())
    submission = pd.DataFrame({
        "ImageId": range(1, len(predictions) + 1),
        "Label": predictions
    })

    submission.to_csv("submission.csv", index=False)


NUM_EPOCHS = 20
for epoch in range(NUM_EPOCHS):
    train_loss = train(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)
    
    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.2f}%"
    )

test(model, test_loader)