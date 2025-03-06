import torch
import torch.optim as optim
import torch.nn as nn
from load_dataset import load_datasets 
from model import get_model  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader = load_datasets("pedestrian_risk_analysis.csv")

model = get_model(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss, correct = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    train_accuracy = correct / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

model.eval()
correct = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()

test_accuracy = correct / len(test_loader.dataset)
print(f"Test Accuracy: {test_accuracy:.4f}")

torch.save(model.state_dict(), "pedestrian_cnn.pth")
print("Model saved!")
