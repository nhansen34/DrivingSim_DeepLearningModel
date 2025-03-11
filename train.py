import torch
import torch.optim as optim
import torch.nn as nn
from load_dataset import load_datasets 
from model import get_model  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader = load_datasets("pedestrian_risk_analysis.csv")

model = get_model(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

_loss_plot_test = []
num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    total_loss, correct, total_samples = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    avg_train_loss = total_loss / len(train_loader)
    train_accuracy = correct / total_samples

    model.eval()  
    val_loss, val_correct, val_samples = 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            val_correct += (outputs.argmax(dim=1) == labels).sum().item()
            val_samples += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = val_correct / val_samples
    _loss_plot_test.append(avg_val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

torch.save(model.state_dict(), "pedestrian_cnn.pth")
print("Model saved!")
