from load_dataset import load_datasets 
from model import get_model  
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, val_loader, test_loader = load_datasets("pedestrian_risk_analysis.csv")

model = torch.hub.load("pytorch/vision:v0.10.0", "deeplabv3_resnet101", weights=DeepLabV3_ResNet101_Weights.DEFAULT)
model.eval()

image = Image.open("DAVID-sim/m1596437/Images/Video_002/v002_0002.png")
transform = T.Compose([T.ToTensor()])
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(image_tensor)["out"][0]  # Get segmentation map

output_predictions = output.argmax(0)

plt.imshow(output_predictions, cmap="jet")
plt.show()