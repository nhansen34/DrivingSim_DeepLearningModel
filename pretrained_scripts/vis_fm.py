import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
from torchvision.utils import draw_bounding_boxes
from PIL import Image
import cv2
from sklearn.manifold import TSNE
import seaborn as sns
from pytorch_lightning import Trainer
from torch.nn import functional as F
from pedestrian_model import PedestrianRiskClassifier

def load_trained_model(checkpoint_path):
    """Load a trained PedestrianRiskClassifier model from a checkpoint."""
    model = PedestrianRiskClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model

def visualize_pedestrian_detection(model, image_path, threshold=0.7):
    """
    Visualize pedestrian detection results on an image.
    
    Args:
        model: Trained PedestrianRiskClassifier model
        image_path: Path to the image to visualize
        threshold: Confidence threshold for detections
    
    Returns:
        Matplotlib figure with visualization
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0)
    
     # Move tensor to the same device as model
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    # Run pedestrian detection
    model.detector.eval()
    with torch.no_grad():
        detections = model.detector(img_tensor)
        
    # Extract pedestrian detections
    boxes = detections[0]['boxes'][detections[0]['labels'] == model.person_class_idx]
    scores = detections[0]['scores'][detections[0]['labels'] == model.person_class_idx]
    
    # Filter by confidence threshold
    high_conf_idx = scores > threshold
    boxes = boxes[high_conf_idx]
    scores = scores[high_conf_idx]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Convert tensor to numpy for display
    img_np = img_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    
    # Normalize image for display
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    
    # Display image
    ax.imshow(img_np)
    
    # Draw bounding boxes
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box.int().tolist()
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                             fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1, f"Person: {score:.2f}", 
                bbox=dict(facecolor='yellow', alpha=0.5))
    
    ax.set_title(f"Pedestrian Detection Results - {len(boxes)} pedestrians detected")
    ax.axis('off')
    
    return fig

def predict_risk_level(model, image_path):
    """
    Predict risk level for an image and visualize the result.
    
    Args:
        model: Trained PedestrianRiskClassifier model
        image_path: Path to the image to analyze
    
    Returns:
        Matplotlib figure with visualization and risk level
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0)

     # Move tensor to the same device as model
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    # Run prediction
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()  # Changed to use probs instead of logits
    
    # Risk level mapping
    risk_levels = ["Low", "Medium", "High"]
    risk_level = risk_levels[prediction]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display image
    img_np = img_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    # Normalize image for display
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    ax1.imshow(img_np)
    ax1.set_title("Input Image")
    ax1.axis('off')
    
    # Display risk level probabilities
    probs_np = probs.cpu().squeeze(0).numpy()
    bars = ax2.bar(risk_levels, probs_np, color=['green', 'yellow', 'red'])
    ax2.set_title(f"Predicted Risk Level: {risk_level}")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Probability")
    
    # Highlight the predicted class
    bars[prediction].set_edgecolor('blue')
    bars[prediction].set_linewidth(2)
    
    # Add text with probability values
    for i, prob in enumerate(probs_np):
        ax2.text(i, prob + 0.05, f"{prob:.2f}", ha='center')
    
    plt.tight_layout()
    return fig, risk_level, probs_np

def visualize_feature_maps(model, image_path, num_maps=16):
    """
    Visualize feature maps from the backbone network.
    
    Args:
        model: Trained PedestrianRiskClassifier model
        image_path: Path to the image to analyze
        num_maps: Number of feature maps to visualize
    
    Returns:
        Matplotlib figure with visualization of feature maps
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0)
    
     # Move tensor to the same device as model
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    # Extract features from the backbone
    backbone = model.backbone
    backbone.eval()
    
    # Create hooks to capture feature maps
    feature_maps = []
    
    def hook_fn(module, input, output):
        feature_maps.append(output)
    
    # Register hooks for all convolutional layers
    hooks = []
    for name, module in backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Forward pass
    with torch.no_grad():
        _ = backbone(img_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Select a subset of feature maps
    if len(feature_maps) > 0:
        # Get the last conv layer feature maps for more interesting visualizations
        last_layer_features = feature_maps[-1][0]
        
        # Create figure
        rows = int(np.ceil(np.sqrt(num_maps)))
        fig, axs = plt.subplots(rows, rows, figsize=(12, 12))
        axs = axs.flatten()
        
        # Display feature maps
        num_maps = min(num_maps, last_layer_features.size(0))
        
        for i in range(num_maps):
            # Normalize feature map
            fmap = last_layer_features[i].detach().cpu().numpy()
            if fmap.max() > fmap.min():  # Avoid division by zero
                fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min())
            
            # Display feature map
            axs[i].imshow(fmap, cmap='viridis')
            axs[i].set_title(f"Feature Map {i+1}")
            axs[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_maps, len(axs)):
            axs[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    return None

def visualize_activation_heatmap(model, image_path):
    """
    Visualize activation heatmap using Grad-CAM technique.
    
    Args:
        model: Trained PedestrianRiskClassifier model
        image_path: Path to the image to analyze
    
    Returns:
        Matplotlib figure with visualization of activation heatmap
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0)

     # Move tensor to the same device as model
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    # Make sure we're using the device the model is on
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    # Extract the last convolutional layer
    target_layer = None
    for name, module in model.backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    
    if target_layer is None:
        print("No convolutional layer found in the backbone")
        return None
    
    # Create Grad-CAM
    gradients = []
    activations = []
    
    def hook_fn_activation(module, input, output):
        activations.append(output.detach())
    
    def hook_fn_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())
    
    # Register hooks
    activation_hook = target_layer.register_forward_hook(hook_fn_activation)
    gradient_hook = target_layer.register_backward_hook(hook_fn_gradient)
    
    # Forward pass
    model.eval()
    logits = model(img_tensor)
    
    # Get the class with the highest probability
    probs = F.softmax(logits, dim=1)
    pred_class = probs.argmax(dim=1).item()
    
    # Backward pass for the predicted class
    model.zero_grad()
    one_hot = torch.zeros_like(logits)
    one_hot[0, pred_class] = 1
    logits.backward(gradient=one_hot)
    
    # Remove hooks
    activation_hook.remove()
    gradient_hook.remove()
    
    # Calculate Grad-CAM
    if len(gradients) > 0 and len(activations) > 0:
        gradients = gradients[0]
        activations = activations[0]
        
        # Calculate weights
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Create heatmap
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # ReLU to keep only positive influences
        
        # Normalize CAM
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize CAM to input image size
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Convert to numpy for visualization
        cam = cam.squeeze().cpu().numpy()
        
        # Convert to RGB heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Convert input image to numpy
        img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        img_np = np.uint8(255 * img_np)
        
        # Superimpose heatmap on input image
        superimposed = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Display original image
        ax1.imshow(img_np)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Display heatmap
        ax2.imshow(heatmap)
        ax2.set_title("Activation Heatmap")
        ax2.axis('off')
        
        # Display superimposed image
        ax3.imshow(superimposed)
        risk_levels = ["Low", "Medium", "High"]
        ax3.set_title(f"Superimposed Image - {risk_levels[pred_class]} Risk")
        ax3.axis('off')
        
        plt.tight_layout()
        return fig
    
    print("Failed to generate Grad-CAM: No gradients or activations captured")
    return None

def visualize_feature_space(model, dataloader, num_samples=200):
    """
    Visualize the feature space using t-SNE.
    
    Args:
        model: Trained PedestrianRiskClassifier model
        dataloader: DataLoader with samples to visualize
        num_samples: Number of samples to use for visualization
    
    Returns:
        Matplotlib figure with visualization of feature space
    """
    # Extract features and labels
    features = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            if len(features) * data.size(0) >= num_samples:
                break
                
            # Move data to the same device as the model
            device = next(model.parameters()).device
            data = data.to(device)
            
            # Extract features from the backbone
            batch_features = model.backbone(data).cpu().numpy()
            batch_labels = target.cpu().numpy()
            
            features.append(batch_features)
            labels.append(batch_labels)
    
    # Concatenate features and labels
    if not features:
        print("No features extracted. Check if the dataloader is empty.")
        return None
        
    features = np.vstack(features)[:num_samples]
    labels = np.concatenate(labels)[:num_samples]
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
    
    # Flatten features if they're not already flat
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)
    
    features_2d = tsne.fit_transform(features)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define color palette for risk levels
    palette = {0: 'green', 1: 'orange', 2: 'red'}
    risk_levels = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
    
    # Create scatter plot
    for label in np.unique(labels):
        idx = labels == label
        ax.scatter(features_2d[idx, 0], features_2d[idx, 1], 
                   label=risk_levels.get(label, f"Class {label}"),
                   color=palette.get(label, None),
                   alpha=0.7)
    
    ax.set_title("t-SNE Visualization of Feature Space")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend()
    
    return fig

if __name__ == "__main__":
    # Load the trained model
    model_path = "pedestrian_results/checkpoints/pedestrian-epoch=03-val_loss=0.26-val_acc=0.90.ckpt"
    model = load_trained_model(model_path)
    
    # Create output directory
    import os
    output_dir = "visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # List of images to process
    image_paths = [
        "../DAVID-sim/m1596437/Images/Video_002/v002_0044.png",
        "../DAVID-sim/m1596437/Images/Video_002/v002_0222.png",
        "../DAVID-sim/m1596437/Images/Video_002/v002_0100.png"
    ]
    
    for idx, image_path in enumerate(image_paths):
        print(f"Processing image {idx+1}/{len(image_paths)}: {image_path}")
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Create subdirectory for this image
        img_output_dir = f"{output_dir}/{img_name}"
        os.makedirs(img_output_dir, exist_ok=True)
        
        # 1. Generate pedestrian detection visualization
        print("  Generating pedestrian detection visualization...")
        fig1 = visualize_pedestrian_detection(model, image_path)
        if fig1:
            fig1.savefig(f"{img_output_dir}/pedestrian_detection.png", dpi=300, bbox_inches='tight')
            plt.close(fig1)
        
        # 2. Generate risk prediction visualization
        print("  Generating risk prediction visualization...")
        fig2, risk_level, probs = predict_risk_level(model, image_path)
        if fig2:
            fig2.savefig(f"{img_output_dir}/risk_prediction.png", dpi=300, bbox_inches='tight')
            plt.close(fig2)
        
        # 3. Generate feature maps visualization
        print("  Generating feature maps visualization...")
        fig3 = visualize_feature_maps(model, image_path)
        if fig3:
            fig3.savefig(f"{img_output_dir}/feature_maps.png", dpi=300, bbox_inches='tight')
            plt.close(fig3)
        
        # 4. Generate activation heatmap visualization
        print("  Generating activation heatmap visualization...")
        fig4 = visualize_activation_heatmap(model, image_path)
        if fig4:
            fig4.savefig(f"{img_output_dir}/activation_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close(fig4)
        
        print(f"  All visualizations for {img_name} saved to {img_output_dir}")
    
    # Check if a dataloader is available for feature space visualization
    # (You would need to uncomment and modify this code to use your actual dataloader)
    """
    print("Generating feature space visualization...")
    from torch.utils.data import DataLoader
    from your_dataset_module import YourDataset
    
    # Create a dataset and dataloader (replace with your actual dataset)
    test_dataset = YourDataset(...)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Generate and save feature space visualization
    fig5 = visualize_feature_space(model, test_loader)
    if fig5:
        fig5.savefig(f"{output_dir}/feature_space.png", dpi=300, bbox_inches='tight')
        plt.close(fig5)
    """
    
    print(f"All visualizations saved to {output_dir} directory")