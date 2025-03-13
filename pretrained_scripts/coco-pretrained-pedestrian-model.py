import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl

class PretrainedPedestrianModel(pl.LightningModule):
    def __init__(self, num_classes=3, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        
        # Load a pre-trained model (e.g., Faster R-CNN with ResNet-50 backbone)
        # This model is pre-trained on COCO which includes pedestrian detection
        self.backbone = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        
        # Freeze the backbone layers
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Get the output features from the backbone
        # Use the ResNet backbone from the Faster R-CNN model
        self.feature_extractor = self.backbone.backbone.body
        
        # Create a feature pooling layer
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        # Create custom classification head for risk analysis
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),  # ResNet-50 outputs 2048 features
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Initialize classifier weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features using the pre-trained backbone
        # We need to extract features before the detection head
        features = self.feature_extractor(x)
        
        # Get the last layer features
        x = features['layer4']
        
        # Apply pooling
        x = self.pooling(x)
        x = x.flatten(1)
        
        # Apply classifier
        x = self.classifier(x)
        return x
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.functional.cross_entropy(outputs, labels)
        
        # Calculate accuracy
        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.functional.cross_entropy(outputs, labels)
        
        # Calculate accuracy
        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.functional.cross_entropy(outputs, labels)
        
        # Calculate accuracy
        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


# Alternative approach: Two-stage pipeline where we first detect pedestrians, then classify risk
class TwoStagePedestrianRiskModel:
    def __init__(self, risk_classifier_path=None):
        # Load pre-trained pedestrian detector
        self.detector = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.detector.eval()  # Set to evaluation mode
        
        # Load your fine-tuned risk classifier
        self.risk_classifier = ImprovedPedestrianCNN()
        if risk_classifier_path:
            self.risk_classifier.load_state_dict(torch.load(risk_classifier_path))
        self.risk_classifier.eval()
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector.to(self.device)
        self.risk_classifier.to(self.device)
    
    def predict(self, image):
        """
        Predict pedestrian risk from an image
        
        Args:
            image: PIL Image or tensor
            
        Returns:
            List of detections with risk scores
        """
        # Convert to tensor if needed
        if not isinstance(image, torch.Tensor):
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image = transform(image).unsqueeze(0)
        
        image = image.to(self.device)
        
        # Step 1: Detect pedestrians
        with torch.no_grad():
            detections = self.detector(image)
        
        results = []
        for detection in detections:
            # Filter for people class (class 1 in COCO)
            person_mask = detection['labels'] == 1
            
            if person_mask.sum() == 0:
                continue  # No people detected
                
            # Get bounding boxes for people
            boxes = detection['boxes'][person_mask]
            scores = detection['scores'][person_mask]
            
            # For each detected person
            for box, score in zip(boxes, scores):
                if score < 0.7:  # Confidence threshold
                    continue
                
                # Extract person region
                x1, y1, x2, y2 = box.int()
                person_crop = image[0, :, y1:y2, x1:x2]
                
                # Resize to expected input size
                if person_crop.numel() > 0:  # Check if crop is valid
                    person_crop = nn.functional.interpolate(
                        person_crop.unsqueeze(0), 
                        size=(128, 128),  # Assuming this is the expected input size
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)
                    
                    # Step 2: Classify risk
                    with torch.no_grad():
                        risk_outputs = self.risk_classifier(person_crop.unsqueeze(0))
                        risk_class = torch.argmax(risk_outputs, dim=1).item()
                        risk_score = torch.softmax(risk_outputs, dim=1)[0]
                    
                    results.append({
                        'box': box.cpu().numpy(),
                        'detection_score': score.item(),
                        'risk_class': risk_class,
                        'risk_scores': risk_score.cpu().numpy()
                    })
        
        return results


# Example usage:
def main():
    # Option 1: End-to-end model with transfer learning
    model = PretrainedPedestrianModel()
    
    # Option 2: Two-stage approach
    # two_stage_model = TwoStagePedestrianRiskModel("path_to_risk_classifier.pth")
    # from PIL import Image
    # image = Image.open("test_image.jpg")
    # results = two_stage_model.predict(image)
    # for r in results:
    #     print(f"Pedestrian detected with {r['detection_score']:.2f} confidence")
    #     print(f"Risk class: {r['risk_class']}, Risk scores: {r['risk_scores']}")

if __name__ == "__main__":
    main()
