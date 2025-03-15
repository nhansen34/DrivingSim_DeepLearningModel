import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import pytorch_lightning as pl
import torchmetrics

class PedestrianRiskClassifier(pl.LightningModule):
    def __init__(self, num_classes=3, backbone='resnet50', pretrained=True, learning_rate=0.001):
        super(PedestrianRiskClassifier, self).__init__()
        self.save_hyperparameters()
        
        # Load object detection model pre-trained on COCO
        self.detector = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        
        # We only need to detect people (class 1 in COCO)
        self.person_class_idx = 1
        
        if backbone == 'resnet50':
            if pretrained:
                self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.resnet50(weights=None)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity() 
        elif backbone == 'resnet101':
            if pretrained:
                self.backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.resnet101(weights=None)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'efficientnet':
            if pretrained:
                self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.efficientnet_b0(weights=None)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self.density_features = nn.Sequential(
            nn.Linear(feature_dim + 3, 128), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        
        self.test_predictions = []
        self.test_targets = []
    
    def forward(self, x, detect_pedestrians=True):
        batch_size = x.size(0)
        
        if detect_pedestrians:
            pedestrian_features = torch.zeros(batch_size, 3, device=x.device)
            
            self.detector.eval()
            with torch.no_grad():
                for i in range(batch_size):
                    detections = self.detector([x[i]])
                    
                    boxes = detections[0]['boxes'][detections[0]['labels'] == self.person_class_idx]
                    scores = detections[0]['scores'][detections[0]['labels'] == self.person_class_idx]
                    
                    high_conf_idx = scores > 0.7
                    boxes = boxes[high_conf_idx]
                    
                    num_pedestrians = len(boxes)
                    
                    if num_pedestrians > 0:
                        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                        total_area = areas.sum().item()
                        avg_area = total_area / num_pedestrians
                        
                        img_area = x[i].shape[1] * x[i].shape[2]
                        
                        pedestrian_features[i, 0] = num_pedestrians / 10.0  # Normalized count
                        pedestrian_features[i, 1] = total_area / img_area  # Ratio of image covered
                        pedestrian_features[i, 2] = avg_area / img_area  # Average size
        else:
            pedestrian_features = torch.zeros(batch_size, 3, device=x.device)
        
        visual_features = self.backbone(x)
        
        # Combine with pedestrian detection features
        combined_features = torch.cat([visual_features, pedestrian_features], dim=1)
        
        # Final classification
        output = self.density_features(combined_features)
        
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        
        self.test_predictions.extend(preds.cpu().numpy())
        self.test_targets.extend(y.cpu().numpy())
        
        return loss
    
    def on_test_epoch_end(self):
        try:
            from sklearn.metrics import confusion_matrix, classification_report
            import numpy as np
            
            cm = confusion_matrix(self.test_targets, self.test_predictions)
            cr = classification_report(
                self.test_targets, 
                self.test_predictions,
                target_names=["Low", "Medium", "High"],
                output_dict=True
            )
            
            self.logger.experiment.add_figure(
                "Confusion Matrix",
                self.plot_confusion_matrix(cm, ["Low", "Medium", "High"]),
                self.current_epoch
            )
            
            for class_name in ["Low", "Medium", "High"]:
                metrics = cr[class_name]
                self.log(f"test_{class_name}_precision", metrics["precision"])
                self.log(f"test_{class_name}_recall", metrics["recall"])
                self.log(f"test_{class_name}_f1-score", metrics["f1-score"])
            
            self.test_predictions = []
            self.test_targets = []
            
        except ImportError:
            print("Install scikit-learn for detailed metrics")
    
    def plot_confusion_matrix(self, cm, class_names):
        """Returns a matplotlib figure containing the plotted confusion matrix."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        threshold = cm.max() / 2
        for i, j in np.ndindex(cm.shape):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
            
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        return figure
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            },
        }
