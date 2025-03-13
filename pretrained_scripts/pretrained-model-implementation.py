import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

class PretrainedPedestrianRiskModel(pl.LightningModule):
    def __init__(self, num_classes=3, learning_rate=1e-4, backbone_type="resnet50"):
        super().__init__()
        self.save_hyperparameters()
        
        # Select and load the pretrained backbone
        if backbone_type == "resnet50":
            self.backbone = models.resnet50(pretrained=True)
        elif backbone_type == "resnet101":
            self.backbone = models.resnet101(pretrained=True)
        elif backbone_type == "efficientnet":
            self.backbone = models.efficientnet_b0(pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
        
        # Freeze the early layers of the backbone
        # This helps prevent overfitting when training with a small dataset
        if backbone_type.startswith("resnet"):
            # Freeze the first 6 layers
            layers_to_freeze = [
                self.backbone.conv1,
                self.backbone.bn1,
                self.backbone.relu,
                self.backbone.maxpool,
                self.backbone.layer1,
                self.backbone.layer2
            ]
            for layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
        elif backbone_type == "efficientnet":
            # Freeze the first 3 blocks (approximately)
            for i, block in enumerate(self.backbone.features):
                if i < 3:
                    for param in block.parameters():
                        param.requires_grad = False
        
        # Get the feature dimension based on backbone type
        if backbone_type.startswith("resnet"):
            feature_dim = self.backbone.fc.in_features
            # Remove the classification head of the pretrained model
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif backbone_type == "efficientnet":
            feature_dim = self.backbone.classifier[1].in_features
            # Remove the classifier
            self.backbone = self.backbone.features
            # Add global pooling
            self.backbone = nn.Sequential(
                self.backbone,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
        
        # Create custom classification head for risk analysis
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features using the pretrained backbone
        features = self.backbone(x)
        
        # In ResNet, features come out as [batch_size, channels, 1, 1]
        # Need to flatten to [batch_size, channels]
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        
        # Apply classifier
        x = self.classifier(features)
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
        # Use separate learning rates for backbone and classifier
        # This is helpful when fine-tuning pretrained models
        backbone_params = []
        classifier_params = []
        
        # Separate parameters
        for name, param in self.named_parameters():
            if param.requires_grad:
                if name.startswith('backbone'):
                    backbone_params.append(param)
                else:
                    classifier_params.append(param)
        
        # Create optimizer with parameter groups
        optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': self.hparams.learning_rate / 10},  # Lower LR for pretrained
            {'params': classifier_params, 'lr': self.hparams.learning_rate}
        ])
        
        # Learning rate scheduler
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


def train_model(csv_path, batch_size=32, backbone_type="resnet50", max_epochs=30):
    # Import here to avoid circular imports
    from load_dataset import load_datasets
    
    # Load datasets with transformations for pretrained models
    train_loader, val_loader, test_loader = load_datasets(
        csv_path, 
        batch_size=batch_size, 
        use_pretrained=True
    )
    
    # Create model
    model = PretrainedPedestrianRiskModel(
        num_classes=3, 
        learning_rate=1e-4,
        backbone_type=backbone_type
    )
    
    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename=f"pedestrian-{backbone_type}-" + "{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min"
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=7,
        mode="min"
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # Create logger
    logger = TensorBoardLogger("lightning_logs", name=f"pedestrian-risk-{backbone_type}")
    
    # Initialize the Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",  # Automatically detect GPU/CPU
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    # Test the model
    trainer.test(model, test_loader)
    
    # Save the final model
    torch.save(model.state_dict(), f"pedestrian_{backbone_type}_model.pth")
    print("Model saved!")
    
    return model, trainer


if __name__ == "__main__":
    import argparse
    import numpy as np
    
    # Make results reproducible
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    parser = argparse.ArgumentParser(description='Train pedestrian risk model')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--backbone', type=str, default='resnet50', 
                       choices=['resnet50', 'resnet101', 'efficientnet'], 
                       help='Backbone architecture')
    parser.add_argument('--epochs', type=int, default=30, help='Maximum number of epochs')
    
    args = parser.parse_args()
    
    model, trainer = train_model(
        args.csv_path,
        batch_size=args.batch_size,
        backbone_type=args.backbone,
        max_epochs=args.epochs
    )
