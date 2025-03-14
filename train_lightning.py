import torch
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from load_dataset import load_datasets
from simple_model import get_model

class PedestrianRiskModel(pl.LightningModule):
    def __init__(self, learning_rate=0.001, num_classes=3):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        # Create the model using your existing get_model function
        # Note: We now let Lightning handle device placement
        self.model = get_model(torch.device("cpu"))  # Lightning will move to appropriate device
        
        # For confusion matrix
        self.test_preds = []
        self.test_labels = []
        
    def forward(self, x):
        # Forward pass through your model
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
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
        loss = F.cross_entropy(outputs, labels)
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
        loss = F.cross_entropy(outputs, labels)
        
        # Calculate accuracy
        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean()
        
        # Store predictions and labels for confusion matrix
        self.test_preds.append(preds.cpu())
        self.test_labels.append(labels.cpu())
        
        # Log metrics
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss
    
    def on_test_epoch_end(self):
        # Concatenate all predictions and labels
        all_preds = torch.cat(self.test_preds).numpy()
        all_labels = torch.cat(self.test_labels).numpy()
        
        # Create and save confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=["Low", "Medium", "High"],
                    yticklabels=["Low", "Medium", "High"])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Pedestrian Risk Confusion Matrix')
        plt.tight_layout()
        
        # Save the confusion matrix
        plt.savefig('confusion_matrix.png')
        print("Confusion matrix saved as 'confusion_matrix.png'")
        
        # Reset lists for next test epoch
        self.test_preds = []
        self.test_labels = []
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def main():
    # Load datasets using your existing function
    train_loader, val_loader, test_loader = load_datasets("pedestrian_risk_analysis.csv")
    
    # Set number of classes to 3 for low, medium, high
    num_classes = 3
    
    # Create model
    model = PedestrianRiskModel(learning_rate=0.001, num_classes=num_classes)
    
    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="pedestrian-cnn-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min"
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min"
    )
    
    # Create logger
    logger = TensorBoardLogger("lightning_logs", name="dropout-pedestrian-risk")
    
    # Initialize the Trainer
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",  # Automatically detect GPU/CPU
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=10
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    # Test the model and generate confusion matrix
    trainer.test(model, test_loader)
    
    # Save the final model
    torch.save(model.state_dict(), "pedestrian_cnn.pth")
    print("Model saved!")


if __name__ == "__main__":
    main()