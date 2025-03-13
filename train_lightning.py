import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from load_dataset import load_datasets
from model import get_model

class PedestrianRiskModel(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        # Create the model using your existing get_model function
        # Note: We now let Lightning handle device placement
        self.model = get_model(torch.device("cpu"))  # We'll initialize on CPU, Lightning will move to appropriate device
        
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
        
        # Log metrics
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def main():
    # Load datasets using your existing function
    train_loader, val_loader, test_loader = load_datasets("pedestrian_risk_analysis.csv")
    
    # Create model
    model = PedestrianRiskModel(learning_rate=0.001)
    
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
    
    # Test the model
    trainer.test(model, test_loader)
    
    # Save the final model
    torch.save(model.state_dict(), "pedestrian_cnn.pth")
    print("Model saved!")


if __name__ == "__main__":
    main()