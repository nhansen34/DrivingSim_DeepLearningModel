import torch
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
import sys
sys.path.append('../')
from dataset.load_dataset import load_datasets
from model.simple_model import get_model

class PedestrianRiskModel(pl.LightningModule):
    def __init__(self, learning_rate=0.001, num_classes=3):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.model = get_model(torch.device("cpu")) 
        
        self.test_preds = []
        self.test_labels = []
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        
        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean()
        
        self.test_preds.append(preds.cpu())
        self.test_labels.append(labels.cpu())
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss
    
    def on_test_epoch_end(self):
        all_preds = torch.cat(self.test_preds).numpy()
        all_labels = torch.cat(self.test_labels).numpy()
        
        cm = confusion_matrix(all_labels, all_preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=["Low", "Medium", "High"],
                    yticklabels=["Low", "Medium", "High"])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Pedestrian Risk Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        print("Confusion matrix saved as 'confusion_matrix.png'")
        
        self.test_preds = []
        self.test_labels = []
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def main():
    train_loader, val_loader, test_loader = load_datasets("pedestrian_risk_analysis.csv")
    
    num_classes = 3
    
    model = PedestrianRiskModel(learning_rate=0.001, num_classes=num_classes)
    
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
    
    logger = TensorBoardLogger("lightning_logs", name="dropout-pedestrian-risk")
    
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",  # Automatically detect GPU/CPU
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=10
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    trainer.test(model, test_loader)
    
    # Save the final model
    torch.save(model.state_dict(), "pedestrian_cnn.pth")
    print("Model saved!")


if __name__ == "__main__":
    main()