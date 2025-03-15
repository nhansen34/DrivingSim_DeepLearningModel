import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import sys
sys.path.append('../')
from model.pedestrian_model import PedestrianRiskClassifier
from dataset.data_loader_job import load_datasets
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

def train_pedestrian_classifier(args):
    pl.seed_everything(args.seed)
    
    train_loader, val_loader, test_loader = load_datasets(
        args.data_file, 
        batch_size=args.batch_size,
        use_pretrained=True
    )
    
    model = PedestrianRiskClassifier(
        num_classes=3,
        backbone=args.backbone,
        pretrained=True,
        learning_rate=args.learning_rate
    )
    
    if args.resume_from_pytorch_model:
        print(f"Loading weights from {args.resume_from_pytorch_model}")
        state_dict = torch.load(args.resume_from_pytorch_model, map_location='cpu')
        
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]  # Remove 'module.' prefix
                
            if k in model_state_dict and v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v
            else:
                print(f"Skipping parameter {k} due to shape mismatch or not found in model")
        
        model.load_state_dict(filtered_state_dict, strict=False)
        print(f"Successfully loaded weights from {args.resume_from_pytorch_model}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints'),
        filename='pedestrian-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        mode='min'
    )
    
    # Logger
    logger = TensorBoardLogger(args.output_dir, name="pedestrian_risk_classifier")
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=10
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    test_results = trainer.test(model, test_loader)
    
    visualize_results(model, test_loader, args.output_dir)
    
    return model, checkpoint_callback.best_model_path

def visualize_results(model, test_loader, output_dir):
    """Create visualizations for model performance"""
    
    model.eval()
    
    y_true = []
    y_pred = []
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    classes = ["Low Risk", "Medium Risk", "High Risk"]
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=classes,
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))
    
    metrics = report_df.iloc[0:3][['precision', 'recall', 'f1-score']]
    metrics.plot(kind='bar', figsize=(10, 6))
    plt.title('Performance Metrics by Risk Class')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_metrics.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train Pedestrian Risk Classifier')
    parser.add_argument('--data_file', type=str, default='pedestrian_risk_analysis.csv',
                        help='Path to the data CSV file')
    parser.add_argument('--output_dir', type=str, default='./pedestrian_results',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'efficientnet'],
                        help='Backbone architecture for the model')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--resume_from_pytorch_model', type=str, default=None,
                        help='Path to a PyTorch model checkpoint (not Lightning) to resume training from')
    
    args = parser.parse_args()
    
    model, best_model_path = train_pedestrian_classifier(args)
    
    print(f"Training completed. Best model saved at: {best_model_path}")

if __name__ == "__main__":
    main()
