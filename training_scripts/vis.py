import os
import torch
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pytorch_lightning import LightningModule
from typing import Dict, Any, List, Optional, Type

class ModelVisualizer:
    """Helper class to load Lightning checkpoints and visualize model data"""
    
    def __init__(self, checkpoint_path: str, output_dir: str = "visualization_output", model_class: Optional[Type[LightningModule]] = None):
        """
        Initialize visualizer with a checkpoint path and optionally a model class
        
        Args:
            checkpoint_path: Path to the .ckpt file
            output_dir: Directory to save visualizations
            model_class: LightningModule class to load the checkpoint with (optional)
        """
        self.checkpoint_path = checkpoint_path
        self.model_class = model_class
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Extract checkpoint filename for naming visualizations
        self.checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        
        # Load the checkpoint
        self.checkpoint = self._load_checkpoint()
        
    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load the checkpoint file and return the checkpoint dictionary"""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")
            
        checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
        print(f"Loaded checkpoint from {self.checkpoint_path}")
        return checkpoint
    
    def _save_text_summary(self, content: str, filename: str):
        """Save text content to a file"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Saved summary to {filepath}")
    
    def get_model(self) -> Optional[LightningModule]:
        """Load the model from the checkpoint if model_class is provided"""
        if self.model_class is None:
            print("No model class provided. Can only visualize checkpoint metadata.")
            return None
            
        model = self.model_class.load_from_checkpoint(self.checkpoint_path)
        return model
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Extract hyperparameters from the checkpoint"""
        hyperparams = {}
        
        if 'hyper_parameters' in self.checkpoint:
            hyperparams = self.checkpoint['hyper_parameters']
        else:
            # Try to find hyperparameters in other possible locations
            for key in ['hparams', 'kwargs']:
                if key in self.checkpoint:
                    hyperparams = self.checkpoint[key]
                    break
                    
        return hyperparams
    
    def get_checkpoint_metadata(self) -> Dict[str, Any]:
        """Extract metadata from the checkpoint excluding model weights"""
        metadata = {}
        
        # Copy checkpoint dict excluding state_dict and optimizer_states
        for key, value in self.checkpoint.items():
            if key not in ['state_dict', 'optimizer_states']:
                metadata[key] = value
                
        return metadata
    
    def get_training_metrics(self) -> Optional[Dict[str, Any]]:
        """Extract training metrics if available in the checkpoint"""
        # Some checkpoints store metrics history
        metrics = {}
        
        for key in ['metrics', 'logged_metrics', 'training_metrics', 'callbacks']:
            if key in self.checkpoint:
                metrics[key] = self.checkpoint[key]
                
        # LightningModule callbacks like ModelCheckpoint might store best metrics
        if 'callbacks' in self.checkpoint:
            callbacks = self.checkpoint['callbacks']
            if 'ModelCheckpoint' in callbacks:
                metrics['best_model_metrics'] = callbacks['ModelCheckpoint']
                
        return metrics if metrics else None
    
    def visualize_weights_distribution(self, figsize=(12, 8), save=True):
        """Visualize the distribution of model weights"""
        if 'state_dict' not in self.checkpoint:
            print("No state_dict found in checkpoint")
            return
            
        state_dict = self.checkpoint['state_dict']
        
        # Get a list of all parameter tensors
        weights = []
        labels = []

        for name, param in state_dict.items():
            if param.dim() > 0:  # Skip scalar parameters
                weights.append(param.flatten().numpy())
                labels.append(name)
        
        # Create matplotlib plots
        fig, axes = plt.subplots(min(len(weights), 9), 1, figsize=figsize)
        if len(weights) == 1:
            axes = [axes]  # Make axes iterable if only one layer
        
        for i, (w, name) in enumerate(zip(weights[:9], labels[:9])):  # Show at most 9 layers
            axes[i].hist(w, bins=50, alpha=0.7)
            axes[i].set_title(f'Layer: {name}')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save the matplotlib figure
        if save:
            matplotlib_path = os.path.join(self.output_dir, f"{self.checkpoint_name}_weight_distributions.png")
            fig.savefig(matplotlib_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved weight distribution plot to {matplotlib_path}")
        else:
            plt.show()
        
        # Create an interactive plotly figure for all weights combined
        all_weights = np.concatenate([w for w in weights])
        fig = px.histogram(all_weights, nbins=100, title='Distribution of All Model Weights')
        fig.update_layout(xaxis_title='Weight Value', yaxis_title='Frequency')
        
        # Save the plotly figure
        if save:
            plotly_path = os.path.join(self.output_dir, f"{self.checkpoint_name}_all_weights_histogram.html")
            fig.write_html(plotly_path)
            print(f"Saved all weights histogram to {plotly_path}")
        else:
            fig.show()
    
    def visualize_layer_norms(self, save=True):
        """Visualize the L1 and L2 norms of each layer"""
        if 'state_dict' not in self.checkpoint:
            print("No state_dict found in checkpoint")
            return
            
        state_dict = self.checkpoint['state_dict']
        
        # Calculate norms for each layer
        layer_names = []
        l1_norms = []
        l2_norms = []
        
        for name, param in state_dict.items():
            if param.dim() > 0:  # Skip scalar parameters
                param_np = param.flatten().numpy()
                layer_names.append(name)
                l1_norms.append(np.sum(np.abs(param_np)))
                l2_norms.append(np.sqrt(np.sum(param_np**2)))
        
        # Create dataframe for easier plotting
        df = pd.DataFrame({
            'layer': layer_names,
            'l1_norm': l1_norms,
            'l2_norm': l2_norms
        })
        
        # Save data to CSV
        if save:
            csv_path = os.path.join(self.output_dir, f"{self.checkpoint_name}_layer_norms.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved layer norms data to {csv_path}")
        
        # Plot with plotly
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['layer'], y=df['l1_norm'], name='L1 Norm'))
        fig.add_trace(go.Bar(x=df['layer'], y=df['l2_norm'], name='L2 Norm'))
        
        fig.update_layout(
            title='Layer Norms',
            xaxis_title='Layer',
            yaxis_title='Norm Value',
            barmode='group'
        )
        
        # Save the plotly figure
        if save:
            plotly_path = os.path.join(self.output_dir, f"{self.checkpoint_name}_layer_norms.html")
            fig.write_html(plotly_path)
            print(f"Saved layer norms visualization to {plotly_path}")
        else:
            fig.show()
    
    def plot_hyperparameter_importance(self, save=True):
        """
        Simulated visualization of hyperparameter settings
        (Actual importance would require multiple checkpoints)
        """
        hyperparams = self.get_hyperparameters()
        
        if not hyperparams:
            print("No hyperparameters found in checkpoint")
            return
            
        # Filter and convert hyperparameters to numeric if possible
        numeric_params = {}
        for k, v in hyperparams.items():
            try:
                if isinstance(v, (int, float, bool)):
                    numeric_params[k] = float(v)
                elif isinstance(v, str) and v.replace('.', '', 1).isdigit():
                    numeric_params[k] = float(v)
            except:
                pass
                
        if not numeric_params:
            print("No numeric hyperparameters found to visualize")
            return
        
        # Save data to CSV
        if save:
            df = pd.DataFrame({
                'parameter': list(numeric_params.keys()),
                'value': list(numeric_params.values())
            })
            csv_path = os.path.join(self.output_dir, f"{self.checkpoint_name}_hyperparameters.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved hyperparameters data to {csv_path}")
            
        # Create bar chart of hyperparameter values
        fig = px.bar(
            x=list(numeric_params.keys()),
            y=list(numeric_params.values()),
            title='Hyperparameter Values'
        )
        fig.update_layout(xaxis_title='Hyperparameter', yaxis_title='Value')
        
        # Save the plotly figure
        if save:
            plotly_path = os.path.join(self.output_dir, f"{self.checkpoint_name}_hyperparameters.html")
            fig.write_html(plotly_path)
            print(f"Saved hyperparameters visualization to {plotly_path}")
        else:
            fig.show()
    
    def generate_checkpoint_summary(self, save=True):
        """Generate a summary of the checkpoint contents"""
        summary = [f"Checkpoint Summary for: {self.checkpoint_path}"]
        summary.append("-" * 50)
        
        # Add top-level keys
        summary.append("Top-level keys in checkpoint:")
        for key in self.checkpoint:
            summary.append(f"- {key}")
        summary.append("")
        
        # Add hyperparameters
        hyperparams = self.get_hyperparameters()
        if hyperparams:
            summary.append("Hyperparameters:")
            for k, v in hyperparams.items():
                summary.append(f"- {k}: {v}")
            summary.append("")
            
        # Add model architecture summary if model is loaded
        model = self.get_model()
        if model:
            summary.append("Model Architecture:")
            summary.append(str(model))
            summary.append("")
            
        # Add metrics if available
        metrics = self.get_training_metrics()
        if metrics:
            summary.append("Training Metrics:")
            for k, v in metrics.items():
                summary.append(f"- {k}: {v}")
                
        summary.append("-" * 50)
        
        # Convert list to string
        summary_text = "\n".join(summary)
        
        # Print or save summary
        if save:
            summary_path = os.path.join(self.output_dir, f"{self.checkpoint_name}_summary.txt")
            self._save_text_summary(summary_text, f"{self.checkpoint_name}_summary.txt")
        else:
            print(summary_text)
            
        return summary_text
        
    def run_all_visualizations(self):
        """Run all visualization methods and save results"""
        print(f"Saving all visualizations to {self.output_dir}")
        
        # Generate text summary
        self.generate_checkpoint_summary(save=True)
        
        # Generate visualizations
        self.visualize_weights_distribution(save=True)
        self.visualize_layer_norms(save=True)
        self.plot_hyperparameter_importance(save=True)
        
        print(f"All visualizations saved to {self.output_dir}")


# Example usage
if __name__ == "__main__":
    # For this example, create a custom class that matches your model architecture
    # You would need to define this class based on your model architecture
    
    """
    # Example model class definition (you should replace this with your actual model)
    class MyModel(LightningModule):
        def __init__(self, hidden_dim=128, learning_rate=0.001):
            super().__init__()
            self.save_hyperparameters()
            # Define your model architecture here
    """
    
    # Path to your checkpoint file
    checkpoint_path = "pretrained_scripts/pedestrian_results/checkpoints/pedestrian-epoch=03-val_loss=0.26-val_acc=0.90.ckpt"
    
    # Output directory for visualizations
    output_dir = "checkpoint_visualizations2"
    
    # Option 1: Use without model class (only metadata analysis)
    visualizer = ModelVisualizer(checkpoint_path, output_dir=output_dir)
    
    # Option 2: Use with model class (for more detailed analysis)
    # visualizer = ModelVisualizer(checkpoint_path, output_dir=output_dir, model_class=MyModel)
    
    # Run all visualizations and save results
    visualizer.run_all_visualizations()
    
    # Or run individual visualizations
    # visualizer.generate_checkpoint_summary(save=True)
    # visualizer.visualize_weights_distribution(save=True)
    # visualizer.visualize_layer_norms(save=True)
    # visualizer.plot_hyperparameter_importance(save=True)