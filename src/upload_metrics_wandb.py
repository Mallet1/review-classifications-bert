"""
Script to upload BERT model metrics and graphs to Weights & Biases.
"""

import wandb
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

def plot_training_history(metrics):
    """Create training history plots."""
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss curves
    ax1.plot(metrics['training_history']['train_loss'], label='Train Loss')
    ax1.plot(metrics['training_history']['val_loss'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot F1, Precision, and Recall
    ax2.plot(metrics['training_history']['val_f1'], label='F1')
    ax2.plot(metrics['training_history']['val_precision'], label='Precision')
    ax2.plot(metrics['training_history']['val_recall'], label='Recall')
    ax2.set_title('Validation Metrics')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def upload_to_wandb(metrics_path, output_dir):
    """Upload metrics and plots to Weights & Biases."""
    # Initialize wandb
    wandb.init(
        project="bert-sam",
        config={
            "model": "BERT-Base-Uncased",
            "dataset": "Healthcare App Reviews",
            "architecture": "BERT with Multi-Label Classification"
        }
    )
    
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Log model configuration
    wandb.log({
        "model_config": metrics['model_config'],
        "training_config": metrics['training_config']
    })
    
    # Log final test metrics
    wandb.log({
        "test_f1": metrics['test_metrics']['f1'],
        "test_precision": metrics['test_metrics']['precision'],
        "test_recall": metrics['test_metrics']['recall']
    })
    
    # Log training history
    for epoch in range(len(metrics['training_history']['train_loss'])):
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": metrics['training_history']['train_loss'][epoch],
            "val_loss": metrics['training_history']['val_loss'][epoch],
            "val_f1": metrics['training_history']['val_f1'][epoch],
            "val_precision": metrics['training_history']['val_precision'][epoch],
            "val_recall": metrics['training_history']['val_recall'][epoch]
        })
    
    # Create and log plots
    fig = plot_training_history(metrics)
    wandb.log({"training_plots": wandb.Image(fig)})
    plt.close()
    
    # Log model artifacts
    model_path = os.path.join(output_dir, "model.pt")
    if os.path.exists(model_path):
        artifact = wandb.Artifact(
            name=f"bert-model-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            type="model",
            description="Trained BERT model for healthcare app review classification"
        )
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
    
    # Finish the run
    wandb.finish()

if __name__ == "__main__":
    # Get the most recent output directory
    output_dir = "output"
    if not os.path.exists(output_dir):
        raise ValueError("Output directory not found")
    
    # Get the most recent run directory
    run_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    if not run_dirs:
        raise ValueError("No run directories found")
    
    latest_run = max(run_dirs, key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
    run_path = os.path.join(output_dir, latest_run)
    
    # Path to metrics file
    metrics_path = os.path.join(run_path, "metrics.json")
    if not os.path.exists(metrics_path):
        raise ValueError(f"Metrics file not found at {metrics_path}")
    
    # Upload to wandb
    upload_to_wandb(metrics_path, run_path) 