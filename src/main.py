"""
Main script for training the healthcare app review classifier.
"""

import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import json
from datetime import datetime

from processor import ReviewProcessor
from classifier import HealthcareReviewClassifier
from trainer import ReviewTrainer

def setup_logging(output_dir: Path):
    """Setup logging configuration."""
    log_file = output_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def save_metrics(metrics: dict, output_dir: Path):
    """Save training metrics to a JSON file."""
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train healthcare app review classifier')
    parser.add_argument('--data_path', type=str, default='data/labeled_data.csv', help='Path to the training data')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='Name of the pretrained model to use')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save model and results')
    
    args = parser.parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    logger = logging.getLogger(__name__)
    
    # Initialize data processor
    processor = ReviewProcessor(tokenizer_name=args.model_name)
    
    # Load and preprocess data
    logger.info(f"Loading data from {args.data_path}")
    df = processor.load_data(args.data_path)
    logger.info(f"Loaded {len(df)} reviews")
    
    # Split data
    train_data, val_data, test_data = processor.split_data(df)
    logger.info(f"Split data into {len(train_data)} train, {len(val_data)} validation, and {len(test_data)} test samples")
    
    # Create data loaders
    train_loader, val_loader, test_loader = processor.create_data_loaders(
        train_data, val_data, test_data, batch_size=args.batch_size
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = HealthcareReviewClassifier(model_name=args.model_name)
    
    # Initialize trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    trainer = ReviewTrainer(
        model=model,
        device=device
    )
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs
    )
    
    # Save final model
    model_path = output_dir / 'final_model.pt'
    model.save_model(str(model_path))
    logger.info(f"Saved final model to {model_path}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_loss, test_metrics = trainer.evaluate(test_loader)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test F1: {test_metrics['f1']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
    
    # Save metrics
    metrics = {
        'training_history': history,
        'test_metrics': test_metrics,
        'model_config': model.config,
        'training_config': {
            'model_name': args.model_name,
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'learning_rate': args.learning_rate,
            'device': device
        }
    }
    save_metrics(metrics, output_dir)
    
    logger.info("Training completed!")
    logger.info(f"Best validation F1 score: {max(history['val_f1']):.4f}")
    logger.info(f"Best validation precision: {max(history['val_precision']):.4f}")
    logger.info(f"Best validation recall: {max(history['val_recall']):.4f}")

if __name__ == "__main__":
    main() 