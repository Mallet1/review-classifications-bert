"""
Training module for the healthcare app review classifier.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import f1_score, precision_score, recall_score
import logging

from classifier import HealthcareReviewClassifier
from config import MODEL_CONFIG

class ReviewTrainer:
    def __init__(self, model: HealthcareReviewClassifier, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize the trainer."""
        self.model = model.to(device)
        self.device = device
        self.config = MODEL_CONFIG
        self.criterion = nn.BCELoss()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer, 
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)

    def evaluate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Evaluate the model on validation data."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                predictions = (outputs > 0.5).float()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        metrics = {
            'f1': f1_score(all_labels, all_predictions, average='weighted'),
            'precision': precision_score(all_labels, all_predictions, average='weighted'),
            'recall': recall_score(all_labels, all_predictions, average='weighted')
        }
        
        return total_loss / len(val_loader), metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """Train the model for multiple epochs."""
        if num_epochs is None:
            num_epochs = self.config['num_epochs']
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        num_training_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=num_training_steps
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': []
        }
        
        best_val_f1 = 0
        
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            history['train_loss'].append(train_loss)
            
            # Evaluate
            val_loss, metrics = self.evaluate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_f1'].append(metrics['f1'])
            history['val_precision'].append(metrics['precision'])
            history['val_recall'].append(metrics['recall'])
            
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            self.logger.info(f"Val Loss: {val_loss:.4f}")
            self.logger.info(f"Val F1: {metrics['f1']:.4f}")
            
            # Save best model
            if metrics['f1'] > best_val_f1:
                best_val_f1 = metrics['f1']
                self.model.save_model('best_model.pt')
        
        return history 