"""
Multi-label classifier model for healthcare app reviews using BERT.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, List, Optional
from config import MODEL_CONFIG, LABELS

class HealthcareReviewClassifier(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased"):
        """Initialize the classifier model."""
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Calculate total number of labels
        self.num_labels = sum(len(labels) for labels in LABELS.values())
        
        # Add classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.num_labels),
            nn.Sigmoid()  # Multi-label classification
        )
        
        self.config = MODEL_CONFIG

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token output for classification
        pooled_output = outputs[1]
        
        # Pass through classifier
        logits = self.classifier(pooled_output)
        
        return logits

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                threshold: float = 0.5) -> torch.Tensor:
        """Make predictions with thresholding."""
        self.eval()
        with torch.no_grad():
            logits = self(input_ids, attention_mask)
            predictions = (logits > threshold).float()
        return predictions

    def get_label_names(self) -> List[str]:
        """Get the list of label names in order."""
        label_names = []
        for dimension in LABELS.values():
            for label in dimension.keys():
                label_names.append(label)
        return label_names

    def save_model(self, path: str):
        """Save the model to disk."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'label_names': self.get_label_names()
        }, path)

    @classmethod
    def load_model(cls, path: str, model_name: str = "bert-base-uncased") -> 'HealthcareReviewClassifier':
        """Load the model from disk."""
        checkpoint = torch.load(path)
        model = cls(model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model 