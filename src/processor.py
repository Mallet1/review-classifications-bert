"""
Data processing module for handling healthcare app reviews.
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from config import DATA_CONFIG, MODEL_CONFIG

class ReviewDataset(Dataset):
    """Custom Dataset for healthcare app reviews."""
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        self.labels = torch.FloatTensor(labels)
        
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item
    
    def __len__(self):
        return len(self.labels)

class ReviewProcessor:
    def __init__(self, tokenizer_name: str = "bert-base-uncased"):
        """Initialize the review processor with a tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.config = DATA_CONFIG
        self.model_config = MODEL_CONFIG

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load review data from a CSV file."""
        df = pd.read_csv(file_path)
        
        # Ensure required columns exist
        required_columns = ['content', 'sentiment']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in data: {missing_columns}")
        
        # Clean the data
        df = df.dropna(subset=['content'])
        df['content'] = df['content'].astype(str)
        
        # Filter reviews by length
        df['review_length'] = df['content'].str.len()
        df = df[
            (df['review_length'] >= self.config['min_review_length']) &
            (df['review_length'] <= self.config['max_review_length'])
        ]
        
        # Convert sentiment to multi-label format
        from config import LABELS
        for dimension in LABELS.values():
            for label in dimension.keys():
                df[label] = (df['sentiment'] == label).astype(int)
        
        return df

    def preprocess_text(self, text: str) -> str:
        """Preprocess a single review text."""
        # Basic preprocessing steps
        text = text.lower().strip()
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def create_data_loaders(self, train_data: pd.DataFrame, val_data: pd.DataFrame, 
                          test_data: pd.DataFrame, batch_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create PyTorch DataLoaders for training, validation, and test sets."""
        if batch_size is None:
            batch_size = self.model_config['batch_size']
        
        # Preprocess texts
        train_texts = [self.preprocess_text(text) for text in train_data['content']]
        val_texts = [self.preprocess_text(text) for text in val_data['content']]
        test_texts = [self.preprocess_text(text) for text in test_data['content']]
        
        # Get labels
        label_columns = self.get_label_columns()
        train_labels = train_data[label_columns].values
        val_labels = val_data[label_columns].values
        test_labels = test_data[label_columns].values
        
        # Create datasets
        train_dataset = ReviewDataset(
            train_texts, 
            train_labels, 
            self.tokenizer, 
            self.model_config['max_length']
        )
        val_dataset = ReviewDataset(
            val_texts, 
            val_labels, 
            self.tokenizer, 
            self.model_config['max_length']
        )
        test_dataset = ReviewDataset(
            test_texts, 
            test_labels, 
            self.tokenizer, 
            self.model_config['max_length']
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        return train_loader, val_loader, test_loader

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets."""
        # First split: separate test set
        train_val, test = train_test_split(
            df,
            test_size=self.config['test_split'],
            random_state=self.config['random_seed']
        )

        # Second split: separate train and validation sets
        train, val = train_test_split(
            train_val,
            test_size=self.config['val_split'] / (self.config['train_split'] + self.config['val_split']),
            random_state=self.config['random_seed']
        )

        return train, val, test

    @staticmethod
    def get_label_columns() -> List[str]:
        """Get the list of label columns based on the configuration."""
        from config import LABELS
        label_columns = []
        for dimension in LABELS.values():
            for label in dimension.keys():
                label_columns.append(label)
        return label_columns 