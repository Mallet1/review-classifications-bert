"""
Script to analyze metrics for data_quality and data_control labels from the labeled data.
"""

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def analyze_label_metrics():
    # Read the labeled data
    df = pd.read_csv('data/labeled_data.csv')
    
    # Define label mappings (CSV format -> code format)
    label_mappings = {
        'data quality': 'data_quality',
        'data control': 'data_control',
        'ethicality': 'ethicality',
        'competence': 'competence',
        'reliability': 'reliability',
        'support': 'support',
        'risk': 'risk'
    }
    
    # Convert sentiment column to multi-label format
    all_labels = list(label_mappings.values())
    
    # Create binary columns for each label
    for csv_label, code_label in label_mappings.items():
        df[code_label] = (df['sentiment'].str.lower() == csv_label.lower()).astype(int)
    
    # Calculate metrics for each label
    print("\nMetrics for each label:")
    print("-" * 50)
    for label in all_labels:
        # Count positive and negative examples
        pos_count = df[label].sum()
        neg_count = len(df) - pos_count
        
        print(f"\n{label.upper()}:")
        print(f"Positive examples: {pos_count}")
        print(f"Negative examples: {neg_count}")
        
        # Only calculate metrics if we have positive examples
        if pos_count > 0:
            precision = precision_score(df[label], df[label])
            recall = recall_score(df[label], df[label])
            f1 = f1_score(df[label], df[label])
            
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
        else:
            print("No positive examples found for this label")
        print("-" * 30)
    
    # Print label distribution
    print("\nLabel Distribution:")
    print("-" * 50)
    label_counts = df['sentiment'].value_counts()
    for csv_label, code_label in label_mappings.items():
        count = label_counts.get(csv_label, 0)
        percentage = (count / len(df)) * 100
        print(f"{csv_label}: {count} ({percentage:.2f}%)")

if __name__ == "__main__":
    analyze_label_metrics() 