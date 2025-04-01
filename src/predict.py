"""
Script to load the trained BERT model and make predictions on new reviews.
"""

import torch
import json
import os
from datetime import datetime
from processor import ReviewProcessor
from classifier import HealthcareReviewClassifier
from config import MODEL_CONFIG, LABELS

def load_model(model_path):
    """Load the trained model and processor."""
    # Initialize processor
    processor = ReviewProcessor()
    
    # Initialize model
    model = HealthcareReviewClassifier()
    
    # Load model weights
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, processor

def predict_review(review_text, model, processor, threshold=0.5):
    """
    Make prediction on a single review.
    
    Args:
        review_text (str): The review text to classify
        model: The trained model
        processor: The data processor
        threshold (float): Classification threshold for binary predictions
    
    Returns:
        dict: Dictionary containing predictions and probabilities for each label
    """
    # Process the review
    inputs = processor.process_text(review_text)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        probabilities = torch.sigmoid(outputs)
    
    # Convert to binary predictions using threshold
    predictions = (probabilities > threshold).int()
    
    # Create results dictionary
    results = {}
    label_idx = 0
    for dimension, labels in LABELS.items():
        results[dimension] = {}
        for label, _ in labels.items():
            results[dimension][label] = {
                'predicted': bool(predictions[0][label_idx]),
                'probability': float(probabilities[0][label_idx])
            }
            label_idx += 1
    
    return results

def format_predictions(predictions):
    """Format predictions into a readable string."""
    output = []
    for dimension, labels in predictions.items():
        output.append(f"\n{dimension.upper()}:")
        for label, result in labels.items():
            status = "✓" if result['predicted'] else "✗"
            output.append(f"  {status} {label}: {result['probability']:.2%}")
    return "\n".join(output)

def main():
    # Get the most recent model
    output_dir = "output"
    if not os.path.exists(output_dir):
        raise ValueError("Output directory not found")
    
    # Get the most recent run directory
    run_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    if not run_dirs:
        raise ValueError("No run directories found")
    
    latest_run = max(run_dirs, key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
    model_path = "best_model.pt"
    
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found at {model_path}")
    
    # Load model and processor
    print("Loading model...")
    model, processor = load_model(model_path)
    print("Model loaded successfully!")
    
    # Interactive prediction loop
    print("\nEnter your review (or 'quit' to exit):")
    while True:
        review = input("\nReview: ").strip()
        if review.lower() == 'quit':
            break
        
        if not review:
            print("Please enter a valid review.")
            continue
        
        # Make prediction
        predictions = predict_review(review, model, processor)
        
        # Print results
        print("\nPredictions:")
        print(format_predictions(predictions))
        print("\n" + "="*50)

if __name__ == "__main__":
    main() 