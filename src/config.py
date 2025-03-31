"""
Configuration file containing label definitions and other constants for the healthcare app review classification project.
"""

# Label definitions organized by dimension
LABELS = {
    'privacy_concerns': {
        'data_quality': {
            'description': 'Accuracy and completeness of personal health information',
            'keywords': ['accuracy', 'completeness', 'correct', 'information', 'data quality', 'personal health']
        },
        'data_control': {
            'description': 'Users ability to access, manage, and restrict their data',
            'keywords': ['access', 'control', 'manage', 'restrict', 'data control', 'privacy settings']
        }
    },
    'trust_in_providers': {
        'ethicality': {
            'description': 'Whether users believe the provider acts transparently and respects privacy',
            'keywords': ['transparency', 'ethical', 'privacy', 'respect', 'trustworthy', 'honest']
        }
    },
    'trust_in_applications': {
        'reliability': {
            'description': 'Apps performance and uptime',
            'keywords': ['reliable', 'stable', 'uptime', 'performance', 'functioning', 'working']
        },
        'support': {
            'description': 'Availability and responsiveness of user assistance',
            'keywords': ['support', 'help', 'assistance', 'customer service', 'response', 'guidance']
        },
        'risk': {
            'description': 'User concerns over potential negative consequences like data breaches or fraud',
            'keywords': ['risk', 'security', 'breach', 'fraud', 'safety', 'concern']
        }
    }
}

# Model configuration
MODEL_CONFIG = {
    'max_length': 512,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 5,
    'warmup_steps': 1000,
    'weight_decay': 0.01,
}

# Data processing configuration
DATA_CONFIG = {
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'random_seed': 42,
    'min_review_length': 10,
    'max_review_length': 1000,
}

# Paths
PATHS = {
    'data': 'data',
    'models': 'models',
    'results': 'results',
    'logs': 'logs',
} 