"""
Configuration file containing label definitions and other constants for the healthcare app review classification project.
"""

# Label definitions organized by dimension
LABELS = {
    'privacy_concerns': {
        'data_quality': {
            'description': 'Ensuring the data is accurate, complete and kept up to date for its intended use',
            'keywords': ['accuracy', 'completeness', 'correct', 'information', 'data quality', 'personal health', 'up to date', 'reliable']
        },
        'data_control': {
            'description': 'The user\'s ability to access their data, control the data collected, make corrections, and request deletion or cessation of processing',
            'keywords': ['access', 'control', 'manage', 'restrict', 'data control', 'privacy settings', 'delete', 'corrections', 'permission']
        }
    },
    'trust_in_providers': {
        'ethicality': {
            'description': 'The belief that the provider adheres to ethical standards in their practice and interactions. This involves respecting user privacy, being transparent and having clear guidelines',
            'keywords': ['transparency', 'ethical', 'privacy', 'respect', 'trustworthy', 'honest', 'consent', 'terms of service', 'guidelines']
        },
        'competence': {
            'description': 'The belief that the provider can deliver effective (medical) care through technology. It includes the provider\'s capacity to offer useful recommendations and treatment',
            'keywords': ['competence', 'ability', 'care', 'recommendations', 'treatment', 'diagnosis', 'prescription', 'consultation', 'medical advice']
        }
    },
    'trust_in_applications': {
        'reliability': {
            'description': 'Pertains to consistent performance and availability of the technology. It relates to the app\'s dependability in providing consistent, continuous service without disruptions',
            'keywords': ['reliable', 'stable', 'uptime', 'performance', 'functioning', 'working', 'crash', 'login', 'access']
        },
        'support': {
            'description': 'The resources and support available to help users access, navigate or troubleshoot their use of the technology. This includes customer support, in-app help guides, or other resources',
            'keywords': ['support', 'help', 'assistance', 'customer service', 'response', 'guidance', 'troubleshoot', 'navigate', 'resources']
        },
        'risk': {
            'description': 'User\'s perceived risk associated with using the technology. It involves user\'s concerns about potential negative consequences such as data leakage, breaches, fraud or security issues',
            'keywords': ['risk', 'security', 'breach', 'fraud', 'safety', 'concern', 'data leakage', 'personal info', 'privacy']
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