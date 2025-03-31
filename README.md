# Healthcare App Review Classification

This project focuses on developing a model to classify user reviews of healthcare apps, specifically those related to patient portal and telehealth services. The classification system is based on a set of well-defined labels that reflect different dimensions of user concerns and experiences.

## Classification Dimensions and Labels

### 1. Privacy Concerns
- **Data Quality**: Accuracy and completeness of personal health information
- **Data Control**: Users' ability to access, manage, and restrict their data

### 2. Trust in Providers
- **Ethicality**: Whether users believe the provider acts transparently and respects privacy
- **Competence**: User's trust in the provider's ability to offer effective and appropriate care through the app

### 3. Trust in Applications
- **Reliability**: App's performance and uptime
- **Support**: Availability and responsiveness of user assistance
- **Risk**: User concerns over potential negative consequences like data breaches or fraud

## Project Structure

```
.
├── data/               # Data storage directory
├── src/               # Source code
│   ├── data/         # Data processing modules
│   ├── models/       # Model implementations
│   ├── training/     # Training scripts
│   └── evaluation/   # Evaluation metrics and analysis
├── notebooks/        # Jupyter notebooks for analysis
├── tests/           # Unit tests
└── requirements.txt  # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

python src/main.py --data_path path/to/your/data.csv --output_dir output

## Contributing

[Contribution guidelines will be added]

## License

[License information will be added] 