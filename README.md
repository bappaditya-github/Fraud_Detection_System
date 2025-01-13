# Fraud Detection System

# Under Construction

This project aims to build a fraud detection system using anomaly detection techniques and imbalanced data handling strategies.

## Project Structure:

```
├── fraud_detection_system
│   ├── data
│   │   └── raw_data.csv
│   ├── notebooks
│   │   ├── 01_data_exploration.ipynb
│   │   ├── 02_feature_engineering.ipynb
│   │   ├── 03_modeling_isolation_forest.ipynb
│   │   ├── 04_modeling_autoencoders.ipynb
│   │   ├── 05_modeling_lof.ipynb
│   │   └── 06_model_evaluation.ipynb
│   ├── src
│   │   ├── data_preprocessing.py
│   │   ├── feature_engineering.py
│   │   ├── train_model.py
│   │   └── evaluate_model.py
│   ├── models
│   │   └── saved_models
│   ├── reports
│   │   └── figures
│   ├── README.md
│   ├── requirements.txt
│   ├── setup.py
│   └── .gitignore
```

## Description:
```
- **data:** Contains the raw dataset.
- **notebooks:** Jupyter notebooks for step-by-step data exploration, feature engineering, and model building.
- **scripts:** Python scripts for data preprocessing, training, and evaluation.
- **models:** Directory to store trained models.
- **reports:** Stores generated figures and results.
```

## Setup:
```
1. Create a virtual environment: `python -m venv venv`
2. Activate the environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
```

## Technologies Used:
```
- Python
- Scikit-learn, TensorFlow, Keras
- SMOTE, Cost-sensitive learning
- Pandas, Matplotlib, Seaborn
```

## Usage:
- Run the notebooks in sequence for a complete workflow, or execute the scripts for modular execution.

## Contributions
- Fork the repository and create a pull request for contributions.

## License
- MIT License

## Contact
- For inquiries, contact bappadityaghosh.tn@gmail.com.
