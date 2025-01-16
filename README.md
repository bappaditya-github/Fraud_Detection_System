# Fraud Detection System

# Under Construction

This project aims to build a fraud detection system using anomaly detection techniques and imbalanced data handling strategies.

## Project Structure:

```
├── fraud_detection_system
│   ├── data
│   │   ├── processed_data.csv
│   │   ├── raw_data.csv
│   │   ├── test_data.csv
│   │   └── train_data.csv
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
│   │   ├── autoencoder_model.keras
│   │   ├── isolation_forest_model.pkl
│   │   └── lof_model.pkl
│   ├── README.md
│   ├── requirements.txt
│   ├── setup.py
│   └── .gitignore
```

## Description:
```
- **data:** Contains the raw dataset in the file raw_data.csv. The files processed_data.csv, test_data.csv, and train_data.csv will be generated at the time of executions.
- **notebooks:** Jupyter notebooks for step-by-step data exploration, feature engineering, and model building.
- **scripts:** Python scripts for data preprocessing, training, and evaluation.
- **models:** Directory to store trained models. This code generates three model files autoencoder_model.keras, isolation_forest_model.pkl, and lof_model.pkl after running python scripts.
- **reports:** Stores generated figures and results.
```

## Data:
```
- Raw data can be downloaded from various internet source or may contact with the author of this repository through email given below.
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
