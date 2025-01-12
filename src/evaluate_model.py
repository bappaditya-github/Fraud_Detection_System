import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(columns=['Class'])
    y = data['Class']
    return X, y

def load_model(model_path):
    return joblib.load(model_path)

def evaluate_model(model, X, y):
    predictions = model.predict(X)
    y_pred = [1 if pred == -1 else 0 for pred in predictions]
    print("Classification Report:")
    print(classification_report(y, y_pred))
    print("Accuracy:", accuracy_score(y, y_pred))
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

if __name__ == "__main__":
    X, y = load_data('../data/processed_data.csv')
    model = load_model('../models/isolation_forest_model.pkl')
    evaluate_model(model, X, y)
    print("Model evaluation completed.")
