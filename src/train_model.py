import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, accuracy_score
import joblib
from tensorflow import keras
from tensorflow.keras import layers

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(columns=['Class'])
    y = data['Class']
    return X, y

# Isolation Forest Training
def train_isolation_forest(X, y, contamination=0.01):
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    model.fit(X)
    predictions = model.predict(X)
    y_pred = [1 if pred == -1 else 0 for pred in predictions]
    pd.DataFrame({'Predicted': y_pred}).to_csv('../results/isolation_forest_predictions.csv', index=False)
    print("\nIsolation Forest Classification Report:")
    print(classification_report(y, y_pred))
    print("Accuracy:", accuracy_score(y, y_pred))
    joblib.dump(model, '../models/isolation_forest_model.pkl')

# Autoencoder Training
def build_autoencoder(input_dim):
    model = keras.Sequential([
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(input_dim, activation=None)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_autoencoder(X, y, epochs=20, batch_size=32):
    autoencoder = build_autoencoder(X.shape[1])
    autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=1)
    reconstructions = autoencoder.predict(X)
    reconstruction_error = ((X - reconstructions) ** 2).sum(axis=1)
    threshold = reconstruction_error.quantile(0.99)
    y_pred = [1 if error > threshold else 0 for error in reconstruction_error]
    pd.DataFrame({'Predicted': y_pred}).to_csv('../results/autoencoder_predictions.csv', index=False)
    print("\nAutoencoder Classification Report:")
    print(classification_report(y, y_pred))
    print("Accuracy:", accuracy_score(y, y_pred))
    autoencoder.save('../models/autoencoder_model.keras')

# LOF Training
def train_lof(X, y, contamination=0.01):
    model = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    predictions = model.fit_predict(X)
    y_pred = [1 if pred == -1 else 0 for pred in predictions]
    pd.DataFrame({'Predicted': y_pred}).to_csv('../results/lof_predictions.csv', index=False)
    print("\nLOF Classification Report:")
    print(classification_report(y, y_pred))
    print("Accuracy:", accuracy_score(y, y_pred))
    joblib.dump(model, '../models/lof_model.pkl')

if __name__ == "__main__":
    X, y = load_data('../data/processed_data.csv')
    print("Starting training for all models...")

    train_isolation_forest(X, y)
    train_autoencoder(X, y)
    train_lof(X, y)

    print("Training completed for all models. Models and predictions have been saved.")
