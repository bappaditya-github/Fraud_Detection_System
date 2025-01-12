import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def load_data(file_path):
    return pd.read_csv(file_path)

def feature_engineering(df):
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Standardization
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Polynomial Features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = pd.DataFrame(poly.fit_transform(X_scaled), columns=poly.get_feature_names_out(X_scaled.columns))

    return X_poly, y

def save_features(X, y, output_path):
    data = pd.concat([X, y.reset_index(drop=True)], axis=1)
    data.to_csv(output_path, index=False)

if __name__ == "__main__":
    df = load_data('../data/raw_data.csv')
    X_poly, y = feature_engineering(df)
    save_features(X_poly, y, '../data/processed_data.csv')
    print("Feature engineering completed and data saved.")
