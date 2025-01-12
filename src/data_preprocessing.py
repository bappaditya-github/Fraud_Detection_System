import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    X = df.drop(columns=['Class'])
    y = df['Class']
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, y

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def save_preprocessed_data(X_train, X_test, y_train, y_test, output_path):
    train_data = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    test_data = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)
    train_data.to_csv(f'{output_path}/train_data.csv', index=False)
    test_data.to_csv(f'{output_path}/test_data.csv', index=False)

if __name__ == "__main__":
    df = load_data('../data/raw_data.csv')
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    save_preprocessed_data(X_train, X_test, y_train, y_test, '../data')
    print("Data preprocessing completed and saved.")
