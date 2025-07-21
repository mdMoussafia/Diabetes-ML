import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_diabetes_data(filepath='diabetes.csv'):
    # Charge le dataset diabetes et prépare les données
    dataset = pd.read_csv(filepath)
    X = np.array(dataset.drop(columns=['Outcome']))
    y = np.array(dataset['Outcome'])
    return X, y, dataset

def prepare_data(X, y, test_size=0.2, random_state=23, scale=False):
    # Divise les données et optionnellement les normalise
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test, scaler
    
    return X_train, X_test, y_train, y_test

def generate_synthetic_regression_data(n_samples=100, noise_std=2, random_state=44):
    # Génère des données synthétiques pour la régression
    np.random.seed(random_state)
    
    # Coefficients théoriques
    a1, a2, b = 2, 3, 5
    
    # Génération des features
    X1 = np.random.rand(n_samples) * 10
    X2 = np.random.rand(n_samples) * 10
    X = np.column_stack((X1, X2))
    
    # Génération du bruit et de la target
    noise = np.random.randn(n_samples) * noise_std
    y = a1 * X1 + a2 * X2 + b + noise
    
    return X, y, (a1, a2, b)