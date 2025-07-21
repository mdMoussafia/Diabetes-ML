import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def create_neural_network(input_dim, task='regression', learning_rate=0.01):
    # Crée un réseau de neurones simple
    model = Sequential()
    
    if task == 'regression':
        model.add(Dense(1, input_shape=(input_dim,), activation='linear'))
        model.compile(optimizer=Adam(learning_rate=learning_rate), 
                     loss='mse', metrics=['mse'])
    elif task == 'classification':
        model.add(Dense(1, input_shape=(input_dim,), activation='sigmoid'))
        model.compile(optimizer=Adam(learning_rate=learning_rate), 
                     loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def sigmoid(z):
    # Fonction sigmoid stable
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def compute_binary_crossentropy(y_true, y_pred):
    # Calcule la cross-entropy binaire
    m = len(y_true)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    cost = -(1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return cost

class LogisticRegressionScratch:
    # Implémentation from scratch de la régression logistique
    
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.costs = []
        self.accuracies = []
        
    def fit(self, X, y):
        # Initialisation
        m, n = X.shape
        self.W = np.random.normal(0, 0.01, (n, 1))
        self.b = 0.0
        
        y = y.reshape(-1, 1)
        
        # Entraînement
        for i in range(self.max_iter):
            # Forward pass
            z = X @ self.W + self.b
            a = sigmoid(z)
            
            # Coût et accuracy
            cost = compute_binary_crossentropy(y, a)
            predictions = (a > 0.5).astype(int)
            accuracy = np.mean(predictions == y)
            
            self.costs.append(cost)
            self.accuracies.append(accuracy)
            
            # Backward pass
            dz = a - y
            dW = (1/m) * (X.T @ dz)
            db = (1/m) * np.sum(dz)
            
            # Mise à jour
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db
    
    def predict_proba(self, X):
        z = X @ self.W + self.b
        return sigmoid(z).flatten()
    
    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

class LinearRegressionScratch:
    # Implémentation from scratch de la régression linéaire
    
    def __init__(self, learning_rate=0.0001, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.costs = []
        
    def fit(self, X, y):
        # Initialisation
        m, n = X.shape
        self.W = np.zeros((n, 1))
        self.b = 0.0
        
        y = y.reshape(-1, 1)
        
        # Entraînement
        for i in range(self.max_iter):
            # Forward pass
            y_pred = X @ self.W + self.b
            
            # Coût
            cost = (1/(2*m)) * np.sum((y_pred - y)**2)
            self.costs.append(cost)
            
            # Backward pass
            error = y_pred - y
            dW = (1/m) * (X.T @ error)
            db = (1/m) * np.sum(error)
            
            # Mise à jour
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db
    
    def predict(self, X):
        return (X @ self.W + self.b).flatten()