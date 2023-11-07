# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 16:01:08 2023

@author: 17245
"""

import numpy as np
import sys
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

"""
from sklearn.datasets import make_classification
# Generate toy dataset (TOY 1: synthetic data)
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, n_classes=2)
y = y.reshape(-1, 1)  # Make y a column vector

# Split into training and test set
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
"""

### Let's try another Toy example (TOY 2: Breast Cancer dataset)
data = load_breast_cancer()
X = data.data
y = data.target.reshape(-1, 1)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)



class MLP:

    def __init__(self, hidden_dim1=15, hidden_dim2=15, output_dim=1):
        self.W1 = self.b1 = self.W2 = self.b2 = self.W3 = self.b3 = None
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim

        self.mW1 = self.vW1 = self.mW2 = self.vW2 = self.mW3 = self.vW3 = None
        self.mb1 = self.vb1 = self.mb2 = self.vb2 = self.mb3 = self.vb3 = None

    def forward(self, X):
        if self.W1 is None:
            self.W1 = np.random.randn(X.shape[1], self.hidden_dim1) / np.sqrt(X.shape[1])
            self.b1 = np.zeros((self.hidden_dim1, 1))
            self.W2 = np.random.randn(self.hidden_dim1, self.hidden_dim2) / np.sqrt(self.hidden_dim1)
            self.b2 = np.zeros((self.hidden_dim2, 1))
            self.W3 = np.random.randn(self.hidden_dim2, self.output_dim) / np.sqrt(self.hidden_dim2)
            self.b3 = np.zeros((self.output_dim, 1))

            self.mW1 = np.zeros_like(self.W1)
            self.vW1 = np.zeros_like(self.W1)
            self.mb1 = np.zeros_like(self.b1)
            self.vb1 = np.zeros_like(self.b1)

            self.mW2 = np.zeros_like(self.W2)
            self.vW2 = np.zeros_like(self.W2)
            self.mb2 = np.zeros_like(self.b2)
            self.vb2 = np.zeros_like(self.b2)

            self.mW3 = np.zeros_like(self.W3)
            self.vW3 = np.zeros_like(self.W3)
            self.mb3 = np.zeros_like(self.b3)
            self.vb3 = np.zeros_like(self.b3)

        self.z1 = X.dot(self.W1) + self.b1.T
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.a1.dot(self.W2) + self.b2.T
        self.a2 = np.maximum(0, self.z2)
        self.z3 = self.a2.dot(self.W3) + self.b3.T
        self.output = sigmoid(self.z3)
        return self.output
    
    def train(self, X, y, X_val, y_val, epochs=20, initial_lr=0.001, batch_size=64, beta1=0.9, beta2=0.9, epsilon=1e-8, patience=7, lambda_l2=0.001):
        learning_rate = initial_lr
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        losses = []
        val_losses = []

        t = 0
        for epoch in range(epochs):
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for i in range(0, X.shape[0], batch_size):
                t += 1
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                output = self.forward(X_batch)
                error = output - y_batch
                
                
                # Weight the error by instance
                instance_weights = np.where(y_batch == 1, 1, 1) # Determine instance weights based on class labels
                weighted_error = error * instance_weights
    
                # Adjust your loss computation to use the weighted error
                mse_loss = np.mean(np.square(weighted_error))
                l2_loss = lambda_l2 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)) + np.sum(np.square(self.W3)))
                total_loss = mse_loss + l2_loss
                    
                
                losses.append(total_loss)

                # Adjust backpropagation to use the weighted error
                sigmoid_derivative = output * (1 - output)
                weighted_error *= sigmoid_derivative
                
                # Now continue with backpropagation as usual, but use 'weighted_error' instead of 'error'
                dW3 = (self.a2.T).dot(2 * weighted_error)
                db3 = np.sum(2 * weighted_error, axis=0, keepdims=True).T
                da2 = (2 * weighted_error).dot(self.W3.T)
                dz2 = da2 * (self.a2 > 0)
                dW2 = (self.a1.T).dot(dz2)
                db2 = np.sum(dz2, axis=0, keepdims=True).T
                da1 = dz2.dot(self.W2.T)
                dz1 = da1 * (self.a1 > 0)
                dW1 = np.dot(X_batch.T, dz1)
                db1 = np.sum(dz1, axis=0, keepdims=True).T


                # Adding the regularisation term to the gradients
                dW3 += 2 * lambda_l2 * self.W3
                dW2 += 2 * lambda_l2 * self.W2
                dW1 += 2 * lambda_l2 * self.W1

                self.mW1 = beta1 * self.mW1 + (1 - beta1) * dW1
                self.vW1 = beta2 * self.vW1 + (1 - beta2) * np.square(dW1)
                mW1_corr = self.mW1 / (1 - beta1 ** t)
                vW1_corr = self.vW1 / (1 - beta2 ** t)

                self.mW2 = beta1 * self.mW2 + (1 - beta1) * dW2
                self.vW2 = beta2 * self.vW2 + (1 - beta2) * np.square(dW2)
                mW2_corr = self.mW2 / (1 - beta1 ** t)
                vW2_corr = self.vW2 / (1 - beta2 ** t)

                self.mW3 = beta1 * self.mW3 + (1 - beta1) * dW3
                self.vW3 = beta2 * self.vW3 + (1 - beta2) * np.square(dW3)
                mW3_corr = self.mW3 / (1 - beta1 ** t)
                vW3_corr = self.vW3 / (1 - beta2 ** t)

                self.mb1 = beta1 * self.mb1 + (1 - beta1) * db1
                self.vb1 = beta2 * self.vb1 + (1 - beta2) * np.square(db1)
                mb1_corr = self.mb1 / (1 - beta1 ** t)
                vb1_corr = self.vb1 / (1 - beta2 ** t)

                self.mb2 = beta1 * self.mb2 + (1 - beta1) * db2
                self.vb2 = beta2 * self.vb2 + (1 - beta2) * np.square(db2)
                mb2_corr = self.mb2 / (1 - beta1 ** t)
                vb2_corr = self.vb2 / (1 - beta2 ** t)

                self.mb3 = beta1 * self.mb3 + (1 - beta1) * db3
                self.vb3 = beta2 * self.vb3 + (1 - beta2) * np.square(db3)
                mb3_corr = self.mb3 / (1 - beta1 ** t)
                vb3_corr = self.vb3 / (1 - beta2 ** t)

                self.W1 -= learning_rate * (mW1_corr / (np.sqrt(vW1_corr) + epsilon) + 2 * lambda_l2 * self.W1)
                self.W2 -= learning_rate * (mW2_corr / (np.sqrt(vW2_corr) + epsilon) + 2 * lambda_l2 * self.W2)
                self.W3 -= learning_rate * (mW3_corr / (np.sqrt(vW3_corr) + epsilon) + 2 * lambda_l2 * self.W3)

                self.b1 -= learning_rate * (mb1_corr / (np.sqrt(vb1_corr) + epsilon))
                self.b2 -= learning_rate * (mb2_corr / (np.sqrt(vb2_corr) + epsilon))
                self.b3 -= learning_rate * (mb3_corr / (np.sqrt(vb3_corr) + epsilon))

            val_output = self.forward(X_val)
            val_error = val_output - y_val
            val_loss = np.mean(np.square(val_error))
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                learning_rate *= 0.9

            print(f'\nEpoch {epoch+1}/{epochs}\n')
            print(f'Train loss = {total_loss:.3f}, Val Loss: {val_loss:.3f}, Learning Rate: {learning_rate:.6f}\n')

            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

        return losses, val_losses

# Initialize and train MLP
mlp = MLP()
losses = mlp.train(X_train, y_train, X_val, y_val, epochs=300)

# Evaluate the model on test set
test_output = mlp.forward(X_test)
test_predictions = (test_output >= 0.5).astype(int)
accuracy = np.mean(test_predictions == y_test)
print(f"Test Accuracy: {accuracy * 100}%")


