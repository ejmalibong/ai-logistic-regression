import numpy as np

class LogisticRegression:

  def __init__(self, learning_rate = 0.01, n_epochs = 1000):
    self.learning_rate = learning_rate
    self.n_epochs = n_epochs
    self.weights = None
    self.bias = 0.0
    
  def fit(self, X, y):
    n_samples, n_features = X.shape
    np.unique(y)
    
    self.weights = np.zeros(n_features)
    self.bias = 0.0

    for epoch in range(self.n_epochs):
      # forward
      z = X @ self.weights + self.bias
      z = np.clip(z, -500, 500)
      y_hat = 1 / (1 + np.exp(-z))

      # gradient
      error = y_hat - y
      dw = (X.T @ error) / n_samples
      db = np.mean(error)

      # update
      self.weights -= self.learning_rate * dw
      self.bias -= self.bias * db

      # loggin
      if epoch % 100 == 0:
        loss = -np.mean(
            y * np.log(y_hat + 1e-9) + (1 - y) * np.log(1 - y_hat + 1e-9)
        )
        print(f"epoch={epoch} loss={loss:.4f}")

  def predict(self, X):
    z = X @ self.weights + self.bias
    proba = 1 / (1 + np.exp(-z))
    return (proba >= 0.5).astype(int)