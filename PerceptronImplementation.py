import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def step_function(self, z):
        return 1 if z >= 0 else 0  # Step activation function

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Initialize weights
        self.bias = 0  # Initialize bias

        for _ in range(self.epochs):
            for i in range(n_samples):
                linear_output = np.dot(X[i], self.weights) + self.bias
                y_predicted = self.step_function(linear_output)
                
                # Update rule: w = w + learning_rate * (y - y_pred) * x
                update = self.learning_rate * (y[i] - y_predicted)
                self.weights += update * X[i]
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([self.step_function(x) for x in linear_output])

# Example dataset (AND logic gate)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input features
y = np.array([0, 0, 0, 1])  # AND gate output

# Train Perceptron
perceptron = Perceptron(learning_rate=0.1, epochs=10)
perceptron.fit(X, y)

# Make predictions
predictions = perceptron.predict(X)
print("Predictions:", predictions)
