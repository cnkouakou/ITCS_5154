import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.weight_history = []  # Store weight changes

    def step_function(self, z):
        return 1 if z >= 0 else 0  # Step function

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Initialize weights
        self.bias = 0  # Initialize bias

        for epoch in range(self.epochs):
            for i in range(n_samples):
                linear_output = np.dot(X[i], self.weights) + self.bias
                y_predicted = self.step_function(linear_output)

                # Perceptron weight update rule
                update = self.learning_rate * (y[i] - y_predicted)
                self.weights += update * X[i]
                self.bias += update
            
            # Store weights after each epoch
            self.weight_history.append((self.weights.copy(), self.bias))

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([self.step_function(x) for x in linear_output])

# Create dataset (AND gate)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # AND logic output

# Train Perceptron
perceptron = Perceptron(learning_rate=0.1, epochs=10)
perceptron.fit(X, y)

# Visualization function for weight updates
def plot_weight_updates(X, y, model):
    x_min, x_max = -0.2, 1.2
    y_min, y_max = -0.2, 1.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    plt.figure(figsize=(10, 6))
    for i, (w, b) in enumerate(model.weight_history):
        Z = np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, levels=[0], linewidths=1.5, label=f'Epoch {i+1}', alpha=(i+1)/len(model.weight_history))

    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.Paired)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Perceptron Weight Updates Across Epochs")
    plt.legend([f'Epoch {i+1}' for i in range(len(model.weight_history))])
    plt.show()

plot_weight_updates(X, y, perceptron)
