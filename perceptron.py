import torch
import matplotlib.pyplot as plt
from time import perf_counter

class Perceptron:
    def __init__(self, input_dim, learning_rate=0.01, threshold=0.0):
        self.weights = torch.zeros(input_dim)
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.losses = []

    def predict(self, x):
        return (torch.dot(self.weights, x) >= self.threshold).float()

    def train(self, inputs, targets, epochs=1):
        for epoch in range(epochs):








            
            total_loss = 0
            for x, target in zip(inputs, targets):
                prediction = self.predict(x)
                self.weights += self.learning_rate * (target - prediction) * x
                loss = (target - prediction).pow(2).sum().item()
                total_loss += loss
            self.losses.append(total_loss)
            print(total_loss)

    def backward_test(self, inputs, targets):
        misclassified = 0
        for x, target in zip(inputs, targets):
            prediction = self.predict(x)
            if prediction != target:
                misclassified += 1
        return misclassified

# Generate 100 random linearly separable 2D vectors
torch.manual_seed(0)
inputs = torch.randn(100, 2)
true_weights = torch.tensor([1.0, -1.0])
targets = (torch.mv(inputs, true_weights) > 0).float()

# Train the perceptron
start_time = perf_counter()
perceptron = Perceptron(input_dim=2)
perceptron.train(inputs, targets, epochs=10)
end_time = perf_counter()

training_time = end_time - start_time
print(f"Training time: {training_time:.6f} seconds")

# Test the perceptron
misclassified_points = perceptron.backward_test(inputs, targets)
print(f"Number of misclassified points: {misclassified_points}")

# Plot the data and decision boundary
plt.scatter(inputs[:, 0], inputs[:, 1], c=targets, cmap='coolwarm')
x_vals = torch.linspace(inputs[:, 0].min(), inputs[:, 0].max(), 100)
y_vals = -(perceptron.weights[0] * x_vals + perceptron.threshold) / perceptron.weights[1]
plt.plot(x_vals, y_vals, color='black')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Perceptron Decision Boundary')
plt.show()
