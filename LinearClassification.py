import torch
import time
import math
import matplotlib.pyplot as plt

# Generate synthetic data
def generate_data():
    x = torch.randn(100, 2)
    w = torch.randn(2, 1)
    b = torch.randn(1)
    y = torch.sign(x @ w + b)
    y_binary = (y + 1) / 2  # Convert labels from {-1, 1} to {0, 1}
    return x, y_binary

# Least Squares Training with Misclassification Count
def train_least_squares(x, y_binary, alpha=0.01, threshold=0.1):
    w_binary = torch.randn((2, 1), requires_grad=True)
    b_binary = torch.randn(1, requires_grad=True)
    loss_values_ls = []

    while True:
        y_pred = torch.sigmoid(x @ w_binary + b_binary)
        loss = ((y_binary - y_pred) ** 2).mean()
        if loss.item() <= threshold or torch.isnan(loss).any():
            break
        print(f"Least Squares Loss: {loss.item()}")
        loss_values_ls.append(loss.item())

        loss.backward()
        with torch.no_grad():
            w_binary -= alpha * w_binary.grad
            b_binary -= alpha * b_binary.grad
        w_binary.grad.zero_()
        b_binary.grad.zero_()

    # Misclassification Count
    with torch.no_grad():
        y_final_pred = torch.sigmoid(x @ w_binary + b_binary)
        y_pred_labels = (y_final_pred > 0.5).float()
        misclassified = (y_pred_labels != y_binary).sum().item()

    print(f"Least Squares Misclassified Samples: {misclassified}")
    return loss_values_ls, misclassified

# Cross Entropy Training with Misclassification Count
def train_cross_entropy(x, y_binary, alpha=0.01, threshold=0.1):
    w_binary = torch.randn((2, 1), requires_grad=True)
    b_binary = torch.randn(1, requires_grad=True)
    loss_values_ce = []

    while True:
        y_pred = torch.sigmoid(x @ w_binary + b_binary)
        #P = 1/len(x)
        loss = (y_binary * torch.log(y_pred) + (1 - y_binary) * torch.log(1 - y_pred)).mean()

        if loss.item() <= threshold or torch.isnan(loss).any():
            break

        print(f"Cross Entropy Loss: {loss.item()}")
        loss_values_ce.append(loss.item())

        loss.backward()
        with torch.no_grad():
            w_binary -= alpha * w_binary.grad
            b_binary -= alpha * b_binary.grad
        w_binary.grad.zero_()
        b_binary.grad.zero_()

    # Misclassification Count
    with torch.no_grad():
        y_final_pred = torch.sigmoid(x @ w_binary + b_binary)
        y_pred_labels = (y_final_pred > 0.5).float()
        misclassified = (y_pred_labels != y_binary).sum().item()

    print(f"Cross Entropy Misclassified Samples: {misclassified}")
    return loss_values_ce, misclassified

# Softmax Training with Misclassification Count
def train_soft_max(x, y_binary, num_classes=2, alpha=0.01, threshold=0.1):
    w_multiclass = torch.randn((2, num_classes), requires_grad=True)
    b_multiclass = torch.randn(num_classes, requires_grad=True)
    loss_values_sm = []

    y_binary = y_binary.squeeze()
    y_one_hot = torch.nn.functional.one_hot(y_binary.to(torch.int64), num_classes).float()

    while True:
        logits = x @ w_multiclass + b_multiclass
        y_pred = torch.nn.functional.softmax(logits, dim=1)
        loss = -(y_one_hot * torch.log(y_pred)).sum(dim=1).mean()

        if loss.item() <= threshold or torch.isnan(loss).any():
            break
        print(f"Softmax Loss: {loss.item()}")
        loss_values_sm.append(loss.item())

        loss.backward()
        with torch.no_grad():
            w_multiclass -= alpha * w_multiclass.grad
            b_multiclass -= alpha * b_multiclass.grad
        w_multiclass.grad.zero_()
        b_multiclass.grad.zero_()

    # Misclassification Count
    with torch.no_grad():
        logits_final = x @ w_multiclass + b_multiclass
        y_final_pred = torch.nn.functional.softmax(logits_final, dim=1)
        y_pred_labels = torch.argmax(y_final_pred, dim=1)
        misclassified = (y_pred_labels != y_binary).sum().item()

    print(f"Softmax Misclassified Samples: {misclassified}")
    return loss_values_sm, misclassified

# Plot Loss Curves
def plot_losses(loss_values_ls, loss_values_ce, loss_values_sm):
    plt.figure(figsize=(12, 6))
    plt.plot(loss_values_ls, label='Least Squares Loss')
    plt.plot(loss_values_ce, label='Cross Entropy Loss')
    plt.plot(loss_values_sm, label='Softmax Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Main Execution Function
def main1():
    x, y_binary = generate_data()

    start_time_ls = time.perf_counter()
    loss_values_ls, misclassified_ls = train_least_squares(x, y_binary)
    end_time_ls = time.perf_counter()
    elapsed_time_ls = end_time_ls - start_time_ls
    print(f"Least Squares Training Time: {elapsed_time_ls:.4f} seconds")

    start_time_ce = time.perf_counter()
    loss_values_ce, misclassified_ce = train_cross_entropy(x, y_binary)
    end_time_ce = time.perf_counter()
    elapsed_time_ce = end_time_ce - start_time_ce
    print(f"Cross Entropy Training Time: {elapsed_time_ce:.4f} seconds")

    start_time_sm = time.perf_counter()
    loss_values_sm, misclassified_sm = train_soft_max(x, y_binary)
    end_time_sm = time.perf_counter()
    elapsed_time_sm = end_time_sm - start_time_sm
    print(f"Softmax Training Time: {elapsed_time_sm:.4f} seconds")

    print("\nFinal Misclassifications:")
    print(f"Least Squares: {misclassified_ls}")
    print(f"Cross Entropy: {misclassified_ce}")
    print(f"Softmax: {misclassified_sm}")

    # plot_losses(loss_values_ls, loss_values_ce, loss_values_sm)

def main():
    for i in range(100):
        main1()

# Run the script
if __name__ == "__main__":
    main()
