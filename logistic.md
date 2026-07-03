# MLX Logistic Regression Using Gradient Descent

This code manually trains a **logistic regression model** using **gradient descent** in MLX.

Unlike the previous linear regression example, this model is used for **binary classification**.

---

## 1. Hyperparameters

```python
num_features = 100
num_examples = 1_000
num_iters = 10_000
lr = 0.1
```

- `num_features = 100`: each example has 100 input features.
- `num_examples = 1000`: there are 1,000 training examples.
- `num_iters = 10000`: the model trains for 10,000 updates.
- `lr = 0.1`: learning rate for gradient descent.

---

## 2. Create True Parameters

```python
w_star = mx.random.normal((num_features,))
```

This creates the true weight vector used to generate the labels.

Shape:

```text
w_star = (100,)
```

---

## 3. Generate Input Data

```python
X = mx.random.normal((num_examples, num_features))
```

This creates the feature matrix.

Shape:

```text
X = (1000, 100)
```

There are 1,000 examples, each with 100 features.

---

## 4. Generate Binary Labels

```python
y = (X @ w_star) > 0
```

This creates binary labels.

First:

```python
X @ w_star
```

computes a score for each example.

Then:

```python
> 0
```

converts the score into a Boolean label.

So:

```text
If X @ w_star > 0, label = True
If X @ w_star <= 0, label = False
```

This means the dataset is linearly separable.

---

## 5. Initialize Model Weights

```python
w = 1e-2 * mx.random.normal((num_features,))
```

This initializes the model weights randomly.

The model will learn a weight vector `w` that separates the two classes.

---

## 6. Define the Logistic Loss Function

```python
def loss_fn(w):
    logits = X @ w
    return mx.mean(mx.logaddexp(0.0, logits) - y * logits)
```

### Step 1: Compute logits

```python
logits = X @ w
```

A **logit** is the raw model score before converting it into a probability.

For each example:

```text
logit = weighted sum of input features
```

---

## 7. Binary Cross-Entropy Loss

```python
mx.logaddexp(0.0, logits) - y * logits
```

This is the numerically stable form of **binary cross-entropy loss with logits**.

It is equivalent to:

```text
Loss = -y log(sigmoid(logit)) - (1 - y) log(1 - sigmoid(logit))
```

where:

```text
sigmoid(logit) = 1 / (1 + e^(-logit))
```

The sigmoid converts the raw logit into a probability between 0 and 1.

---

## 8. Why `logaddexp` Is Used

```python
mx.logaddexp(0.0, logits)
```

This computes:

```text
log(1 + exp(logits))
```

in a numerically stable way.

So the loss becomes:

```text
log(1 + exp(logit)) - y * logit
```

This avoids numerical issues when logits become very large or very small.

---

## 9. Compute Gradients Automatically

```python
grad_fn = mx.grad(loss_fn)
```

This creates a function that computes:

```text
gradient of loss with respect to w
```

The gradient tells the model how to update the weights to reduce classification error.

---

## 10. Training Loop

```python
tic = time.perf_counter()

for _ in range(num_iters):
    grad = grad_fn(w)
    w = w - lr * grad
    mx.eval(w)

toc = time.perf_counter()
```

Each iteration does three things:

### Compute gradient

```python
grad = grad_fn(w)
```

### Update weights

```python
w = w - lr * grad
```

This is gradient descent:

```text
new weights = old weights - learning rate * gradient
```

### Force execution

```python
mx.eval(w)
```

MLX uses lazy evaluation, so `mx.eval(w)` forces the computation to actually run.

---

## 11. Final Loss

```python
loss = loss_fn(w)
```

Computes the final logistic loss after training.

Lower loss means the model is more confident and more correct.

---

## 12. Final Predictions

```python
final_preds = (X @ w) > 0
```

The model predicts class labels using the learned weights.

If:

```text
X @ w > 0
```

the prediction is `True`.

Otherwise, the prediction is `False`.

This is equivalent to saying:

```text
sigmoid(X @ w) > 0.5
```

---

## 13. Accuracy

```python
acc = mx.mean(final_preds == y)
```

This compares predicted labels with true labels.

For example:

```text
Accuracy = 0.98
```

means the model classified 98% of the examples correctly.

---

## 14. Throughput

```python
throughput = num_iters / (toc - tic)
```

This measures how many training iterations are completed per second.

---

## 15. Print Results

```python
print(
    f"Loss {loss.item():.5f}, Accuracy {acc.item():.5f} "
    f"Throughput {throughput:.5f} (it/s)"
)
```

This prints:

- Final logistic loss
- Final classification accuracy
- Training speed in iterations per second

---

# Overall Workflow

```text
Generate true weights w_star
            │
            ▼
Generate input matrix X
            │
            ▼
Create binary labels:
y = (X @ w_star) > 0
            │
            ▼
Initialize random model weights w
            │
            ▼
Repeat 10,000 times:
    │
    ├── Compute logits: X @ w
    ├── Compute logistic loss
    ├── Compute gradient
    ├── Update weights
    └── Force execution with mx.eval
            │
            ▼
Compute final predictions
            │
            ▼
Compute accuracy
            │
            ▼
Print loss, accuracy, and throughput
```

---

# Simple Summary

This code trains a **logistic regression classifier** from scratch.

It creates fake binary classification data, initializes random weights, uses binary cross-entropy loss, updates weights using gradient descent, and finally reports how accurately the model separates the two classes.
