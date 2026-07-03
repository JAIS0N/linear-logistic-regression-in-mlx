# MLX Linear Regression Using Gradient Descent

This code manually trains a **linear regression model** using **gradient descent** with MLX's automatic differentiation.

---

## 1. Import Libraries

```python
import time
import mlx.core as mx
```

- `time` is used to measure training speed.
- `mlx.core` provides tensors, mathematical operations, automatic differentiation, and random number generation.

---

## 2. Define Hyperparameters

```python
num_features = 100
num_examples = 1_000
num_iters = 10_000
lr = 0.01
```

- `num_features = 100` → Each training example has 100 input features.
- `num_examples = 1000` → The dataset contains 1,000 examples.
- `num_iters = 10000` → Perform 10,000 gradient descent updates.
- `lr = 0.01` → Learning rate that controls the size of each weight update.

---

## 3. Create the True Weights

```python
w_star = mx.random.normal((num_features,))
```

This generates the **true weight vector** that is used to create the dataset.

Think of `w_star` as the hidden answer that the model will try to learn.

Shape:

```text
w_star = (100,)
```

---

## 4. Generate Input Data

```python
X = mx.random.normal((num_examples, num_features))
```

Creates the input feature matrix.

Shape:

```text
X = (1000, 100)
```

Each row represents one training example, and each column represents one feature.

---

## 5. Generate Target Values

```python
eps = 1e-2 * mx.random.normal((num_examples,))
y = X @ w_star + eps
```

The targets are generated using the equation:

\[
y = Xw_{star} + \epsilon
\]

where:

- `X @ w_star` computes the true output.
- `eps` adds a small amount of random noise.

Without noise, the relationship would be perfectly linear.

---

## 6. Initialize Model Weights

```python
w = 1e-2 * mx.random.normal((num_features,))
```

The model begins with a random guess for the weights.

Initially,

```text
w ≠ w_star
```

Training will gradually move `w` closer to `w_star`.

---

## 7. Define the Loss Function

```python
def loss_fn(w):
    return 0.5 * mx.mean(mx.square(X @ w - y))
```

The loss function measures how far the predictions are from the true values.

### Step 1: Make predictions

```python
X @ w
```

### Step 2: Compute prediction errors

```python
X @ w - y
```

### Step 3: Square the errors

```python
mx.square(...)
```

### Step 4: Compute the average

```python
mx.mean(...)
```

### Step 5: Multiply by 0.5

```python
0.5 * ...
```

The factor of `0.5` is commonly used because it simplifies the gradient during differentiation.

This is essentially the **Mean Squared Error (MSE)** loss.

---

## 8. Compute Gradients Automatically

```python
grad_fn = mx.grad(loss_fn)
```

MLX automatically computes the gradient of the loss with respect to the weights.

Instead of manually deriving derivatives, `grad_fn` returns:

```text
∂Loss / ∂w
```

This gradient indicates the direction in which the weights should move to reduce the loss.

---

## 9. Training Loop

```python
tic = time.perf_counter()

for _ in range(num_iters):
    grad = grad_fn(w)
    w = w - lr * grad
    mx.eval(w)

toc = time.perf_counter()
```

Each iteration performs the following steps:

### Compute the gradient

```python
grad = grad_fn(w)
```

The gradient measures how much each weight contributes to the current loss.

---

### Update the weights

```python
w = w - lr * grad
```

Gradient descent updates the weights according to:

\[
w = w - \eta \nabla L
\]

where:

- `η` is the learning rate.
- `∇L` is the gradient.

The weights move in the direction that reduces the loss.

---

### Execute the computation

```python
mx.eval(w)
```

MLX uses **lazy evaluation**, meaning computations are scheduled but not executed immediately.

`mx.eval()` forces the computation to execute during each iteration.

---

## 10. Compute Final Loss

```python
loss = loss_fn(w)
```

Calculates the loss after training is complete.

A smaller loss indicates better predictions.

---

## 11. Measure Weight Error

```python
error_norm = mx.sum(mx.square(w - w_star)).item() ** 0.5
```

This computes the Euclidean (L2) distance between the learned weights and the true weights.

Mathematically,

\[
\|w - w_{star}\|_2
\]

A value close to zero means the model successfully recovered the original weights.

---

## 12. Measure Training Speed

```python
throughput = num_iters / (toc - tic)
```

Calculates the number of gradient descent iterations completed per second.

---

## 13. Print Results

```python
print(
    f"Loss {loss.item():.5f}, "
    f"L2 distance: |w-w*| = {error_norm:.5f}, "
    f"Throughput {throughput:.5f} (it/s)"
)
```

Outputs:

- Final loss
- Distance between learned and true weights
- Training speed in iterations per second

---

# Overall Workflow

```text
Generate true weights (w_star)
            │
            ▼
Generate input features (X)
            │
            ▼
Create targets (y = Xw_star + noise)
            │
            ▼
Initialize random weights (w)
            │
            ▼
Repeat for 10,000 iterations:
    │
    ├── Compute predictions
    ├── Compute loss
    ├── Compute gradients
    ├── Update weights
    └── Execute computation
            │
            ▼
Evaluate final loss
            │
            ▼
Compare learned weights with true weights
            │
            ▼
Report training speed
```
