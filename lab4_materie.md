# Anomaly Detection - Lab 4
## One-Class SVM and Deep SVDD

**Date:** November 2024

This lab explores advanced anomaly detection algorithms focusing on **One-Class SVM (OC-SVM)**, **Support Vector Data Description (SVDD)**, and **Deep SVDD** - powerful methods that learn tight boundaries around normal data.

---

## 📚 Overview

In this lab, we will explore:
- **OC-SVM (One-Class SVM)**: Maximum-margin hyperplane approach
- **SVDD (Support Vector Data Description)**: Minimum enclosing hypersphere
- **Deep SVDD**: Neural network extension of SVDD

### Evolution from Previous Labs

**Labs 1-2 (Distance/Density Methods):**
- KNN, LOF: Local neighborhood analysis
- ❌ Computational: O(n²) complexity
- ❌ Struggle with high dimensions

**Lab 3 (Isolation Methods):**
- IForest, LODA: Fast, scalable
- ✅ O(n log n) complexity
- ✅ Good for large datasets

**Lab 4 (Boundary Methods):**
- OC-SVM, SVDD, Deep SVDD: Learn decision boundaries
- ✅ Theoretically grounded (kernel methods)
- ✅ Flexible with kernel functions
- ✅ Deep learning integration (Deep SVDD)

---

## 🎯 Part 1: One-Class SVM (OC-SVM)

### The Core Idea: "Maximum Margin Hyperplane"

**Key Insight:**
> One-Class SVM finds a hyperplane in feature space that best separates normal data from the origin, maximizing the margin while minimizing violations.

**Think of it like this:**
- Imagine drawing a boundary that captures "normal" data
- Push this boundary as far from the origin as possible
- Allow some points to violate the boundary (controlled by ν)
- Points far from this boundary are anomalies

### Mathematical Formulation

OC-SVM solves the following optimization problem:

$$\min_{w,\xi\in\mathbb{R}^m,\rho\in\mathbb{R}} \frac{1}{2}\|w\|^2 + \frac{1}{m\nu}\sum_{i=1}^{m}\xi_i - \rho$$

**Subject to constraints:**

$$\langle w, \phi(x_i) \rangle \geq \rho - \xi_i, \quad \xi_i \geq 0 \quad \forall i \in \{1, \ldots, m\}$$

### Understanding the Components

#### **Decision Variables:**
- **w**: Weight vector (defines the hyperplane orientation)
- **ρ (rho)**: Offset from origin (defines hyperplane position)
- **ξᵢ (xi)**: Slack variables (allow some violations)

#### **Objective Function Terms:**

1. **½‖w‖²**: Maximizes the margin (distance from hyperplane to origin)
   - Smaller ‖w‖ → Larger margin
   - Regularization term (prevents overfitting)

2. **1/(mν) Σξᵢ**: Penalizes violations
   - Points that fall on wrong side of hyperplane
   - Controlled by parameter ν

3. **-ρ**: Pushes hyperplane away from origin
   - Maximizes the offset
   - Creates larger "normal" region

#### **Constraints:**
- **⟨w, φ(xᵢ)⟩ ≥ ρ - ξᵢ**: Each point should be on correct side of hyperplane
  - φ(xᵢ): Kernel transformation of data point
  - Violations allowed through ξᵢ
- **ξᵢ ≥ 0**: Slack variables must be non-negative

### How OC-SVM Makes Predictions

**Decision Function:**

$$f(x) = \text{sign}(\langle w, \phi(x) \rangle - \rho)$$

- **f(x) ≥ 0**: Point x is classified as **normal** (inlier)
- **f(x) < 0**: Point x is classified as **anomaly** (outlier)

**Anomaly Score:**

The distance from the decision boundary:

$$\text{score}(x) = -(\langle w, \phi(x) \rangle - \rho)$$

- **Negative scores**: Normal points (inside boundary)
- **Positive scores**: Anomalies (outside boundary)
- **Larger positive values**: More anomalous

---

## 🔧 Main Parameters of OC-SVM

### 1. **Kernel Function**

The kernel function φ(·) maps data to a higher-dimensional space where it may be easier to separate normal from anomalous data.

**Common Kernel Choices:**

#### **Linear Kernel:**
$$K(x, x') = \langle x, x' \rangle$$
- **Use when:** Data is already linearly separable
- **Advantages:** Fast, interpretable
- **Disadvantages:** Limited expressiveness

#### **RBF (Radial Basis Function) Kernel:**
$$K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right)$$
- **Use when:** Non-linear patterns, most common choice
- **Parameter γ (gamma):** Controls smoothness
  - **Small γ**: Smooth, large influence radius (may underfit)
  - **Large γ**: Complex, small influence radius (may overfit)
- **Default:** γ = 1/n_features

#### **Polynomial Kernel:**
$$K(x, x') = (\langle x, x' \rangle + r)^d$$
- **Use when:** Polynomial relationships expected
- **Parameters:**
  - **d (degree)**: Polynomial degree
  - **r (coef0)**: Independent term

#### **Sigmoid Kernel:**
$$K(x, x') = \tanh(\gamma \langle x, x' \rangle + r)$$
- **Use when:** Data resembles neural network patterns

### 2. **ν (nu) Parameter**

**Definition:**
> ν is a hyperparameter that controls the trade-off between maximizing the margin and allowing violations.

**Properties:**
- **Range:** 0 < ν ≤ 1
- **Upper bound** on the fraction of outliers in the training set
- **Lower bound** on the fraction of support vectors

**Interpretation:**

| ν Value | Behavior | Use Case |
|---------|----------|----------|
| **Small (0.01-0.1)** | Strict boundary, few outliers allowed | Clean training data |
| **Medium (0.1-0.3)** | Balanced, moderate flexibility | Typical contamination |
| **Large (0.3-0.5)** | Loose boundary, many outliers allowed | Noisy training data |

**Example:**
- ν = 0.1 → At most 10% of training points can be outliers
- ν = 0.1 → At least 10% of training points will be support vectors

**Support Vectors (SVs):**
- Points that lie on or violate the decision boundary
- These are the critical points that define the boundary
- More SVs → More complex boundary

---

## 🔵 Part 2: Support Vector Data Description (SVDD)

### The Core Idea: "Minimum Enclosing Hypersphere"

**Key Insight:**
> Instead of finding a hyperplane, SVDD finds the smallest hypersphere (ball) that encloses the normal data in feature space.

**Think of it like this:**
- Draw the smallest circle (in 2D) or sphere (in higher dimensions) around normal data
- Allow some points to fall outside (controlled by C)
- Points far outside this sphere are anomalies

### Mathematical Formulation

SVDD solves the following optimization problem:

$$\min_{c, R \geq 0, \xi \in \mathbb{R}^m} R^2 + C \sum_{i=1}^{m} \xi_i$$

**Subject to constraints:**

$$\|c - \phi(x_i)\|^2 \leq R^2 + \xi_i, \quad \xi_i \geq 0 \quad \forall i \in \{1, \ldots, m\}$$

### Understanding the Components

#### **Decision Variables:**
- **c**: Center of the hypersphere (in feature space)
- **R**: Radius of the hypersphere
- **ξᵢ**: Slack variables (allow points outside sphere)

#### **Objective Function Terms:**

1. **R²**: Minimizes the sphere radius
   - Smaller sphere → Tighter fit around normal data
   - More compact representation

2. **C Σξᵢ**: Penalizes violations
   - Points that fall outside the sphere
   - **C parameter** controls the trade-off:
     - **Large C**: Fewer violations, tighter fit (may overfit)
     - **Small C**: More violations, looser fit (may underfit)

#### **Constraints:**
- **‖c - φ(xᵢ)‖² ≤ R² + ξᵢ**: Each point should be inside or near the sphere
  - Distance from center c to point xᵢ should be ≤ R
  - Violations allowed through ξᵢ

### How SVDD Makes Predictions

**Decision Function:**

$$f(x) = \|c - \phi(x)\|^2 - R^2$$

- **f(x) ≤ 0**: Point x is **normal** (inside or on sphere)
- **f(x) > 0**: Point x is **anomaly** (outside sphere)

**Anomaly Score:**

$$\text{score}(x) = \|c - \phi(x)\|^2 - R^2$$

- **Negative scores**: Normal points (inside sphere)
- **Positive scores**: Anomalies (outside sphere)
- **Larger positive values**: Farther from sphere, more anomalous

### OC-SVM vs SVDD: Key Differences

| Aspect | OC-SVM | SVDD |
|--------|---------|------|
| **Geometry** | Hyperplane (linear boundary) | Hypersphere (circular boundary) |
| **Separates from** | Origin | Data center |
| **Parameters** | ν (outlier fraction) | C (penalty) |
| **Interpretation** | Maximum margin | Minimum volume |
| **Best for** | Data concentrated in half-space | Data in compact clusters |

**When to use which:**
- **OC-SVM**: When data forms a directional pattern (can be separated from origin)
- **SVDD**: When data forms compact, spherical clusters

---

## 🧠 Part 3: Deep SVDD

### The Core Idea: "Neural Networks + SVDD"

**Key Insight:**
> Deep SVDD combines the representation learning power of deep neural networks with the geometrical elegance of SVDD.

**Traditional SVDD limitations:**
- Relies on hand-crafted kernels
- Fixed feature representation
- May not capture complex patterns

**Deep SVDD solution:**
- Learn the feature representation φ(x; W) using a neural network
- Optimize network weights W to create better features
- Then apply SVDD in the learned feature space

### Mathematical Formulation

Deep SVDD solves the following optimization problem:

$$\min_{W} \frac{1}{n} \sum_{i=1}^{n} \|\phi(x_i; W) - c\|^2 + \frac{\lambda}{2} \sum_{l=1}^{L} \|W_l\|^2_F$$

### Understanding the Components

#### **Decision Variables:**
- **W**: All neural network weights (W₁, W₂, ..., Wₗ)
- **φ(x; W)**: Neural network output (learned representation)
- **c**: Center of the hypersphere (in learned feature space)

#### **Objective Function Terms:**

1. **1/n Σ‖φ(xᵢ; W) - c‖²**: SVDD objective in learned space
   - Minimize distance from network outputs to center c
   - Encourages all normal data to map near center
   - No explicit radius R (implicitly minimized)

2. **λ/2 Σ‖Wₗ‖²_F**: Weight regularization
   - ‖Wₗ‖²_F: Frobenius norm of layer l weights
   - **λ (lambda)**: Regularization strength
     - Prevents overfitting
     - Encourages simpler networks
   - L2 regularization (weight decay)

### Network Architecture Components

#### **Key Parameters:**

1. **L**: Number of hidden layers
   - Determines network depth
   - More layers → More complex representations
   - Typical: 2-5 layers for Deep SVDD

2. **Wₗ**: Weight matrices for layer l
   - Wₗ ∈ ℝ^(dₗ × dₗ₋₁)
   - dₗ: Number of neurons in layer l
   - Learned through backpropagation

3. **c**: Hypersphere center
   - Fixed or learned
   - Common strategy: Initialize as mean of network outputs on training data
   - Then keep fixed during training

4. **λ**: Regularization parameter
   - Controls overfitting
   - Typical values: 10⁻⁶ to 10⁻³

### How Deep SVDD Works: Step-by-Step

#### **Training Phase:**

1. **Initialize Network:**
   - Random weight initialization (e.g., Xavier, He)
   - Or pre-train with autoencoder

2. **Compute Center c:**
   - Forward pass on training data
   - c = mean of network outputs
   - Keep c fixed for rest of training

3. **Training Loop:**
   ```
   For each epoch:
       For each batch:
           1. Forward pass: z = φ(x; W)
           2. Compute loss: L = ‖z - c‖² + λ‖W‖²
           3. Backward pass: compute ∇W L
           4. Update weights: W ← W - η∇W L
   ```

4. **Compute Radius R:**
   - After training, compute on validation set
   - R = quantile of distances {‖φ(xᵢ; W) - c‖²}
   - Example: 95th percentile → 5% contamination

#### **Testing Phase:**

For new point x:
1. **Forward pass:** z = φ(x; W)
2. **Compute distance:** d = ‖z - c‖²
3. **Anomaly score:** score = d - R²
   - score > 0 → Anomaly
   - score ≤ 0 → Normal

### Network Architecture Design

**Typical Deep SVDD Architecture:**

```
Input Layer (d dimensions)
    ↓
Dense Layer 1: d → 128 (ReLU)
    ↓
Dense Layer 2: 128 → 64 (ReLU)
    ↓
Dense Layer 3: 64 → 32 (ReLU)
    ↓
Output Layer: 32 → 16 (Linear)
    ↓
Output: φ(x; W) ∈ ℝ¹⁶
```

**Design Principles:**

1. **Decreasing Dimensions:**
   - Progressively compress information
   - Forces network to learn compact representations

2. **Activation Functions:**
   - Use ReLU for hidden layers
   - **No activation** on output layer
   - Avoid bounded activations (tanh, sigmoid) that could cause all outputs to collapse to same value

3. **Output Dimension:**
   - Not too small (information loss)
   - Not too large (inefficient)
   - Typical: 16-64 dimensions

4. **Avoid Trivial Solutions:**
   - Without regularization, network could map everything to c
   - L2 regularization prevents this
   - Bias terms often set to zero

### Advantages of Deep SVDD

✅ **Learned Representations:**
- Automatically discovers relevant features
- No manual feature engineering

✅ **End-to-End Training:**
- Single objective function
- Joint optimization of features and boundary

✅ **Scalability:**
- Can use mini-batch training
- GPU acceleration

✅ **Flexibility:**
- Can use any neural network architecture
- Can incorporate domain knowledge (CNNs for images, RNNs for sequences)

### Disadvantages of Deep SVDD

❌ **Complexity:**
- More hyperparameters to tune
- Requires careful architecture design

❌ **Training Time:**
- Slower than traditional SVDD
- Needs GPU for large datasets

❌ **Data Requirements:**
- Needs sufficient training data
- Risk of overfitting with small datasets

❌ **Interpretability:**
- Deep network = black box
- Hard to understand learned features

---

## 📊 Comparison: OC-SVM vs SVDD vs Deep SVDD

| Aspect | OC-SVM | SVDD | Deep SVDD |
|--------|---------|------|-----------|
| **Geometry** | Hyperplane | Hypersphere | Hypersphere (learned space) |
| **Feature Space** | Fixed kernel | Fixed kernel | Learned representation |
| **Optimization** | Convex QP | Convex QP | Non-convex (neural network) |
| **Scalability** | Medium | Medium | High (mini-batch) |
| **Expressiveness** | Kernel-dependent | Kernel-dependent | Very high |
| **Training Time** | Medium | Medium | High |
| **Interpretability** | Low | Medium | Very low |
| **Data Requirements** | Moderate | Moderate | High |
| **Best For** | Moderate data, kernel patterns | Compact clusters | Large data, complex patterns |

---

## 🛠️ Utility Functions for Lab 4

### Data Generation

#### **`generate_data()`**

Custom function to create synthetic anomaly detection datasets.

**Purpose:** Generates normal samples from multivariate Gaussian and outliers from uniform distribution.

**Parameters:**
- `n_train`: Number of training samples (default=200)
- `n_test`: Number of test samples (default=100)
- `n_features`: Number of features/dimensions (default=2)
- `contamination`: Proportion of outliers (default=0.1, i.e., 10%)
- `offset`: Range of values for uniform outliers (default=10)

**Returns:**
- `X_train`, `X_test`: Feature matrices
- `y_train`, `y_test`: Labels (0=normal, 1=outlier)

**Example:**
```python
X_train, X_test, y_train, y_test = generate_data(
    n_train=300, 
    n_test=100, 
    n_features=2, 
    contamination=0.1,
    offset=15
)
```

**How it works:**
1. Normal samples: `np.random.multivariate_normal(mean, cov, n_normal)`
2. Outliers: `np.random.uniform(-offset, offset, (n_outliers, n_features))`

---

### Data Preprocessing

#### **`sklearn.model_selection.train_test_split()`**

Splits datasets into random train and test subsets.

**Parameters:**
- `*arrays`: Input data arrays to split (e.g., X, y)
- `test_size`: Fraction or absolute number of test samples (default=None)
  - Example: `test_size=0.3` → 30% for testing
  - Example: `test_size=50` → 50 samples for testing
- `train_size`: Fraction or absolute number of train samples (default=None)
- `random_state`: Random seed for reproducibility (default=None)
- `shuffle`: Whether to shuffle data before splitting (default=True)
- `stratify`: Array to maintain class proportions (default=None)
  - Example: `stratify=y` → Same proportion of classes in train/test

**Returns:** Split arrays in same order as input

**Example:**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y  # Maintain class balance
)
```

#### **`pyod.utils.utility.standardizer()`**

Transforms data to zero-mean and unit variance (z-score normalization).

**Purpose:** Ensures all features are on same scale, improving algorithm performance.

**Parameters:**
- `X`: Training samples (required)
- `X_t`: Test samples (default=None)

**Returns:** 
- Standardized `X` (and `X_t` if provided)

**Formula:** 
$$z = \frac{x - \mu}{\sigma}$$

Where μ and σ are computed from training data only.

**Example:**
```python
X_train_std, X_test_std = standardizer(X_train, X_test)
```

**⚠️ Important:** 
- Always fit on training data only
- Apply same transformation to test data
- Prevents data leakage

---

### File I/O

#### **`scipy.io.loadmat()`**

Loads data from MATLAB .mat files.

**Parameters:**
- `file_name`: Path to .mat file (required)
- `mdict`: Dictionary to insert MATLAB variables into (default=None)
- `appendmat`: Append .mat extension if missing (default=True)

**Returns:** Dictionary with MATLAB variable names as keys

**Example:**
```python
from scipy.io import loadmat

mat_data = loadmat('shuttle.mat')
X = mat_data['X']  # Features
y = mat_data['y'].ravel()  # Labels (flatten)

print(f"Data shape: {X.shape}")
print(f"Contamination: {np.mean(y) * 100:.2f}%")
```

**Common MATLAB datasets:**
- `shuttle.mat`, `cardio.mat`, `arrhythmia.mat`
- From ODDS (Outlier Detection DataSets) repository

---

### Evaluation Metrics

#### **`sklearn.metrics.roc_auc_score()`**

Computes Area Under the ROC Curve (AUC).

**Purpose:** Threshold-independent measure of classification performance.

**Parameters:**
- `y_true`: True binary labels (0/1)
- `y_score`: Target scores (anomaly scores, not binary predictions)

**Returns:** AUC score (range: 0 to 1)

**Interpretation:**
- **AUC = 1.0**: Perfect classifier
- **AUC = 0.5**: Random classifier
- **AUC > 0.9**: Excellent
- **AUC > 0.8**: Good
- **AUC > 0.7**: Fair

**Example:**
```python
from sklearn.metrics import roc_auc_score

# Train model
model.fit(X_train)

# Get anomaly scores (not binary predictions!)
y_scores = model.decision_function(X_test)

# Compute AUC
auc = roc_auc_score(y_test, y_scores)
print(f"AUC Score: {auc:.4f}")
```

**⚠️ Important:** 
- Use `decision_function()` or `decision_scores_`, NOT `predict()`
- Higher scores should indicate anomalies
- For sklearn models, may need to negate: `y_scores = -model.score_samples(X_test)`

---

### NumPy Utilities

#### **`numpy.quantile()`**

Computes quantiles of data (value below which q% of data falls).

**Purpose:** Determine threshold for anomaly detection based on contamination level.

**Parameters:**
- `a`: Input array (required)
- `q`: Quantile value(s) between 0 and 1 (required)
  - Example: `q=0.9` → 90th percentile
  - Example: `q=[0.25, 0.5, 0.75]` → Quartiles
- `axis`: Axis along which to compute (default=None, flattened array)

**Returns:** Quantile value(s)

**Example:**
```python
# Get anomaly scores
anomaly_scores = model.decision_scores_

# Set threshold at 90th percentile (assume 10% contamination)
threshold = np.quantile(anomaly_scores, 0.9)

# Classify based on threshold
y_pred = (anomaly_scores > threshold).astype(int)

print(f"Threshold: {threshold:.4f}")
print(f"Outliers detected: {np.sum(y_pred)}")
```

**Common use cases:**
- `q=0.95` for 5% contamination
- `q=0.90` for 10% contamination
- `q=0.99` for 1% contamination

#### **`numpy.random.uniform()`**

Draws samples from uniform distribution.

**Purpose:** Generate uniformly distributed outliers for synthetic datasets.

**Parameters:**
- `low`: Lower boundary of distribution (default=0.0)
- `high`: Upper boundary of distribution (default=1.0)
- `size`: Output shape (default=None)
  - Example: `size=10` → 1D array of 10 samples
  - Example: `size=(100, 5)` → 2D array of 100 samples × 5 features

**Returns:** Array of random samples

**Example:**
```python
# Generate 50 outliers in 3D space, range [-10, 10]
outliers = np.random.uniform(
    low=-10, 
    high=10, 
    size=(50, 3)
)

# Generate outliers far from normal data
# Normal data in [0, 5], outliers in [10, 20]
extreme_outliers = np.random.uniform(10, 20, size=(20, 2))
```

---

## 🎯 When to Use Each Method

### Decision Guide

**Choose OC-SVM when:**
- ✅ Moderate-sized dataset (<10,000 samples)
- ✅ Need theoretical guarantees (convex optimization)
- ✅ Data can be separated from origin
- ✅ Good kernel choice available (e.g., RBF for most cases)
- ✅ Want interpretable parameters (ν)

**Choose SVDD when:**
- ✅ Data forms compact, spherical clusters
- ✅ Need geometric interpretation (sphere radius)
- ✅ Similar constraints as OC-SVM
- ✅ Want to explicitly minimize enclosing volume

**Choose Deep SVDD when:**
- ✅ Large dataset (>10,000 samples)
- ✅ Complex, high-dimensional patterns
- ✅ Have GPUs available
- ✅ Can afford longer training time
- ✅ Don't need interpretability
- ✅ Have sufficient normal training data

**Practical Tips:**

1. **Start Simple:**
   - Try OC-SVM with RBF kernel first
   - Use default parameters or grid search

2. **Scale Up if Needed:**
   - If OC-SVM too slow → Try approximation methods
   - If patterns too complex → Try Deep SVDD

3. **Always Preprocess:**
   - Standardize features (`standardizer()`)
   - Remove duplicates
   - Handle missing values

4. **Validate Properly:**
   - Use AUC for threshold-independent evaluation
   - Cross-validate hyperparameters
   - Visualize results when possible (2D/3D)

---

## 🎓 Summary: Key Takeaways for Lab 4

### **Core Concepts**

1. **Boundary-Based Methods:**
   - Find decision boundaries that separate normal from anomalous
   - OC-SVM: Hyperplane
   - SVDD: Hypersphere
   - Deep SVDD: Hypersphere in learned space

2. **Kernel Methods:**
   - Transform data to higher-dimensional space
   - RBF kernel most common and effective
   - Implicit mapping (no explicit computation needed)

3. **Deep Learning Integration:**
   - Deep SVDD learns optimal feature representation
   - End-to-end training
   - Scales to complex, high-dimensional data

### **Parameter Tuning Guidelines**

**OC-SVM:**
- Start with: `kernel='rbf'`, `nu=0.1`, `gamma='scale'`
- Increase ν if too many false positives
- Adjust γ for smoothness

**SVDD:**
- Start with: `kernel='rbf'`, `C=1.0`, `gamma='scale'`
- Increase C for tighter fit
- Similar to OC-SVM in practice

**Deep SVDD:**
- Architecture: Decreasing dimensions (e.g., 128→64→32→16)
- λ: Start with 1e-4, adjust for overfitting
- Learning rate: 1e-4 to 1e-3
- Epochs: 50-200 depending on dataset size

### **Comparison with Previous Labs**

| Lab | Methods | Key Idea | Best For |
|-----|---------|----------|----------|
| **Lab 1-2** | KNN, LOF | Distance/density | Small, low-dim data |
| **Lab 3** | IForest, LODA | Isolation/projection | Large, fast needed |
| **Lab 4** | OC-SVM, SVDD, Deep SVDD | Boundary learning | Theoretical guarantees, complex patterns |

### **Practical Workflow**

1. **Load & Explore Data:**
   ```python
   mat = loadmat('dataset.mat')
   X, y = mat['X'], mat['y'].ravel()
   ```

2. **Preprocess:**
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
   X_train, X_test = standardizer(X_train, X_test)
   ```

3. **Train Model:**
   ```python
   from sklearn.svm import OneClassSVM
   model = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
   model.fit(X_train)
   ```

4. **Evaluate:**
   ```python
   y_scores = -model.score_samples(X_test)  # Note the negative sign!
   auc = roc_auc_score(y_test, y_scores)
   print(f"AUC: {auc:.4f}")
   ```

5. **Set Threshold:**
   ```python
   threshold = np.quantile(y_scores, 0.9)  # 10% contamination
   y_pred = (y_scores > threshold).astype(int)
   ```

### **Common Pitfalls to Avoid**

❌ **Forgetting to standardize data**
✅ Always use `standardizer()` before training

❌ **Using `predict()` for AUC calculation**
✅ Use `decision_function()` or `score_samples()` (continuous scores)

❌ **Wrong sign for sklearn models**
✅ Negate `score_samples()`: `y_scores = -model.score_samples(X)`

❌ **Not setting random_state**
✅ Always set for reproducibility

❌ **Using test data for threshold selection**
✅ Use validation set or quantiles from training scores

---

## 🚀 You're Ready for Lab 4 Exercises!

Good luck with your assignments! Remember:
- Start with OC-SVM (simpler, faster)
- Compare multiple kernels and parameters
- Visualize decision boundaries when possible (2D data)
- Use proper evaluation metrics (AUC, not just accuracy)
- Deep SVDD is powerful but requires more tuning

**Key Skills You'll Practice:**
- ✅ Understanding kernel methods
- ✅ Tuning SVM parameters
- ✅ Working with real datasets (MATLAB files)
- ✅ Comparing boundary-based vs isolation-based methods
- ✅ (Optional) Implementing Deep SVDD with neural networks

---

## 📚 Additional Resources

**PyOD Documentation:**
- OC-SVM: `from pyod.models.ocsvm import OCSVM`
- Note: PyOD wraps sklearn's OneClassSVM

**Scikit-learn:**
- `sklearn.svm.OneClassSVM`: Main implementation
- `sklearn.metrics`: Evaluation functions

**Deep Learning:**
- PyTorch: For implementing Deep SVDD
- TensorFlow/Keras: Alternative frameworks

**Papers:**
- OC-SVM: Schölkopf et al. (2001)
- SVDD: Tax & Duin (2004)
- Deep SVDD: Ruff et al. (2018)

---

**Happy Learning! 🎓**
