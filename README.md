# NumCompute: A Modular Machine Learning Framework from Scratch

## Overview

NumCompute is a fully modular, NumPy-based machine learning framework designed to replicate the core architecture of modern ML libraries such as scikit-learn.

This project emphasizes:

- Algorithmic transparency
- Numerical correctness
- Vectorized computation
- Clean and reusable software design
- Modular architecture
- Incremental and streaming machine learning
- Ensemble modelling
- Automated testing and benchmarking

NumCompute provides a complete machine learning workflow built from first principles using NumPy, without relying on external machine learning libraries.

The framework supports both traditional batch workflows and simulated online learning scenarios.

---

## Objectives

- Build ML infrastructure from first principles
- Ensure high-performance computation through NumPy vectorization
- Provide a consistent and extensible API
- Demonstrate end-to-end ML workflows without external ML libraries
- Support incremental and streaming learning
- Implement machine learning algorithms from scratch
- Provide reusable preprocessing, pipeline, metrics, and utility components
- Enable model evaluation, benchmarking, and visualisation

---

## Installation

### Requirements

- Python ≥ 3.8
- NumPy ≥ 1.21

### Install Locally

```bash
git clone <repo-url>
cd NumCompute
pip install -e .

### Development Mode

```bash
pip install -e .[dev]
```
---
## Project Structure

```text
NumCompute/
├── numcompute/
│   ├── __init__.py
│   ├── benchmarking.py
│   ├── io.py
│   ├── metrics.py
│   ├── optim.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   ├── rank.py
│   ├── sort_search.py
│   ├── stats.py
│   ├── stream.py
│   ├── tree.py
│   ├── visualise.py
│   └── utils.py
│
├── demo/
│   ├── quickstart.ipynb
│   └── stream_demo.ipynb
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_pipeline.py
│   ├── test_metrics.py
│   ├── test_tree.py
│   ├── test_stream.py
│   ├── test_ensemble.py
│   └── ...
│
├── README.md
├── pyproject.toml
└── report.pdf
----
```

## Usage Example

```python
import numpy as np
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import Imputer, StandardScaler

X = np.array([[1, 2, np.nan],
              [3, np.nan, 6],
              [7, 8, 9]])

pipeline = Pipeline([
    ("imputer", Imputer()),
    ("scaler", StandardScaler())
])

X_transformed = pipeline.fit(X).transform(X)
print(X_transformed)
```

---

## API Overview

### Data Handling (`io.py`)
- `read_csv()` → Efficient CSV loading with NaN handling and chunking

### Preprocessing (`preprocessing.py`)
- StandardScaler → Z-score normalization
- MinMaxScaler → Feature scaling
- Imputer → Missing value handling
- OneHotEncoder → Categorical encoding

### Sorting & Searching (`sort_search.py`)
- Stable sorting (`np.sort`)
- Top-k selection (`argpartition`)
- Quickselect (k-th element)
- Binary search (logarithmic complexity)

### Ranking (`rank.py`)
- Ranking with tie-handling strategies
- Percentile computation

### Statistics (`stats.py`)
- Mean, variance, min, max
- Histogram generation
- Quantile estimation
- Streaming statistics (Welford algorithm)

### Metrics (`metrics.py`)
- Accuracy, Precision, Recall, F1 Score
- Mean Squared Error (MSE)
- Confusion Matrix
- ROC AUC

### Optimization (`optim.py`)
- Finite-difference gradient estimation
- Jacobian computation

### Pipeline (`pipeline.py`)
- Pipeline (sequential transformations)
- FeatureUnion (parallel transformations)

### Utilities (`utils.py`)
- Distance metrics
- Activation functions
- LogSumExp
- Batch processing

### Benchmarking (`benchmarking.py`)
- Performance comparison tools
  
### Decision Trees (`tree.py`)

- DecisionTreeClassifier
- Gini impurity splitting
- Recursive tree construction
- Predict and predict_proba support
- Streaming updates through partial_fit()

### Ensemble Learning (`ensemble.py`)

- RandomForestClassifier
- Bootstrap sampling
- Feature subsampling
- Majority voting
- Streaming-compatible training

### Streaming Learning (`stream.py`)

- StreamTrainer
- Chunk-based training
- Metric tracking
- Model evaluation
- Performance monitoring

### Visualisation (`visualise.py`)

- Accuracy over time
- Model comparison plots
- Streaming performance graphs
- Benchmark visualisations

---

## Performance Evaluation

## Streaming Learning Results

Experiments were conducted using a synthetic classification dataset containing 2,000 samples and 5 features.

### Decision Tree Performance

| Metric | Value |
|----------|----------|
| Accuracy | 95.35% |
| Precision | 96.16% |
| Recall | 94.54% |
| F1 Score | 95.34% |

### Random Forest Performance

| Metric | Value |
|----------|----------|
| Accuracy | 85.30% |
| Precision | 87.41% |
| Recall | 82.72% |
| F1 Score | 85.00% |

The Decision Tree achieved the strongest overall performance under the streaming configuration used in this project. Accuracy visualisations demonstrated stable learning behaviour across successive stream chunks.
---

## Testing & Validation

Run tests using:

```bash
pytest -v
```

Includes:

- 53 automated unit tests
- Streaming workflow validation
- Decision Tree testing
- Random Forest testing
- Missing value handling
- Numerical stability checks
- Edge-case validation
- Pipeline integration tests

---

## Design Principles

- Vectorization-first approach
- Numerical stability
- Consistent API design
- Modular architecture

---
## Benchmarking

NumCompute includes built-in benchmarking utilities for measuring:

- Model training time
- Prediction latency
- Streaming update performance
- End-to-end workflow execution

Benchmark results can be generated using:

```python
from numcompute.benchmarking import benchmark_stream

## Demo

```bash
cd demo
jupyter notebook quickstart.ipynb
```
cd demo
jupyter notebook stream_demo.ipynb

---

# Future Work

```md
- True online learning algorithms
- Regression Trees
- Random Forest Regressors
- Gradient Boosting
- Hyperparameter optimisation
- Cross-validation framework
- Feature importance analysis
- Concept drift detection
- Distributed stream processing

---
```
# Contributors

## Eruva Akhil — Project Lead & Primary Contributor

Led the overall project and was responsible for the overall architecture, implementation, integration, machine learning components, streaming framework, testing, benchmarking, documentation, and development direction.

## Shweta Bhattacharjee — Core Contributor

Played a key supporting role throughout the project and contributed across multiple aspects of the development process. Worked closely with the project lead on implementation, testing, validation, documentation, troubleshooting, and overall project development.

GitHub: [@ShwetaBhattacharjee](https://github.com/ShwetaBhattacharjee)

## Pravallika — Contributor

Contributed to the project as part of the development team.

---

## Team

The project was developed collaboratively, with team members contributing across development, testing, validation, documentation, and project support.

| Contributor | Role |
|---|---|
| **Eruva Akhil** | Project Lead & Primary Contributor |
| **Shweta Bhattacharjee** | Core Contributor |
| **Pravallika** | Contributor |

---

# Author

**Eruva Akhil**

B.Tech Artificial Intelligence & Machine Learning

---

# Conclusion

NumCompute demonstrates how a modular machine learning framework can be built from scratch using NumPy, with a focus on performance, numerical correctness, modularity, and clarity.

The framework brings together data handling, preprocessing, statistics, optimisation, machine learning metrics, pipelines, decision trees, ensemble learning, streaming learning, benchmarking, visualisation, and automated testing into a unified architecture.

Through its modular design and from-scratch implementations, NumCompute provides a transparent way to understand the core principles behind modern machine learning frameworks while maintaining a structure that can be extended with new algorithms and capabilities.
