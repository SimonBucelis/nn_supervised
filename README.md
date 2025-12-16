# Neural Network Classification Projects

Two supervised learning projects using TensorFlow/Keras for binary classification tasks.

## Projects

### 1. Breast Cancer Classification
Binary classification to predict whether a breast tumor is benign (B) or malignant (M).

**Dataset:** `breastcancer(in).csv`

**Model Architecture:**
- Input layer (30 features)
- Hidden layer: 20 neurons, sigmoid activation
- Output layer: 1 neuron, sigmoid activation

**Features:**
- Data normalization using StandardScaler
- 80/20 train-test split with stratification
- Early stopping (patience=15)
- Binary cross-entropy loss
- Adam optimizer

**Output:**
- `breast_cancer_analysis.png` - Comprehensive visualization with:
  - Training/validation loss curves
  - Training/validation accuracy curves
  - Confusion matrix heatmap
  - Classification metrics bar chart

![Breast Cancer Analysis](breast_cancer_analysis.png)

### 2. Wine Quality Classification
Binary classification to predict wine quality (good: quality ≥ 7, not good: quality < 7).

**Dataset:** `winequality-red.csv`

**Model Architecture:**
- Input layer (11 features)
- Hidden layer 1: 32 neurons, ReLU activation, 30% dropout
- Hidden layer 2: 16 neurons, ReLU activation, 20% dropout
- Output layer: 1 neuron, sigmoid activation

**Features:**
- Data normalization using StandardScaler
- 80/20 train-test split with stratification
- Class weight balancing for imbalanced dataset
- Early stopping (patience=20)
- Learning rate reduction on plateau
- Tracks accuracy, precision, and recall metrics

**Output:**
- `wine_training_curves.png` - Training curves for loss, accuracy, precision, recall
- `wine_confusion_matrix.png` - Confusion matrix visualization
- `wine_metrics.png` - Classification metrics comparison

![Wine Training Curves](wine_mokymo_kreives.png)

![Wine Confusion Matrix](wine_konfusijos_matrica.png)

![Wine Metrics](wine_metrikos.png)

## Requirements

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
```

## Usage

### Breast Cancer Classification
```bash
python breast_cancer_classifier.py
```

### Wine Quality Classification
```bash
python wine_quality_classifier.py
```

## Key Differences Between Projects

| Feature | Breast Cancer | Wine Quality |
|---------|--------------|--------------|
| **Dataset Balance** | Relatively balanced | Imbalanced (few good wines) |
| **Model Complexity** | Simpler (2 layers) | More complex (3 layers + dropout) |
| **Activation** | Sigmoid hidden layer | ReLU hidden layers |
| **Regularization** | None | Dropout layers |
| **Class Weights** | Not used | Used for imbalance |
| **Learning Rate** | Fixed | Adaptive (ReduceLROnPlateau) |
| **Metrics Tracked** | Accuracy only | Accuracy, Precision, Recall |

## Results Interpretation

### Confusion Matrix Terms:
- **TP (True Positive):** Correctly predicted positive class
- **TN (True Negative):** Correctly predicted negative class
- **FP (False Positive):** Incorrectly predicted as positive (Type I error)
- **FN (False Negative):** Incorrectly predicted as negative (Type II error)

### Classification Metrics:
- **Accuracy:** Overall correctness = (TP + TN) / Total
- **Precision:** Of predicted positives, how many are correct = TP / (TP + FP)
- **Recall:** Of actual positives, how many were found = TP / (TP + FN)
- **F1-Score:** Harmonic mean of precision and recall

## Project Structure

```
.
├── breast_cancer_classifier.py
├── wine_quality_classifier.py
├── README.md
├── breastcancer(in).csv (required)
├── winequality-red.csv (required)
└── outputs/
    ├── breast_cancer_analysis.png
    ├── wine_training_curves.png
    ├── wine_confusion_matrix.png
    └── wine_metrics.png
```

## Notes

- Both scripts use `matplotlib.use('Agg')` for non-interactive plotting
- Visualizations are automatically saved as PNG files
- Models use early stopping to prevent overfitting
- Random state is set to 42 for reproducibility
- Data files should be placed in `~/Downloads/` directory

## License

Educational project for learning neural network classification techniques.
