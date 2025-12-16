import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load data
print("Loading wine quality data...")
df = pd.read_csv("winequality-red.csv", sep=None, engine="python")

# Target: quality >= 7 is "good" (1), otherwise "not good" (0)
y = (df["quality"] >= 7).astype(int).values
X = df.drop(columns=["quality"]).values

# Data info
print("\n=== DATA INFO ===")
print(f"Total samples: {len(X)}")
print(f"Features: {X.shape[1]}")
print(f"Not Good (0): {np.sum(y==0)}")
print(f"Good (1): {np.sum(y==1)}")

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Compute class weights for imbalanced data
class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(y_train), 
    y=y_train
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print(f"\nClass weights: {class_weight_dict}")

# Build neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=20,
    restore_best_weights=True,
    verbose=0
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    verbose=0,
    min_lr=1e-6
)

# Train model
print("\n=== TRAINING ===")
history = model.fit(
    X_train, y_train, 
    epochs=150,
    batch_size=32, 
    validation_split=0.2,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Evaluate model
print("\n=== EVALUATION ===")
results = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {results[0]:.4f}")
print(f"Test Accuracy: {results[1]*100:.2f}%")
print(f"Test Precision: {results[2]*100:.2f}%")
print(f"Test Recall: {results[3]*100:.2f}%")

# Predictions
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"[[{cm[0][0]:3d} {cm[0][1]:3d}]")
print(f" [{cm[1][0]:3d} {cm[1][1]:3d}]]")

# Classification report
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=["Not Good", "Good"]))

# Visualization 1: Training curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss
axes[0, 0].plot(history.history["loss"], label="Train", linewidth=2)
axes[0, 0].plot(history.history["val_loss"], label="Validation", linewidth=2)
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_title("Training and Validation Loss")

# Accuracy
axes[0, 1].plot(history.history["accuracy"], label="Train", linewidth=2)
axes[0, 1].plot(history.history["val_accuracy"], label="Validation", linewidth=2)
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_title("Training and Validation Accuracy")

# Precision
axes[1, 0].plot(history.history["precision"], label="Train", linewidth=2)
axes[1, 0].plot(history.history["val_precision"], label="Validation", linewidth=2)
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Precision")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_title("Training and Validation Precision")

# Recall
axes[1, 1].plot(history.history["recall"], label="Train", linewidth=2)
axes[1, 1].plot(history.history["val_recall"], label="Validation", linewidth=2)
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Recall")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_title("Training and Validation Recall")

plt.tight_layout()
plt.savefig("outputs/wine_training_curves.png", dpi=150)
print("\n✓ Saved: outputs/wine_training_curves.png")

# Visualization 2: Confusion matrix
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Good", "Good"])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig("outputs/wine_confusion_matrix.png", dpi=150)
print("✓ Saved: outputs/wine_confusion_matrix.png")

# Visualization 3: Metrics comparison
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

metrics_dict = {
    'Not Good': {'Precision': precision[0], 'Recall': recall[0], 'F1-score': f1[0]},
    'Good': {'Precision': precision[1], 'Recall': recall[1], 'F1-score': f1[1]}
}

fig, ax = plt.subplots(figsize=(10, 6))

categories = list(metrics_dict.keys())
metrics = ['Precision', 'Recall', 'F1-score']
x = np.arange(len(categories))
width = 0.25

for i, metric in enumerate(metrics):
    values = [metrics_dict[cat][metric] for cat in categories]
    ax.bar(x + i*width, values, width, label=metric)
    
    for j, v in enumerate(values):
        ax.text(x[j] + i*width, v + 0.02, f'{v:.2f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Class')
ax.set_ylabel('Score')
ax.set_title('Classification Metrics')
ax.set_xticks(x + width)
ax.set_xticklabels(categories)
ax.legend()
ax.set_ylim(0, 1.1)
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/wine_metrics.png", dpi=150)
print("✓ Saved: outputs/wine_metrics.png")

# Summary
print("\n=== RESULTS SUMMARY ===")
print(f"Final Test Accuracy: {results[1]*100:.1f}%")
print(f"Training Epochs: {len(history.history['loss'])}")
print(f"Model Parameters: {model.count_params()}")
print(f"\nConfusion Matrix Details:")
print(f"  True Positive (TP):  {cm[1][1]:3d} - Good correctly identified")
print(f"  True Negative (TN):  {cm[0][0]:3d} - Not Good correctly identified")
print(f"  False Positive (FP): {cm[0][1]:3d} - Not Good misclassified as Good")
print(f"  False Negative (FN): {cm[1][0]:3d} - Good misclassified as Not Good")
print("\n✓ Analysis complete!")