import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

# Load data
print("Loading breast cancer data...")
df = pd.read_csv("breastcancer(in).csv")

# Target variable: M=1 (malignant), B=0 (benign)
y = df["diagnosis"].map({"M": 1, "B": 0}).astype(int).values

# Features: remove diagnosis, ID, and empty columns
columns_to_drop = ["diagnosis"]
if "id" in df.columns: 
    columns_to_drop.append("id")
if "Unnamed: 32" in df.columns: 
    columns_to_drop.append("Unnamed: 32")
    
X = df.drop(columns=columns_to_drop).values

# Data info
print("\n=== DATA INFO ===")
print(f"Total samples: {len(X)}")
print(f"Features: {X.shape[1]}")
print(f"Benign (B): {np.sum(y==0)}")
print(f"Malignant (M): {np.sum(y==1)}")

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# Build neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(20, activation="sigmoid"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

print("\n=== MODEL ARCHITECTURE ===")
model.summary()

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=0
)

# Train model
print("\n=== TRAINING ===")
history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=32, 
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate model
print("\n=== EVALUATION ===")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Predictions
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"[[{cm[0][0]:3d} {cm[0][1]:3d}]")
print(f" [{cm[1][0]:3d} {cm[1][1]:3d}]]")

# Classification report
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

# Visualizations
fig = plt.figure(figsize=(14, 10))

# Loss curves
plt.subplot(2, 2, 1)
plt.plot(history.history["loss"], label="Train", linewidth=2)
plt.plot(history.history["val_loss"], label="Validation", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)

# Accuracy curves
plt.subplot(2, 2, 2)
plt.plot(history.history["accuracy"], label="Train", linewidth=2)
plt.plot(history.history["val_accuracy"], label="Validation", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

# Confusion matrix heatmap
plt.subplot(2, 2, 3)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

# Metrics bar chart
plt.subplot(2, 2, 4)
report = classification_report(y_test, y_pred, target_names=["B", "M"], output_dict=True)
metrics_df = pd.DataFrame(report).transpose()

if len(metrics_df) >= 3:
    metrics_to_plot = metrics_df.iloc[:-3, :-1]
    if not metrics_to_plot.empty:
        metrics_to_plot.plot(kind='bar', ax=plt.gca())
        plt.title("Classification Metrics")
        plt.xlabel("Class")
        plt.ylabel("Score")
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend(loc='lower right')
        plt.ylim([0, 1.1])

plt.tight_layout()
plt.savefig("outputs/breast_cancer_analysis.png", dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved: outputs/breast_cancer_analysis.png")

# Summary
print("\n=== RESULTS SUMMARY ===")
print(f"Final Test Accuracy: {accuracy*100:.1f}%")
print(f"Training Epochs: {len(history.history['loss'])}")
print(f"Model Parameters: {model.count_params()}")
print(f"\nConfusion Matrix Details:")
print(f"  True Positive (TP):  {cm[1][1]:3d} - Malignant correctly identified")
print(f"  True Negative (TN):  {cm[0][0]:3d} - Benign correctly identified")
print(f"  False Positive (FP): {cm[0][1]:3d} - Benign misclassified as Malignant")
print(f"  False Negative (FN): {cm[1][0]:3d} - Malignant misclassified as Benign")
print("\n✓ Analysis complete!")