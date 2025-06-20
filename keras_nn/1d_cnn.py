import h5py
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, r2_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve, average_precision_score
from scipy.signal import find_peaks

# Force TensorFlow to use GPU if available, otherwise fall back to CPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Using CPU.")

# Load data (no normalization)
with h5py.File("data/pulse_data.h5", "r") as f:
    X = np.array(f["convoluted"])
    Y = np.array(f["truth"])

# Optionally, split into train/test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Model: simple 1D convolutional autoencoder
input_shape = X_train.shape[1:]

model = keras.Sequential([
    layers.Input(shape=input_shape),
    layers.Reshape((input_shape[0], 1)),  # Add channel dimension
    layers.Conv1D(32, 5, activation="relu", padding="same"),
    layers.Conv1D(32, 5, activation="relu", padding="same"),
    layers.Conv1D(1, 5, activation="linear", padding="same"),
    layers.Reshape((input_shape[0],))
])

model.compile(optimizer="adam", loss="mse")
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(
    X_train, Y_train,
    epochs=500,
    batch_size=128,
    validation_data=(X_test, Y_test),
    callbacks=[early_stop]
)

# Predict and plot a few examples
Y_pred = model.predict(X_test)

# --- Metrics calculation (unnormalized) ---
y_true_flat = Y_test.flatten()
y_pred_flat = Y_pred.flatten()

r2 = r2_score(y_true_flat, y_pred_flat)
print(f"R² score (accuracy): {r2:.4f}")

threshold = 0.5
y_true_bin = (y_true_flat > threshold).astype(int)
y_pred_bin = (y_pred_flat > threshold).astype(int)

precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
accuracy = accuracy_score(y_true_bin, y_pred_bin)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Accuracy: {accuracy:.4f}")

#make 1d_cnn_output directory if it doesn't exist
import os
if not os.path.exists("1d_cnn_output"):
    os.makedirs("1d_cnn_output")

# --- Plotting normalized pulses for visualization ---
with PdfPages("1d_cnn_output/cnn_output.pdf") as pdf:
    for i in range(20):
        plt.figure()
        # Normalize each pulse for plotting
        def norm(arr):
            m = np.max(arr)
            return arr / m if m != 0 else arr
        plt.plot(norm(X_test[i]), label="Convoluted (normalized)")
        plt.plot(norm(Y_test[i]), label="Truth (normalized)")
        plt.plot(norm(Y_pred[i]), label="NN Output (normalized)")
        plt.title(f"Test Event {i} (Normalized for plotting only)")
        plt.xlabel("Time Slot")
        plt.ylabel("Amplitude (normalized)")
        plt.legend()
        pdf.savefig()
        plt.close()
print("Saved plots to cnn_output.pdf")

# --- Performance plots (unnormalized data for metrics) ---
with PdfPages("1d_cnn_output/performance_cnn.pdf") as perf_pdf:
    # Loss curves
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    perf_pdf.savefig()
    plt.close()

    # Residuals histogram
    residuals = Y_pred - Y_test
    plt.figure()
    plt.hist(residuals.flatten(), bins=100, alpha=0.7)
    plt.xlabel('Residual (NN Output - Truth)')
    plt.ylabel('Counts')
    plt.title('Distribution of Residuals')
    plt.yscale('log')
    perf_pdf.savefig()
    plt.close()

    # Peak timing difference histogram
    truth_peaks = np.argmax(Y_test, axis=1)
    pred_peaks = np.argmax(Y_pred, axis=1)
    plt.figure()
    plt.hist(truth_peaks - pred_peaks, bins=30)
    plt.xlabel('Truth Peak Bin - NN Peak Bin')
    plt.ylabel('Counts')
    plt.title('Peak Timing Difference')
    plt.yscale('log')
    perf_pdf.savefig()
    plt.close()

    # Photon count correlation: 2D histogram (unnormalized, integer bins only)
    truth_photon_counts = np.sum(Y_test, axis=1).astype(int)
    nn_photon_counts = []
    for event in Y_pred:
        peaks, properties = find_peaks(event, height=1)
        count = int(np.round(properties['peak_heights']).sum()) if len(peaks) > 0 else 0
        nn_photon_counts.append(count)
    nn_photon_counts = np.array(nn_photon_counts, dtype=int)

    # Define integer bin edges for both axes
    max_truth = truth_photon_counts.max()
    max_nn = nn_photon_counts.max()
    bins_truth = np.arange(0, max_truth + 2) - 0.5
    bins_nn = np.arange(0, max_nn + 2) - 0.5

    plt.figure()
    plt.hist2d(truth_photon_counts, nn_photon_counts, bins=[bins_truth, bins_nn], cmap='viridis')
    plt.xlabel("Truth photon count (sum of bins, unnormalized)")
    plt.ylabel("NN photon count (sum of peaks, unnormalized)")
    plt.title("Photon count correlation: NN vs Truth (unnormalized, integer bins)")
    plt.colorbar(label="Counts")
    plt.plot([0, max_truth], [0, max_truth], 'r--', label="y=x")
    plt.legend()
    perf_pdf.savefig()
    plt.close()

    # Confusion matrix for binarized output
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Hit", "Hit"])
    plt.figure()
    disp.plot(values_format='d', cmap='Blues')
    plt.title(f"Confusion Matrix\nAccuracy={accuracy:.2f}, Precision={precision:.2f}, Recall={recall:.2f}")
    perf_pdf.savefig()
    plt.close()


    # Flatten arrays for global metrics
    y_true_flat = Y_test.flatten()
    y_pred_flat = Y_pred.flatten()


    # For precision, binarize the outputs (e.g., threshold at 0.5)
    threshold = 0.5
    y_true_bin = (y_true_flat > threshold).astype(int)
    # y_pred_flat is used as-is (do NOT threshold for ROC/PR curves)

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_true_bin, y_pred_flat)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    perf_pdf.savefig()
    plt.close()


    # --- Precision-Recall Curve ---
    precision_curve, recall_curve, _ = precision_recall_curve(y_true_bin, y_pred_flat)
    avg_precision = average_precision_score(y_true_bin, y_pred_flat)
    plt.figure()
    plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'AP = {avg_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    perf_pdf.savefig()
    plt.close()

# R² score (regression "accuracy")
r2 = r2_score(y_true_flat, y_pred_flat)
print(f"R² score (accuracy): {r2:.4f}")


precision = precision_score(y_true_bin, y_pred_flat, zero_division=0)
recall = recall_score(y_true_bin, y_pred_flat, zero_division=0)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

print("Saved performance plots to performance_cnn.pdf")



