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
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from scipy.signal import find_peaks

# Force TensorFlow to use GPU if available, otherwise fall back to CPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Using CPU.")

# Set this to True to normalize each pulse, False to use raw amplitudes
NORMALIZE = True

# Load data
with h5py.File("data/pulse_data.h5", "r") as f:
    X = np.array(f["convoluted"])
    Y = np.array(f["truth"])

# Normalize each sample to its maximum if requested
if NORMALIZE:
    def normalize(arr):
        arr_max = np.max(arr, axis=1, keepdims=True)
        arr_max[arr_max == 0] = 1  # Prevent division by zero
        return arr / arr_max
    X = normalize(X)
    Y = normalize(Y)

# Optionally, split into train/test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Model: LSTM-based regressor
input_shape = X_train.shape[1:]

model = keras.Sequential([
    layers.Input(shape=input_shape),
    layers.Reshape((input_shape[0], 1)),  # LSTM expects 3D input: (batch, timesteps, features)
    layers.LSTM(64, return_sequences=True),
    layers.TimeDistributed(layers.Dense(1)),  # Output a value per time bin
    layers.Reshape((input_shape[0],))
])

model.compile(optimizer="adam", loss="mse")
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(
    X_train, Y_train,
    epochs=1000,
    batch_size=128,
    validation_data=(X_test, Y_test),
    callbacks=[early_stop]
)

# Predict and plot a few examples
Y_pred = model.predict(X_test)

# Set the range of bins to plot (adjust as needed)
PLOT_START = 0
PLOT_END = X_test.shape[1]  # This will be 200
# make folder for lstm_output if it doesn't exist
import os
if not os.path.exists("lstm_output"):
    os.makedirs("lstm_output")
with PdfPages("lstm_output/lstm_output.pdf") as pdf:
    for i in range(20):
        plt.figure()
        plt.plot(
            np.arange(PLOT_START, PLOT_END),
            X_test[i][PLOT_START:PLOT_END],
            label="Convoluted"
        )
        plt.plot(
            np.arange(PLOT_START, PLOT_END),
            Y_test[i][PLOT_START:PLOT_END],
            label="Truth"
        )
        plt.plot(
            np.arange(PLOT_START, PLOT_END),
            Y_pred[i][PLOT_START:PLOT_END],
            label="NN Output"
        )
        plt.title(f"Test Event {i} (Normalized: {NORMALIZE})")
        plt.xlabel("Time Slot")
        plt.ylabel("Amplitude")
        plt.legend()
        pdf.savefig()
        plt.close()
print("Saved plots to lstm_output/lstm_output.pdf")

# Save performance plots to a separate PDF
with PdfPages("lstm_output/performance_lstm.pdf") as perf_pdf:
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
    #set log scale for better visibility
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

    # --- Photon count correlation: 2D histogram (unnormalized, integer bins only) ---
    truth_photon_counts = np.sum(Y_test, axis=1).astype(int)
    nn_photon_counts = []
    for event in Y_pred:
        peaks, properties = find_peaks(event, height=1)
        # For each peak, round the height to the nearest integer and sum
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
    plt.xlabel("Truth photon count, (unnormalized sum of bins)")
    plt.ylabel("NN photon count, (sum of peaks, unnormalized)")
    plt.title("Photon count correlation: NN vs Truth (Unnormalized, integer bins)")
    plt.colorbar(label="Counts")
    plt.plot([0, max(max_truth, max_nn)], [0, max(max_truth, max_nn)], 'r--', label="y=x")
    plt.legend()
    plt.tight_layout()
    plt.show()


    # Flatten arrays for global metrics
    y_true_flat = Y_test.flatten()
    y_pred_flat = Y_pred.flatten()

    # For precision, binarize the outputs (e.g., threshold at 0.5)
    threshold = 0.5
    y_true_bin = (y_true_flat > threshold).astype(int)
    y_pred_bin = (y_pred_flat > threshold).astype(int)

    # Compute accuracy, precision, recall for confusion matrix
    accuracy = accuracy_score(y_true_bin, y_pred_bin)
    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)

    # Confusion matrix for binarized output
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Hit", "Hit"])
    plt.figure()
    disp.plot(values_format='d', cmap='Blues')
    plt.title(f"Confusion Matrix\nAccuracy={accuracy:.2f}, Precision={precision:.2f}, Recall={recall:.2f}")
    perf_pdf.savefig()
    plt.close()

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_true_bin, y_pred_bin)
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
    precision_curve, recall_curve, _ = precision_recall_curve(y_true_bin, y_pred_bin)
    avg_precision = average_precision_score(y_true_bin, y_pred_bin)
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




