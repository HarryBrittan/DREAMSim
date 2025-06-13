import h5py
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Set this to True to normalize each pulse, False to use raw amplitudes
NORMALIZE = True

# Load data
with h5py.File("deconvolution_data.h5", "r") as f:
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

# Model: simple 1D convolutional autoencoder
input_shape = X_train.shape[1:]

# description of the model:
# - Input layer with shape (n_bins,)
# - Reshape to add a channel dimension (1D Conv expects 3D input)
# - Three Conv1D layers with ReLU activation
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

history = model.fit(
    X_train, Y_train,
    epochs=3000,
    batch_size=16,
    validation_data=(X_test, Y_test)
)

# Predict and plot a few examples
Y_pred = model.predict(X_test)

# Set the range of bins to plot (adjust as needed)
PLOT_START = 0    # e.g., 0 for the first bin
PLOT_END = 500    # e.g., 200 for the 200th bin (or None for all bins)

with PdfPages("nn_output.pdf") as pdf:
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
print("Saved plots to nn_output.pdf")

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

residuals = Y_pred - Y_test
plt.figure()
plt.hist(residuals.flatten(), bins=100, alpha=0.7)
plt.xlabel('Residual (NN Output - Truth)')
plt.ylabel('Counts')
plt.title('Distribution of Residuals')
plt.show()

plt.figure()
plt.scatter(Y_test.flatten(), Y_pred.flatten(), alpha=0.3, s=2)
plt.xlabel('Truth')
plt.ylabel('NN Output')
plt.title('NN Output vs. Truth')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--')  # y=x line
plt.show()

truth_peaks = np.argmax(Y_test, axis=1)
pred_peaks = np.argmax(Y_pred, axis=1)
plt.figure()
plt.hist(truth_peaks - pred_peaks, bins=30)
plt.xlabel('Truth Peak Bin - NN Peak Bin')
plt.ylabel('Counts')
plt.title('Peak Timing Difference')
plt.show()

# Calculate number of photons (sum of bins) for each event
truth_photons = np.sum(Y_test, axis=1)
nn_photons = np.sum(Y_pred, axis=1)

# Plot per event (first 20 events)
plt.figure()
plt.plot(truth_photons[:20], label="Truth photons", marker='o')
plt.plot(nn_photons[:20], label="NN photons", marker='x')
plt.xlabel("Event")
plt.ylabel("Total photon count (sum of bins)")
plt.title("Photon count per event (first 20 events)")
plt.legend()
plt.show()


