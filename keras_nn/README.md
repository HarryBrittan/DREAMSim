## 1D CNN and LSTM Models

This folder contains two main neural network architectures for pulse reconstruction and photon counting: a **1D Convolutional Neural Network (1D CNN)** and a **Long Short-Term Memory (LSTM)** model. Both are designed to process time-series data from calorimeter signals and reconstruct the underlying photon hits.

---

### 1D CNN Model

The 1D CNN model uses convolutional layers to extract local features from the input pulse shapes. It is well-suited for identifying patterns such as photon pulses in noisy time-series data.

**Key features:**
- Input: 1D array representing the convoluted pulse (time bins).
- Architecture: Several 1D convolutional layers followed by a linear output layer.
- Output: Reconstructed pulse or photon hit distribution over time bins.
- Loss: Mean squared error (MSE) between predicted and true pulses.

**Example usage:**
```bash
python 1d_cnn.py
```

---

### LSTM Model

The LSTM model leverages recurrent neural network (RNN) layers, specifically Long Short-Term Memory units, to capture temporal dependencies in the pulse data. This is particularly useful for signals where the timing and sequence of hits are important.

**Key features:**
- Input: 1D array representing the convoluted pulse (time bins).
- Architecture: One or more LSTM layers, possibly followed by dense layers.
- Output: Reconstructed pulse or photon hit distribution over time bins.
- Loss: Mean squared error (MSE) between predicted and true pulses.
- Advantage: LSTM can model long-range dependencies and is robust to varying pulse shapes and timing.

**Example usage:**
```bash
python lstm.py
```

---

Both models can be trained and evaluated using the provided scripts. Results, including performance metrics and plots, are saved to the output directories for further analysis.

## Overview

- **Input:** HDF5 file (`deconvolution_data.h5`) with two datasets:
  - `"convoluted"`: Measured/convoluted pulses (input to the NN)
  - `"truth"`: True underlying pulses (target for the NN)
- **Output:** Diagnostic plots in `nn_output.pdf` and `performance_nn.pdf` showing the NN's performance.

---

## How It Works

### 1. Data Loading

The script loads the convoluted and truth pulses from the HDF5 file into NumPy arrays.

### 2. Train/Test Split

The data is split into training and test sets using scikit-learn's `train_test_split`.

### 3. Model Architecture

The model is a simple 1D convolutional autoencoder/regressor:

- **Input Layer:** Shape = (number of bins,)
- **Reshape Layer:** Adds a channel dimension for Conv1D
- **Conv1D Layer 1:** 32 filters, kernel size 5, ReLU activation
- **Conv1D Layer 2:** 32 filters, kernel size 5, ReLU activation
- **Conv1D Layer 3:** 1 filter, kernel size 5, linear activation (output)
- **Reshape Layer:** Flattens back to (number of bins,)


**Model summary:**
```python

model = keras.Sequential([
    layers.Input(shape=input_shape),
    layers.Reshape((input_shape[0], 1)),  # Add channel dimension
    layers.Conv1D(32, 5, activation="relu", padding="same"),
    layers.Conv1D(32, 5, activation="relu", padding="same"),
    layers.Conv1D(1, 5, activation="linear", padding="same"),
    layers.Reshape((input_shape[0],))
])
```

- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Early Stopping:** Stops training if validation loss does not improve for 20 epochs.

### 5. Training

The model is trained for up to 400 epochs (adjustable), with a batch size of 16. Early stopping is used to prevent overfitting.

### 6. Evaluation & Plots

After training, the script produces several diagnostic plots:

- **Overlayed Pulse Plots:** For 20 test events, overlays the convoluted input, truth, and NN output pulses. Saved to `nn_output.pdf`.
- **Loss Curves:** Training and validation MSE loss vs. epoch.
- **Residuals Histogram:** Distribution of (NN output - truth) across all bins/events.
- **Scatter Plot:** NN output vs. truth for all bins/events, with a y=x reference line.
- **Peak Timing Difference:** Histogram of the difference in peak bin location between truth and NN output.
- **Photon Count Comparison:** Plots and compares the sum of bins (interpreted as "number of photons") for truth and NN output, per event.
- **Photon Count Efficiency:** Plots the ratio (NN/Truth) of photon counts per event.

All performance plots are saved in `performance_nn.pdf`.

---

## Model Explanation

- **Why Conv1D?**
  The pulse is a 1D time series. Conv1D layers are effective at learning local features (like rising/falling edges) and are translation-invariant, making them ideal for pulse deconvolution.

- **Why normalization?**
  Normalizing each pulse helps the network focus on reconstructing the shape and timing, not the absolute amplitude. If amplitude is important, set `NORMALIZE = False`.

- **Loss (MSE):**
  The model minimizes the mean squared error between the predicted and true pulse shapes.

- **ReLU Activation:**
  ReLU (Rectified Linear Unit) is used for non-linearity, defined as `ReLU(x) = max(0, x)`. It helps the network learn complex features.

- **Padding:**
  `"same"` padding ensures the output of each Conv1D layer has the same length as the input, preserving the time structure.

---

## How to Use

1. **Prepare your data:**
   Ensure `deconvolution_data.h5` exists with `"convoluted"` and `"truth"` datasets.

2. **Adjust normalization:**
   Set `NORMALIZE = True` or `False` at the top of the script as needed.

3. **Run the script:**
   ```bash
   python 1d_cnn.py
   ```

4. **Review outputs:**
   - `nn_output.pdf`: Overlay plots of pulses for 20 test events.
   - `performance_nn.pdf`: Diagnostic and performance plots.

---

## Performance Metrics

- **Loss curves** show training progress and potential overfitting.
- **Residuals** and **scatter plots** show how well the NN output matches the truth.
- **Peak timing difference** quantifies how well the NN recovers the pulse timing.
- **Photon count plots** compare the total reconstructed signal to the truth.
- **Photon count efficiency** shows the ratio of reconstructed to true photon counts per event.

---

## Dependencies

- Python 3
- numpy
- h5py
- matplotlib
- scikit-learn
- tensorflow (Keras)

Install with:
```bash
pip install numpy h5py matplotlib scikit-learn tensorflow
```

---

## Contact

For questions or improvements, please open an issue or contact the repository maintainer.

---

## 1D CNN and LSTM Models

This folder contains two main neural network architectures for pulse reconstruction and photon counting: a **1D Convolutional Neural Network (1D CNN)** and a **Long Short-Term Memory (LSTM)** model. Both are designed to process time-series data from calorimeter signals and reconstruct the underlying photon hits.

---

### 1D CNN Model

The 1D CNN model uses convolutional layers to extract local features from the input pulse shapes. It is well-suited for identifying patterns such as photon pulses in noisy time-series data.

**Key features:**
- Input: 1D array representing the convoluted pulse (time bins).
- Architecture: Several 1D convolutional layers followed by a linear output layer.
- Output: Reconstructed pulse or photon hit distribution over time bins.
- Loss: Mean squared error (MSE) between predicted and true pulses.

**Example usage:**
'''
python 1d_cnn.py
'''

---

### LSTM Model

The LSTM model leverages recurrent neural network (RNN) layers, specifically Long Short-Term Memory units, to capture temporal dependencies in the pulse data. This is particularly useful for signals where the timing and sequence of hits are important.

**Key features:**
- Input: 1D array representing the convoluted pulse (time bins).
- Architecture: One or more LSTM layers, possibly followed by dense layers.
- Output: Reconstructed pulse or photon hit distribution over time bins.
- Loss: Mean squared error (MSE) between predicted and true pulses.
- Advantage: LSTM can model long-range dependencies and is robust to varying pulse shapes and timing.

**Example usage:**
'''
python lstm.py
'''

---

Both models can be trained and evaluated using the provided scripts. Results, including performance metrics and plots, are saved to the output directories for further analysis.

