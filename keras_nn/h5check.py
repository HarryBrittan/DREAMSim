import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

with h5py.File("deconvolution_data.h5", "r") as f, PdfPages("pulse_comparison.pdf") as pdf:
    n_events = f["convoluted"].shape[0]
    num_events = min(100, n_events)  # Save up to 100 events
    print(f"Number of events in file: {n_events}")

    for i in range(num_events):
        conv = f["convoluted"][i]
        truth = f["truth"][i]

        # Normalize both to their max
        conv_norm = conv / conv.max() if conv.max() != 0 else conv
        truth_norm = truth / truth.max() if truth.max() != 0 else truth

        plt.figure()
        plt.plot(conv_norm, label="Convoluted (normalized)")
        plt.plot(truth_norm, label="Truth (normalized)")
        plt.xlabel("Bin")
        plt.ylabel("Normalized Content")
        plt.title(f"Event {i}: Convoluted vs Truth (Normalized)")
        plt.legend()
        pdf.savefig()
        plt.close()
    print("Saved plots to pulse_comparison.pdf")
