import numpy as np
import ROOT

# Settings
nBins = 500
time_min = 0
time_max = 20
x = np.linspace(time_min, time_max, nBins)

# 1. Gaussian pulse
gauss_mean = 6
gauss_sigma = 0.3
gauss_pulse = np.exp(-0.5 * ((x - gauss_mean) / gauss_sigma) ** 2)

# 2. Landau pulse (ROOT's Landau)
landau_mean = 6
landau_sigma = 0.2
landau_pulse = np.array([ROOT.TMath.Landau(xx, landau_mean, landau_sigma, True) for xx in x])

# 3. Asymmetric double exponential (fast rise, slow fall)
tau_rise = 1
tau_fall = 2
t0 = 6
asym_pulse = np.where(
    x < t0,
    np.exp((x - t0) / tau_rise),
    np.exp(-(x - t0) / tau_fall)
)
asym_pulse /= asym_pulse.max()  # Normalize

# Normalize all pulses to max 1
gauss_pulse /= gauss_pulse.max()
landau_pulse /= landau_pulse.max()

# Add Gaussian noise to each pulse
noise_level = 0.05  # Adjust for more/less noise
rng = np.random.default_rng(seed=42)
gauss_pulse_noisy = gauss_pulse + rng.normal(0, noise_level, size=gauss_pulse.shape)
landau_pulse_noisy = landau_pulse + rng.normal(0, noise_level, size=landau_pulse.shape)
asym_pulse_noisy = asym_pulse + rng.normal(0, noise_level, size=asym_pulse.shape)

# Optionally, clip to keep non-negative
gauss_pulse_noisy = np.clip(gauss_pulse_noisy, 0, None)
landau_pulse_noisy = np.clip(landau_pulse_noisy, 0, None)
asym_pulse_noisy = np.clip(asym_pulse_noisy, 0, None)

# Save to ROOT file as TH1D
outfile = ROOT.TFile("different_pulses.root", "RECREATE")
h_gauss = ROOT.TH1D("h_gauss", "Gaussian Pulse", nBins, time_min, time_max)
h_landau = ROOT.TH1D("h_landau", "Landau Pulse", nBins, time_min, time_max)
h_asym = ROOT.TH1D("h_asym", "Asymmetric Double Exponential Pulse", nBins, time_min, time_max)
h_gauss_noisy = ROOT.TH1D("h_gauss_noisy", "Gaussian Pulse (noisy)", nBins, time_min, time_max)
h_landau_noisy = ROOT.TH1D("h_landau_noisy", "Landau Pulse (noisy)", nBins, time_min, time_max)
h_asym_noisy = ROOT.TH1D("h_asym_noisy", "Asymmetric Double Exponential Pulse (noisy)", nBins, time_min, time_max)

for i in range(nBins):
    h_gauss.SetBinContent(i+1, gauss_pulse[i])
    h_landau.SetBinContent(i+1, landau_pulse[i])
    h_asym.SetBinContent(i+1, asym_pulse[i])
    h_gauss_noisy.SetBinContent(i+1, gauss_pulse_noisy[i])
    h_landau_noisy.SetBinContent(i+1, landau_pulse_noisy[i])
    h_asym_noisy.SetBinContent(i+1, asym_pulse_noisy[i])

h_gauss.Write()
h_landau.Write()
h_asym.Write()
h_gauss_noisy.Write()
h_landau_noisy.Write()
h_asym_noisy.Write()
outfile.Close()

print("Wrote different_pulses.root with clean and noisy Gaussian, Landau, and Asymmetric pulses.")
