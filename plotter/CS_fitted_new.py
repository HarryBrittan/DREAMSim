import ROOT
import argparse
import sys

def fit_and_draw_histogram(hist, output_png):
    # Extract (x, y) points from the 2D histogram, weighted by bin content
    x_vals = []
    y_vals = []
    min_content = 1e-6  # Only consider bins with nonzero content
    for ix in range(1, hist.GetNbinsX() + 1):
        for iy in range(1, hist.GetNbinsY() + 1):
            content = hist.GetBinContent(ix, iy)
            if content > min_content:
                x = hist.GetXaxis().GetBinCenter(ix)
                y = hist.GetYaxis().GetBinCenter(iy)
                # Add each (x, y) point as many times as its content (rounded)
                for _ in range(int(round(content))):
                    x_vals.append(x)
                    y_vals.append(y)
    import numpy as np
    if len(x_vals) < 2:
        print("Not enough points for PCA fit.")
        return None, None
    # Stack as 2D array for PCA
    data = np.vstack((x_vals, y_vals))
    # Subtract mean
    mean = np.mean(data, axis=1, keepdims=True)
    data_centered = data - mean
    # Compute covariance and eigenvectors
    cov = np.cov(data_centered)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # The principal component is the eigenvector with the largest eigenvalue
    pc = eigvecs[:, np.argmax(eigvals)]
    # Slope of the major axis
    slope = pc[1] / pc[0] if pc[0] != 0 else np.inf
    # Intercept: pass through the mean point
    intercept = mean[1, 0] - slope * mean[0, 0]
    # Draw the original 2D histogram
    canvas = ROOT.TCanvas("canvas", "PCA Fit Result", 800, 600)
    hist.Draw("COLZ")
    # Draw the contours on top
    hist.SetContour(5)  # Number of contour levels (adjust as needed)
    hist.Draw("CONT3 SAME")
    # Draw the major axis (PCA fit)
    x_min = hist.GetXaxis().GetXmin()
    x_max = hist.GetXaxis().GetXmax()
    fit_line = ROOT.TF1("pca_fit", f"{slope}*x+{intercept}", x_min, x_max)
    fit_line.SetLineColor(ROOT.kBlue)
    fit_line.SetLineStyle(2)
    fit_line.Draw("SAME")
    # Move the stats box to the top left
    canvas.Update()
    stats = hist.FindObject("stats")
    if stats:
        stats.SetX1NDC(0.1)
        stats.SetX2NDC(0.3)
        stats.SetY1NDC(0.7)
        stats.SetY2NDC(0.9)
        stats.SetTextColor(ROOT.kBlack)
        canvas.Modified()
        canvas.Update()
    # Save the canvas as a PNG
    canvas.SaveAs(output_png)
    return slope, intercept

def main():
    parser = argparse.ArgumentParser(description="Fit a linear function to a 2D histogram in a ROOT file and save the result.")
    parser.add_argument("filename", help="Input ROOT file")
    parser.add_argument("--histname", default="total_edep_histogram", help="Name of the 2D histogram (default: total_edep_histogram)")
    parser.add_argument("--output", default="fit_result.png", help="Output PNG filename (default: fit_result.png)")
    args = parser.parse_args()

    # Open the ROOT file
    root_file = ROOT.TFile(args.filename, "READ")
    hist = root_file.Get(args.histname)
    if not hist:
        print(f"Histogram '{args.histname}' not found in {args.filename}")
        sys.exit(1)

    # Fit and draw
    slope, intercept = fit_and_draw_histogram(hist, args.output)
    print(f"Slope: {slope}, Intercept: {intercept}")

    root_file.Close()

if __name__ == "__main__":
    main()

