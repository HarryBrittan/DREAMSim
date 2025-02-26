import ROOT
import sys
sys.path.append("./CMSPLOTS")
from myFunction import Normalize  # Import the Normalize function

# Define the dictionary for histogram names with their labels, titles, axis ranges, and log scaling option
hist_info = {
    'OP_pos_produced_r_op_ratio': {
        'xlabel': 'Radius [cm]',
        'ylabel': 'Trapping Efficiency',
        'title': 'Optical Photons Produced Radius Ratio',
        'xrange': (0, 0.043),  # Example x-axis range
        'yrange': (0, .35)     # Example y-axis range
    },
    'OP_pos_produced_r_op': {
        'xlabel': 'Radius [cm]',
        'ylabel': 'Counts',
        'title': 'Optical Photons Produced Radius',
        'xrange': (0, 0.045),  # Example x-axis range
        'yrange': (0, 20)  # Example y-axis range
    },
    'OP_time_final_op': {
        'xlabel': 'Time [ns]',
        'ylabel': 'Counts (Normalized)',
        'title': 'Final Optical Photons Time',
        'xrange': (4, 12),  # Example x-axis range
        'yrange': (0, 0.3)  # Example y-axis range
    },
    'OP_cosTheta_produced_core_op_ratio': {
        'xlabel': 'cos(Theta)',
        'ylabel': 'Ratio',
        'title': 'Optical Photons Produced cos(Theta) Core Ratio',
        'xrange': (0.3, 1.2),   # Example x-axis range
        'yrange': (0, 1)     # Example y-axis range
    },
    'OP_cosTheta_produced_op_ratio': {
        'xlabel': 'cos(Theta)',
        'ylabel': 'Ratio',
        'title': 'Optical Photons Produced cos(Theta) Ratio',
        'xrange': (0.3, 1.2),   # Example x-axis range
        'yrange': (0, 1)     # Example y-axis range
    },
    'OP_time_delta_op': {
        'xlabel': 'Time Delta [ns]',
        'ylabel': 'Counts (Normalized)',
        'title': 'Optical Photons Time Delta',
        'xrange': (4, 12),   # Example x-axis range
        'yrange': (0, .3)  # Example y-axis range
    },
    'OP_cosTheta_produced_total_op': {
        'xlabel': 'cos(Theta)',
        'ylabel': 'Counts',
        'title': 'Optical Photons Produced cos(Theta) Total',
        'xrange': (0.3, 1.2),   # Example x-axis range
        'yrange': (0, 3800)  # Example y-axis range
    },
    'OP_time_produced_op': {
        'xlabel': 'time (ns)',
        'ylabel': 'Counts (Normalized)',
        'title': 'Optical Photons Time Produced',
        'xrange': (0, .05),   # Example x-axis range
        'yrange': (0, 1)  # Example y-axis range
}}

def plot_histograms(root_file1, root_file2, hist_names, output_pdf):
    # Open the ROOT files
    file1 = ROOT.TFile.Open(root_file1)
    file2 = ROOT.TFile.Open(root_file2)
    
    # Create a canvas
    canvas = ROOT.TCanvas("canvas", "Overlay Histograms", 800, 600)
    
    # Open the PDF file
    canvas.Print(f"{output_pdf}(")
    
    for hist_name in hist_names:
        # Get the histograms from the files
        hist1 = file1.Get(hist_name)
        hist2 = file2.Get(hist_name)
        
        # Remove the statistics box
        hist1.SetStats(0)
        hist2.SetStats(0)

        # Normalize the histograms if the hist_name is 'OP_time_delta_op', 'OP_time_final_op', or 'OP_time_produced_op'
        if hist_name in ['OP_time_delta_op', 'OP_time_final_op', 'OP_time_produced_op']:
            hist1 = Normalize(hist1)
            hist2 = Normalize(hist2)
            if not hist1 or not hist2:
                print(f"Warning: Normalization failed for {hist_name}")
                continue

        print(f"Normalized histograms for {hist_name}")

        # Set the axis ranges using the hist_info dictionary
        x_min, x_max = hist_info[hist_name]['xrange']
        y_min, y_max = hist_info[hist_name]['yrange']
        
        # Ensure the specified range is within the valid range of the histogram axis
        hist1.GetXaxis().SetRangeUser(max(x_min, hist1.GetXaxis().GetXmin()), min(x_max, hist1.GetXaxis().GetXmax()))
        hist1.GetYaxis().SetRangeUser(max(y_min, hist1.GetYaxis().GetXmin()), min(y_max, hist1.GetYaxis().GetXmax()))

        # Draw the histograms
        hist1.SetLineColor(ROOT.kRed)
        hist1.SetLineWidth(2)
        hist1.Draw("HIST")
        
        hist2.SetLineColor(ROOT.kBlue)
        hist2.SetLineWidth(2)
        hist2.Draw("HIST SAME")
        
        # Set the labels and title using the hist_info dictionary
        hist1.GetXaxis().SetTitle(hist_info[hist_name]['xlabel'])
        hist1.GetYaxis().SetTitle(hist_info[hist_name]['ylabel'])
        hist1.SetTitle(hist_info[hist_name]['title'])

        # Add a legend
        legend = ROOT.TLegend(0.5, 0.8, 0.9, 0.9)
        legend.AddEntry(hist1, f'{root_file1}: {hist_name}', "l")
        legend.AddEntry(hist2, f'{root_file2}: {hist_name}', "l")
        legend.Draw()
        
        # Save the canvas to the PDF
        canvas.Print(output_pdf)
    
    # Close the PDF file
    canvas.Print(f"{output_pdf})")
    
    # Close the ROOT files
    file1.Close()
    file2.Close()

# Example usage
root_file1 = 'multiclad_output.root'
root_file2 = 'singleclad_output.root'
hist_names = ['OP_pos_produced_r_op_ratio', 
              'OP_time_final_op',
              'OP_pos_produced_r_op',
              'OP_cosTheta_produced_core_op_ratio',
              'OP_cosTheta_produced_op_ratio',
              'OP_time_delta_op',
              'OP_time_produced_op']
output_pdf = 'histograms_overlay.pdf'

plot_histograms(root_file1, root_file2, hist_names, output_pdf)