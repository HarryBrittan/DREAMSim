import ROOT
import array

def overlay_graphs():
    # Open the ROOT files
    file1 = ROOT.TFile.Open("circle.root")
    file2 = ROOT.TFile.Open("multiclad_output.root")
    # Alternatively, you can use singleclad_output.root
    # file2 = ROOT.TFile.Open("singleclad_output.root")

    # Retrieve the TGraph from circle.root
    graph = file1.Get("outerCircle_0")  # Replace "innerCircle_0" with the actual name of the TGraph
    if not graph:
        print("Error: TGraph not found in circle.root")
        return

    # Retrieve the 2D histogram from multiclad_output.root
    hist = file2.Get("OP_pos_final_x_vs_y_op")  # Replace "OP_pos_final_x_vs_y_op" with the actual name of the 2D histogram
    if not hist:
        print("Error: 2D histogram not found in multiclad_output.root")
        return

    # Check if the TGraph has points
    if graph.GetN() == 0:
        print("Error: TGraph has no points to display")
        return

    # Print the points of the TGraph for debugging
    print("TGraph points:")
    for i in range(graph.GetN()):
        x, y = array.array('d', [0]), array.array('d', [0])
        graph.GetPoint(i, x, y)
        print(f"Point {i}: x = {x[0]}, y = {y[0]}")

    # Create a canvas to draw the histogram and graph
    c = ROOT.TCanvas("c", "Overlay TGraph on 2D Histogram", 800, 600)

    # Draw the 2D histogram
    hist.Draw("COLZ")

    # Draw the TGraph on top of the 2D histogram
    graph.SetMarkerColor(ROOT.kRed)
    #graph.SetMarkerStyle(20)
    graph.Draw("P SAME")

    # Save the canvas as an image file
    c.SaveAs("overlay.png")

    # Close the ROOT files
    file1.Close()
    file2.Close()

# Example usage
overlay_graphs()