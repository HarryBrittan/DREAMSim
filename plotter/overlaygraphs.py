import ROOT
import array
import sys
#sys.path.append("./macros")


def overlay_graphs():
    # Open the ROOT files
    file1 = ROOT.TFile.Open("/home/harryb/HEP/g4/calox/myDREAMSim/DREAMSim/plotter/macros/circle.root")
    file2 = ROOT.TFile.Open("multiclad_output.root")
    # Alternatively, you can use singleclad_output.root
    # file2 = ROOT.TFile.Open("singleclad_output.root")

    # Retrieve the TGraph from circle.root
    outer_graph = file1.Get("outerCircle_0")  
    inner_graph = file1.Get("innerCircle_0")  
    

    # Retrieve the 2D histogram from multiclad_output.root
    hist = file2.Get("OP_pos_produced_x_vs_y_op")  # Replace "OP_pos_final_x_vs_y_op" with the actual name of the 2D histogram
 
    """
    # Print the points of the TGraph for debugging
    print("TGraph points:")
    for i in range(graph.GetN()):
        x, y = array.array('d', [0]), array.array('d', [0])
        graph.GetPoint(i, x, y)
        print(f"Point {i}: x = {x[0]}, y = {y[0]}")
    """
 
    # Create a canvas to draw the histogram and graph
    c = ROOT.TCanvas("c", "Overlay TGraph on 2D Histogram", 800, 600)

    # Draw the 2D histogram
    hist.Draw("COLZ")

    # Draw the TGraph on top of the 2D histogram
    inner_graph.SetMarkerColor(ROOT.kRed)
    #graph.SetMarkerStyle(20)
    inner_graph.Draw("L SAME")

    outer_graph.SetMarkerColor(ROOT.kRed)
    #graph.SetMarkerStyle(20)
    outer_graph.Draw("L SAME")



    # Save the canvas as an image file
    c.SaveAs("overlay.png")

    # Close the ROOT files
    file1.Close()
    file2.Close()

# Example usage
overlay_graphs()