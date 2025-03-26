#include "TCanvas.h"
#include "TGraph.h"
#include "TAxis.h"
#include "TMath.h"

void circles() {
    // Parameters
    int nPoints = 100;      // Number of points to approximate the circles
    double r1 = 0.039;          // Radius of the inner circle
    double r2 = 0.04;          // Radius of the outer circle

    double R = r2 * 2.0 + 0.01; // 10 micron gap between central and peripheral fibers
    double cx1 = R * cos(30.0*M_PI/180 );
    double cy1 = R * sin(30.0*M_PI/180);

    std::vector<std::pair<double,double>> centers = {
	    {0,0},{cx1,cy1},{-cx1,cy1},{0,-R},{cx1,-cy1},{0,R},{-cx1,-cy1}};


    TCanvas *c1 = new TCanvas("c1", "CIRCLES!", 600, 600);
    c1->DrawFrame(-1.5, -1.5, 1.5, 1.5, "CIRCLES!!");

    //Open a ROOT file to save the output
    TFile *file = new TFile("circle.root", "RECREATE");


    //loop over each center and draw the concentric circles
    for (size_t i=0; i < centers.size(); ++i) {
	    double centerX = centers[i].first;
	    double centerY = centers[i].second;

    		// Create arrays to store x and y points for each circle
    	    double xInner[nPoints], yInner[nPoints];
	    double xOuter[nPoints], yOuter[nPoints];

    		// Generate points for the inner circle
    	for (int i = 0; i < (nPoints); ++i) {
        	double theta = 2 * TMath::Pi() * i / (nPoints-1);
        	xInner[i] = centerX + r1 * TMath::Cos(theta);
        	yInner[i] = centerY + r1 * TMath::Sin(theta);
    	}

    		// Generate points for the outer circle
    	for (int i = 0; i < (nPoints); ++i) {
        	double theta = 2 * TMath::Pi() * i / (nPoints-1);
        	xOuter[i] = centerX + r2 * TMath::Cos(theta);
        	yOuter[i] = centerY + r2 * TMath::Sin(theta);
    	}

    	// Create TGraph objects for each circle
    	TGraph *innerCircle = new TGraph(nPoints, xInner, yInner);
    	TGraph *outerCircle = new TGraph(nPoints, xOuter, yOuter);

    	// Set line colors
    	innerCircle->SetLineColor(kBlue);
    	outerCircle->SetLineColor(kRed);

    	// Draw circles
    	innerCircle->Draw("L same");  // "L" draws lines between points
    	outerCircle->Draw("L same"); // "same" overlays on the same canvas


    	innerCircle->Write(Form("innerCircle_%zu", i));	
    	outerCircle->Write(Form("outerCircle_%zu", i));	
}

c1->Write();
file->Close();

c1->Update();
}
