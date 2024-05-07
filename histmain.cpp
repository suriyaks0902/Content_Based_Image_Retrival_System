/*
  Suriya Kasiyalan Siva & Saikiran Juttu
  Spring 2024
  02/06/2024
  CS 5330 Computer Vision
*/
//////// this uses 2d histogram (R and G channels)

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

 //Function to calculate a 2D normalized color histogram from the red (R) and green (G) channels of an image

Mat chromaticHist(const Mat& image, int numBins) {
    Mat floatImage;
    Mat rgChromatic;
    Mat hist;

    // Convert image to floating point
    image.convertTo(floatImage, CV_32F);

    // Split image channels
    vector<Mat> channels;
    split(floatImage, channels);

    // Calculate chromaticity
    divide(channels[2], channels[2] + channels[1] + channels[0], rgChromatic, 1.0, CV_32F);

    // Define histogram range
    float range[] = { 0, 1 };
    const float* histRange = { range };

    // Calculate histogram
    calcHist(&rgChromatic, 1, 0, Mat(), hist, 1, &numBins, &histRange, true, false);

    // Normalize histogram
    normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());

    return hist;
}
// Function to calculate histogram intersection distance between two histograms
static double calculateHistogramIntersection(const Mat& hist1, const Mat& hist2) {
    double intersection = 0.0;

    if (hist1.size == hist2.size) {
        for (int i = 0; i < hist1.rows; ++i) {
            for (int j = 0; j < hist1.cols; ++j) {
                intersection += min(hist1.at<float>(i, j), hist2.at<float>(i, j));
            }
        }
    }
    else {
        cerr << "Error: Histograms have different dimensions." << endl;
        return -1.0; // Return negative value to indicate error
    }

    return intersection;
}

// Function to find the most similar images based on histogram intersection distance
static vector<pair<double, String>> findSimilarImages(const Mat& targetHist, const vector<String>& imagePaths, int bins) {
    vector<pair<double, String>> distances;

    for (const auto& imagePath : imagePaths) {
        Mat currentImage = imread(imagePath);

        if (currentImage.empty()) {
            cout << "Error reading image: " << imagePath << endl;
            continue;
        }

        //Mat currentHist = calculateHistogram(currentImage, bins);
        Mat currentHist = chromaticHist(currentImage, bins);

        double distance = ( 1.0 - calculateHistogramIntersection(targetHist, currentHist));

        distances.push_back({ distance, imagePath });
    }

    sort(distances.begin(), distances.end());

    return distances;
}

int main() {
    String targetImagePath = "olympus/pic.0164.jpg";
    String directoryPath = "olympus";

    int bins = 8;  // Number of bins for the histogram

    cout << "Reading target image..." << endl;
    Mat targetImage = imread(targetImagePath);

    if (targetImage.empty()) {
        cout << "Error reading target image." << endl;
        return -1;
    }

    cout << "Target image read successfully." << endl;

    Mat targetHist = chromaticHist(targetImage, bins);

    vector<String> imagePaths;
    glob(directoryPath + "/*.jpg", imagePaths);

    vector<pair<double, String>> similarImages = findSimilarImages(targetHist, imagePaths, bins);

    // Sort the distances in ascending order
    sort(similarImages.begin(), similarImages.end());

    namedWindow("Target Image", WINDOW_NORMAL);
    imshow("Target Image", targetImage);
    waitKey(0);

    int topN = 4;
    for (int i = 0; i < min(topN, static_cast<int>(similarImages.size())); ++i) {
        Mat similarImage = imread(similarImages[i].second);

        if (similarImage.empty()) {
            cout << "Error reading similar image: " << similarImages[i].second << endl;
            continue;
        }
        //cout << "Distance: " << similarImages[i].first << " | Image Path: " << similarImages[i].second << endl;
        namedWindow("Similar Image " + to_string(i + 1), WINDOW_NORMAL);
        imshow("Similar Image " + to_string(i + 1), similarImage);
        //cout << "Distance: " << similarImages[i].first << " | Image Path: " << similarImages[i].second << endl;
    }
    int i = 0;
    cout << "Distance: " << i << ", Image Path: " << similarImages[i].second << endl;

    for (int i = 1; i < min(topN, static_cast<int>(similarImages.size())); ++i) {
        
        cout << "Distance: " << similarImages[i].first << ", Image Path: " << similarImages[i].second << endl;

    }waitKey(0);
    return 0;
}
