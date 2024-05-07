/*
  Suriya Kasiyalan Siva
  Spring 2024
  02/09/2024
  CS 5330 Computer Vision
*/
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm> // For sorting

using namespace cv;
using namespace std;

// Function to calculate the whole image color histogram
static Mat ColorHistogram(const Mat& image) {
    Mat hist;
    int channels[] = { 0, 1, 2 };
    int histSize[] = { 8, 8, 8 };
    float range[] = { 0, 256 };
    const float* ranges[] = { range, range, range };
    calcHist(&image, 1, channels, Mat(), hist, 3, histSize, ranges, true, false);
    normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());
    return hist.reshape(1, 1); // Convert histogram to a row matrix
}

// Function to calculate the whole image texture histogram using Sobel operator
static Mat TextureHistogram(const Mat& image) {
    Mat grayImage, gradX, gradY, gradMag, gradAngle;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // Calculate gradients using Sobel operator
    Sobel(grayImage, gradX, CV_32F, 1, 0);
    Sobel(grayImage, gradY, CV_32F, 0, 1);

    // Calculate magnitude and angle of gradients
    cartToPolar(gradX, gradY, gradMag, gradAngle, true);

    // Calculate histogram of gradient magnitudes
    Mat histMag;
    int histSizeMag = 8; // Number of bins
    float rangeMag[] = { 0, 360 }; // Range of angles
    const float* histRangeMag = { rangeMag };
    calcHist(&gradMag, 1, 0, Mat(), histMag, 1, &histSizeMag, &histRangeMag, true, false);

    // Calculate histogram of gradient orientations
    Mat histAngle;
    int histSizeAngle = 8; // Number of bins
    float rangeAngle[] = { 0, 360 }; // Range of angles
    const float* histRangeAngle = { rangeAngle };
    calcHist(&gradAngle, 1, 0, Mat(), histAngle, 1, &histSizeAngle, &histRangeAngle, true, false);

    // Concatenate histograms
    Mat hist;
    hconcat(histMag.reshape(1, 1), histAngle.reshape(1, 1), hist);

    return hist;
}

// Function to calculate distance between two histograms
static double calculateHistogramDistance(const Mat& hist1, const Mat& hist2) {
    return compareHist(hist1, hist2, HISTCMP_CHISQR);
}

// Function to calculate combined distance between color and texture histograms
static double calculateCombinedDistance(const Mat& colorHist1, const Mat& textureHist1,
    const Mat& colorHist2, const Mat& textureHist2,
    double colorWeight = 0.5, double textureWeight = 0.5) {
    double colorDistance = calculateHistogramDistance(colorHist1, colorHist2);
    double textureDistance = calculateHistogramDistance(textureHist1, textureHist2);
    return colorWeight * colorDistance + textureWeight * textureDistance;
}

int main() {
    // Load target image
    Mat targetImage = imread("olympus/pic.0535.jpg");
    if (targetImage.empty()) {
        cerr << "Could not open or find the target image!" << endl;
        return -1;
    }
    
    int numBins = 8;

    // Calculate color and texture histograms for the target image
    Mat targetColorHist = ColorHistogram(targetImage);
    Mat targetTextureHist = TextureHistogram(targetImage);

    // Load database images
    vector<string> imagePaths;
    glob("olympus/*.jpg", imagePaths);
    if (imagePaths.empty()) {
        cerr << "Error: No images found in the database." << endl;
        return -1;
    }

    // Initialize vector to store image paths and distances
    vector<pair<string, double>> imageDistances;

    // Iterate through the database images
    for (const auto& imagePath : imagePaths) {
        // Load image
        Mat image = imread(imagePath);
        if (!image.empty()) {
            // Calculate color and texture histograms for the current image
            Mat colorHist = ColorHistogram(image);
            Mat textureHist = TextureHistogram(image);

            // Calculate combined distance between target and current image
            double combinedDistance = calculateCombinedDistance(targetColorHist, targetTextureHist,
                colorHist, textureHist);

            // Store the image path and distance
            imageDistances.emplace_back(make_pair(imagePath, combinedDistance));
        }
    }

    // Sort the imageDistances vector based on distances in ascending order
    sort(imageDistances.begin(), imageDistances.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    // Display the top 3 similar images
    int topN = 3;
    
    for (int i = 0; i < min(topN, static_cast<int>(imageDistances.size())); ++i) {
        Mat similarImage = imread(imageDistances[i].first);
        if (similarImage.empty()) {
            cout << "Error reading similar image: " << imageDistances[i].first << endl;
            continue;
        }

        namedWindow("Similar Image " + to_string(i + 1), WINDOW_NORMAL);
        imshow("Similar Image " + to_string(i + 1), similarImage);
        waitKey(0);
    }
    
    for (int i = 0; i < min(topN, static_cast<int>(imageDistances.size())); ++i) {
        cout << "Distance: " << imageDistances[i].first << ", Image Path: " << imageDistances[i].second << endl;
    }
    
    waitKey(0);
    return 0;
}
