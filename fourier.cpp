/*
  Suriya Kasiyalan Siva
  Spring 2024
  02/11/2024
  CS 5330 Computer Vision
*/

#include <iostream> // Input/output stream
#include <vector> // Vector container
#include <string> // String handling
#include <opencv2/opencv.hpp> // OpenCV library
#include <dirent.h> // Directory handling
#include <cmath> // Math functions, used for logarithm

using namespace std;
using namespace cv;

// Function to compute features from a given image using Fourier Transform
Mat computeFeatures(const Mat& image) {
    // Compute Fourier Transform
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY); // Convert image to grayscale
    Mat padded; // Expand input image to optimal size
    int m = getOptimalDFTSize(grayImage.rows);
    int n = getOptimalDFTSize(grayImage.cols);
    copyMakeBorder(grayImage, padded, 0, m - grayImage.rows, 0, n - grayImage.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexImage;
    merge(planes, 2, complexImage); // Add to the expanded another plane with zeros
    dft(complexImage, complexImage); // Fourier transform
    split(complexImage, planes); // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))

    // Compute magnitude spectrum
    Mat mag;
    magnitude(planes[0], planes[1], mag); // sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)
    mag += Scalar::all(1);
    log(mag, mag);

    // Resize the power spectrum to 16x16 image
    resize(mag, mag, Size(16, 16));

    // Normalize the magnitude spectrum
    normalize(mag, mag, 0, 1, NORM_MINMAX);

    return mag.reshape(0, 1);  // Flatten to a single row
}

// Function to compute the squared chi distance between two 16x16 feature matrices
double squaredChiDistance(const Mat& feature1, const Mat& feature2) {
    double distance = 0.0;
    for (int i = 0; i < feature1.rows; ++i) {
        for (int j = 0; j < feature1.cols; ++j) {
            double diff = feature1.at<float>(i, j) - feature2.at<float>(i, j);
            double sum = feature1.at<float>(i, j) + feature2.at<float>(i, j);
            if (sum != 0)
                distance += diff * diff / sum;
        }
    }
    return distance;
}

// Custom comparison function for sorting similarity scores
bool compareScores(const pair<double, string>& a, const pair<double, string>& b) {
    return a.first < b.first; // Sort by distance in ascending order
}

int main() {
    // Read target image
    Mat targetImage = imread("/media/sakiran/Internal/2nd Semester/PRCV/Project/Fourier/olympus/pic.0040.jpg");
    if (targetImage.empty()) {
        cerr << "Error: Could not read target image." << endl;
        return 1;
    }

    // Compute features of target image
    Mat targetFeatures = computeFeatures(targetImage);

    // Loop over the directory of images
    string directoryPath = "/media/sakiran/Internal/2nd Semester/PRCV/Project/Fourier/olympus";
    vector<pair<double, string>> similarityScores;

    // Open the directory
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(directoryPath.c_str())) != nullptr) {
        // Iterate over files in the directory
        while ((ent = readdir(dir)) != nullptr) {
            string filename = ent->d_name;
            if (filename == "." || filename == "..") {
                continue; // Skip current and parent directories
            }
            // Read image
            string imagePath = directoryPath + "/" + filename;
            Mat image = imread(imagePath);
            if (image.empty()) {
                cerr << "Error: Could not read image " << filename << endl;
                continue;
            }
            // Compute features of current image
            Mat imageFeatures = computeFeatures(image);
            // Compute similarity score (Squared Chi distance)
            double distance = squaredChiDistance(targetFeatures, imageFeatures);
            // Store similarity score and image path
            similarityScores.push_back(make_pair(distance, imagePath));
        }
        closedir(dir);
    } else {
        cerr << "Error: Could not open directory." << endl;
        return 1;
    }

    // Sort the list of matches
    sort(similarityScores.begin(), similarityScores.end(), compareScores);

    // Return top N matches (here, N = 5)
    int topN = 10;
    cout << "Top " << topN << " similar images:" << endl;
    for (int i = 0; i < min(topN, static_cast<int>(similarityScores.size())); ++i) {
        cout << similarityScores[i].second << " - Distance: " << similarityScores[i].first << endl;
        // Display similar images
        Mat similarImage = imread(similarityScores[i].second);
        if (!similarImage.empty()) {
            imshow("Similar Image " + to_string(i+1), similarImage);
            waitKey(0); // Wait for any key press
        }
    }

    return 0;
}
