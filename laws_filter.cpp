/*
  Suriya Kasiyalan Siva
  Spring 2024
  02/11/2024
  CS 5330 Computer Vision
*/

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <dirent.h>

using namespace std;
using namespace cv;

// Function to compute texture energy measures using Laws' masks from a given image
Mat computeFeatures(const Mat& image) {
    // Assuming the 7x7 square is at the center
    int startX = image.cols / 2 - 3;
    int startY = image.rows / 2 - 3;
    Rect roi(startX, startY, 7, 7);

    // Extract 7x7 square from the image
    Mat square = image(roi).clone();

    // Convert to grayscale
    Mat graySquare;
    cvtColor(square, graySquare, COLOR_BGR2GRAY);

    // Initialize Laws' masks
    vector<Mat> masks;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            Mat mask = Mat::zeros(7, 7, CV_32F);
            mask.at<float>(i, 3) = 1;
            mask.at<float>(3, j) = 1;
            masks.push_back(mask);
        }
    }

    // Compute texture energy measures
    Mat features(1, masks.size(), CV_32F);
    for (size_t i = 0; i < masks.size(); ++i) {
        Mat filtered;
        filter2D(graySquare, filtered, CV_32F, masks[i]);
        double energy = sum(filtered.mul(filtered))[0]; // Compute energy
        features.at<float>(0, i) = energy;
    }

    return features;
}

// Function to compute Squared Chi-Squared Distance (SQI Chi) between two feature vectors
double squaredChiSquaredDistance(const Mat& features1, const Mat& features2) {
    double distance = 0.0;
    for (int i = 0; i < features1.cols; ++i) {
        double sum = features1.at<float>(0, i) + features2.at<float>(0, i);
        if (sum != 0) {
            double term = pow(features1.at<float>(0, i) - features2.at<float>(0, i), 2) / sum;
            distance += term;
        }
    }
    return distance;
}

int main() {
    // Read target image
    Mat targetImage = imread("/media/sakiran/Internal/2nd Semester/PRCV/Project/task1/olympus/pic.0023.jpg");
    if (targetImage.empty()) {
        cerr << "Error: Could not read target image." << endl;
        return 1;
    }

    // Compute features of target image
    Mat targetFeatures = computeFeatures(targetImage);

    // Loop over the directory of images
    string directoryPath = "/media/sakiran/Internal/2nd Semester/PRCV/Project/task1/olympus";
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
            // Compute similarity score
            double distance = squaredChiSquaredDistance(targetFeatures, imageFeatures);
            // Store similarity score and image path
            similarityScores.push_back(make_pair(distance, imagePath));
        }
        closedir(dir);
    } else {
        cerr << "Error: Could not open directory." << endl;
        return 1;
    }

    // Sort the list of matches
    sort(similarityScores.begin(), similarityScores.end());

    // Return top 3 matches
    int topN = 4;
    cout << "Top " << topN << " similar images:" << endl;
    for (int i = 0; i < min(topN, static_cast<int>(similarityScores.size())); ++i) {
        cout << similarityScores[i].second << " - Distance: " << similarityScores[i].first << endl;
        // Display the similar images
        Mat similarImage = imread(similarityScores[i].second);
        if (!similarImage.empty()) {
            imshow("Similar Image " + to_string(i+1), similarImage);
            waitKey(0); // Wait for key press to proceed to the next image
        } else {
            cerr << "Error: Could not read similar image " << similarityScores[i].second << endl;
        }
    }

    return 0;
}
