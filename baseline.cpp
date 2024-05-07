/*
  Suriya Kasiyalan Siva & Saikiran Juttu
  Spring 2024
  02/06/2024
  CS 5330 Computer Vision
*/

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp> // OpenCV library
#include <dirent.h> // Directory handling library

using namespace std;
using namespace cv;

// Function to compute the sum of squared differences between two 7x7 squares
double sumOfSquaredDifference(const Mat& square1, const Mat& square2) {
    double ssd = 0.0;
    for (int i = 0; i < square1.rows; ++i) {
        for (int j = 0; j < square1.cols; ++j) {
            for (int c = 0; c < square1.channels(); ++c) {
                // Sum of squared differences for each pixel and channel
                ssd += pow(square1.at<Vec3b>(i, j)[c] - square2.at<Vec3b>(i, j)[c], 2);
            }
        }
    }
    return ssd;
}

// Function to compute features from a given image
Mat computeFeatures(const Mat& image) {
    // Assuming the 7x7 square is at the center
    int startX = image.cols / 2 - 3;
    int startY = image.rows / 2 - 3;
    Rect roi(startX, startY, 7, 7); // Region of interest (7x7 square)

    // Extract 7x7 squares from each channel and concatenate
    Mat features;
    for (int c = 0; c < image.channels(); ++c) {
        Mat channelSquare = image(roi).clone().reshape(0, 1); // Extract square from current channel
        features.push_back(channelSquare); // Concatenate squares
    }
    return features.reshape(0, 1);  // Flatten to a single row
}

int main() {
    // Read target image
    Mat targetImage = imread("/media/sakiran/Internal/2nd Semester/PRCV/Project/custom/olympus/pic.1016.jpg");
    if (targetImage.empty()) {
        cerr << "Error: Could not read target image." << endl;
        return 1;
    }

    // Compute features of target image
    Mat targetFeatures = computeFeatures(targetImage);

    // Loop over the directory of images
    string directoryPath = "/media/sakiran/Internal/2nd Semester/PRCV/Project/custom/olympus";
    vector<pair<double, string>> similarityScores; // Stores similarity scores and image paths

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
            double distance = sumOfSquaredDifference(targetFeatures, imageFeatures);
            // Store similarity score and image path
            similarityScores.push_back(make_pair(distance, imagePath));
        }
        closedir(dir);
    } else {
        cerr << "Error: Could not open directory." << endl;
        return 1;
    }

    // Sort the list of matches based on similarity score
    sort(similarityScores.begin(), similarityScores.end());

    // Return top N matches (here, N = 5)
    int topN = 5;
    cout << "Top " << topN << " similar images:" << endl;
    for (int i = 0; i < min(topN, static_cast<int>(similarityScores.size())); ++i) {
        cout << similarityScores[i].second << " - Distance: " << similarityScores[i].first << endl;
    }

    return 0;
}
