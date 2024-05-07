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
#include <dirent.h> // Include for dirent

using namespace std;
using namespace cv;

// Function to detect yellow regions (bananas) in an image
Mat detectYellowRegions(const Mat& image) {
    Mat hsvImage;
    cvtColor(image, hsvImage, COLOR_BGR2HSV);

    // Adjust these values to detect yellow color
    Scalar lowerBound = Scalar(20, 100, 100); // Lower bound for yellow in HSV
    Scalar upperBound = Scalar(30, 255, 255); // Upper bound for yellow in HSV

    Mat mask;
    inRange(hsvImage, lowerBound, upperBound, mask);

    // Apply additional morphological operations to refine the mask
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);

    return mask;
}

int main() {
    // Path to the directory containing images
    string directoryPath = "/media/sakiran/Internal/2nd Semester/PRCV/Project/task1/olympus";

    double minArea = 1000; // Adjust as needed
    double maxArea = 5000; // Adjust as needed

    vector<pair<string, double>> yellowBananaImages; // Updated variable name

    // Load the target image
    string targetImagePath = "/media/sakiran/Internal/2nd Semester/PRCV/Project/task1/olympus/pic.0343.jpg";
    Mat targetImage = imread(targetImagePath);
    if (targetImage.empty()) {
        cerr << "Error: Could not read target image." << endl;
        return 1;
    }
    Mat targetYellowMask = detectYellowRegions(targetImage); // Detect yellow regions in the target image

    // Find contours in the target image
    vector<vector<Point>> targetContours;
    findContours(targetYellowMask, targetContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Iterate through each image in the directory
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(directoryPath.c_str())) != nullptr) {
        while ((ent = readdir(dir)) != nullptr) {
            string filename = ent->d_name;
            if (filename == "." || filename == ".." || filename[0] == '.') { // Skip hidden files
                continue;
            }
            string imagePath = directoryPath + "/" + filename;
            Mat image = imread(imagePath);
            if (image.empty()) {
                cerr << "Error: Could not read image " << filename << endl;
                continue;
            }

            // Detect yellow regions in the current image
            Mat yellowMaskImage = detectYellowRegions(image);

            // Find contours in the yellow mask of the current image
            vector<vector<Point>> contours;
            findContours(yellowMaskImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            // Iterate through each contour and check if it's within the specified area range
            for (const auto& contour : contours) {
                double area = contourArea(contour);
                if (area >= minArea && area <= maxArea) {
                    // Compare the current contour with the contours in the target image
                    for (const auto& targetContour : targetContours) {
                        double similarity = matchShapes(contour, targetContour, CONTOURS_MATCH_I2, 0);
                    
                            yellowBananaImages.push_back(make_pair(imagePath, similarity));
                            break;
                        
                    }
                    break; // Move to the next image once a banana is detected
                }
            }
        }
        closedir(dir);
    } else {
        cerr << "Error: Could not open directory." << endl;
        return 1;
    }

    // Sort the vector based on similarity score
    sort(yellowBananaImages.begin(), yellowBananaImages.end(), [](const pair<string, double>& a, const pair<string, double>& b) {
        return a.second < b.second;
    });

    // Output the paths of top 5 images with bananas
    cout << "Top 5 images with bananas similar to the target image:" << endl;
    for (int i = 0; i < min(5, (int)yellowBananaImages.size()); ++i) {
        cout << yellowBananaImages[i].first << " (Similarity: " << yellowBananaImages[i].second << ")" << endl;
        // Load and display the image (you can remove this part if not needed)
        Mat outputImage = imread(yellowBananaImages[i].first);
        if (!outputImage.empty()) {
            imshow("Bananas", outputImage);
            waitKey(0);
        } else {
            cerr << "Error: Could not display image " << yellowBananaImages[i].first << endl;
        }
    }

    return 0;
}
