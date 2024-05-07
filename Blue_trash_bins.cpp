/*
  Suriya Kasiyalan Siva
  Spring 2024
  02/11/2024
  CS 5330 Computer Vision
*/

// Including necessary libraries
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <dirent.h> // Include for dirent

using namespace std;
using namespace cv;

// Function to detect blue regions in an image
Mat detectBlueRegions(const Mat& image) {
    // Convert image to HSV color space
    Mat hsvImage;
    cvtColor(image, hsvImage, COLOR_BGR2HSV);

    // Define lower and upper bounds for blue color in HSV
    Scalar lowerBound = Scalar(100, 50, 50); // Adjust as needed
    Scalar upperBound = Scalar(140, 255, 255); // Adjust as needed

    // Create mask to identify blue regions
    Mat mask;
    inRange(hsvImage, lowerBound, upperBound, mask);

    return mask;
}

int main() {
    // Load the target image
    string targetImagePath = "/media/sakiran/Internal/2nd Semester/PRCV/Project/shape/olympus/pic.0287.jpg";

    // Define minimum and maximum area for detected blue regions
    double minArea = 10000; // Adjust as needed
    double maxArea = 50000; // Adjust as needed

    // Vector to store paths of images containing blue trash cans
    vector<string> blueBinImages;

    // Read the target image
    Mat targetImage = imread(targetImagePath);
    if (targetImage.empty()) {
        cerr << "Error: Could not read target image." << endl;
        return 1;
    }

    // Detect blue regions in the target image
    Mat targetBlueMask = detectBlueRegions(targetImage);
    vector<vector<Point>> targetContours;
    findContours(targetBlueMask, targetContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

    // Directory containing other images to compare with target image
    string directoryPath = "/media/sakiran/Internal/2nd Semester/PRCV/Project/shape/olympus";

    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(directoryPath.c_str())) != nullptr) {
        while ((ent = readdir(dir)) != nullptr) {
            string filename = ent->d_name;
            if (filename == "." || filename == "..") {
                continue;
            }
            string imagePath = directoryPath + "/" + filename;
            Mat image = imread(imagePath);
            if (image.empty()) {
                cerr << "Error: Could not read image " << filename << endl;
                continue;
            }
            Mat blueMaskImage = detectBlueRegions(image);
            vector<vector<Point>> contours;
            findContours(blueMaskImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

            // Compare blue regions in current image with target image
            for (const auto& contour : contours) {
                double area = contourArea(contour);
                if (area >= minArea && area <= maxArea) {
                    // Compare contours of the current image with the target image
                    for (const auto& targetContour : targetContours) {
                        double similarity = matchShapes(contour, targetContour, CONTOURS_MATCH_I1, 0);
                        
                            blueBinImages.push_back(imagePath);
                            break;
                        
                    }
                    break;
                }
            }
        }
        closedir(dir);
    } else {
        cerr << "Error: Could not open directory." << endl;
        return 1;
    }

    // Output paths of images containing blue trash cans similar to the target image
    cout << "Images with blue trash can bins similar to the target image:" << endl;
    for (const auto& imagePath : blueBinImages) {
        cout << imagePath << endl;
        Mat outputImage = imread(imagePath);
        imshow("Blue Trash Can Bins", outputImage);
        waitKey(0);
    }

    return 0;
}
