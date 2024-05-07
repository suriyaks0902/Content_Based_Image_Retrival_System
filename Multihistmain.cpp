/*
  Suriya Kasiyalan Siva
  Spring 2024
  02/08/2024
  CS 5330 Computer Vision
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <dirent.h>

// Function to compute RGB histogram given the image
cv::Mat RGBHist(const cv::Mat& image, int numBins) {
    cv::Mat hist = cv::Mat::zeros(numBins * 3, numBins, CV_32F); // Create histogram for each channel

    // Loop through each pixel in the image
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            // Extract value in the three channels
            int b = pixel[0];
            int g = pixel[1];
            int r = pixel[2];
            // Increment histogram bin
            hist.at<float>(b * numBins / 256, 0) += 1.0; // Blue channel
            hist.at<float>(g * numBins / 256 + numBins, 0) += 1.0; // Green channel
            hist.at<float>(r * numBins / 256 + 2 * numBins, 0) += 1.0; // Red channel
        }
    }
    // Normalize the histogram
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

    return hist;
}

// Function to compute intersection histogram distance given histograms of two images
double histIntersectionDist(const cv::Mat& hist1, const cv::Mat& hist2) {
    double intersection = 0.0;
    for (int i = 0; i < hist1.rows; ++i) {
        for (int j = 0; j < hist1.cols; ++j) {
            intersection += (hist1.at<float>(i, j) - hist2.at<float>(i, j)) * (hist1.at<float>(i, j) - hist2.at<float>(i, j));
        }
    }
    return intersection;
}

// Function to compute top and bottom half histograms given an image and the number of bins
std::pair<cv::Mat, cv::Mat> topBottomHist(const cv::Mat& image, int numBins) {
    cv::Mat topHalf = image.rowRange(0, image.rows / 2);
    cv::Mat bottomHalf = image.rowRange(image.rows / 2, image.rows);

    cv::Mat topHist = RGBHist(topHalf, numBins);
    cv::Mat bottomHist = RGBHist(bottomHalf, numBins);

    return std::make_pair(topHist, bottomHist);
}

// Function to compute histograms for a target image and loop through an image database.
// Sorts the images in ascending order based on the histogram intersection distance.
std::vector<std::pair<std::string, double>> compHist(const std::string& targetImagePath, const std::string& directoryPath, int numBins) {
    std::vector<std::pair<std::string, double>> distances; // Store image name and distance

    cv::Mat targetImage = cv::imread(targetImagePath);
    if (targetImage.empty()) {
        std::cout << "Cannot open target image.\n";
        return distances;
    }

    // Compute top and bottom half histograms of target image
    auto [targetTopHist, targetBottomHist] = topBottomHist(targetImage, numBins);

    // Loop through directory
    DIR* dirp = opendir(directoryPath.c_str());
    if (!dirp) {
        std::cout << "Cannot open directory.\n";
        return distances;
    }

    dirent* dp;
    while ((dp = readdir(dirp)) != NULL) {
        std::string fileName = dp->d_name;
        if (fileName == "." || fileName == "..") {
            continue;
        }

        std::string imagePath = directoryPath + "/" + fileName;

        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cout << "Cannot read the image: "<< imagePath << std::endl;
            continue;
        }

        // Compute top and bottom half histograms of images in the database
        auto [imageTopHist, imageBottomHist] = topBottomHist(image, numBins);

        // Compute intersection distances for top and bottom histograms
        double topInterDistance = histIntersectionDist(targetTopHist, imageTopHist);
        double bottomInterDistance = histIntersectionDist(targetBottomHist, imageBottomHist);

        // Use weighted combination of top and bottom intersection distances 
        double distance = (0.6*topInterDistance + 0.4*bottomInterDistance);
    
        // Store the image file name and distance in the vector
        distances.push_back(std::make_pair(imagePath, distance));
    }
    closedir(dirp);

    return distances;
}

int main() {
    std::string targetImagePath = "/media/sakiran/Internal/2nd Semester/PRCV/Project/multihistogram/olympus/pic.0274.jpg";
    std::string imageDatabase = "/media/sakiran/Internal/2nd Semester/PRCV/Project/multihistogram/olympus";
    int numBins = 8;

    // Compute histograms and store the image filename and the distance
    std::vector<std::pair<std::string, double>> distances = compHist(targetImagePath, imageDatabase, numBins);

    // Sort in ascending order based on distance
    std::sort(distances.begin(), distances.end(), [](const std::pair<std::string, double>& x, const std::pair<std::string, double>& y){
        return x.second < y.second;
    });

    // Display top 4 similar images
    int N = 4;
    std::cout << "Top " << N << " matching images are: \n";
    for (int i = 0; i < std::min(N, static_cast<int>(distances.size())); ++i) {
        std::cout << "Distance: " << distances[i].second << ", Image: " << distances[i].first << std::endl;
        
        // Display the images
        std::string outputImagePath = distances[i].first;
        cv::Mat outputImage = cv::imread(outputImagePath);
        cv::imshow(distances[i].first, outputImage);
    }

    cv::waitKey(0);

    return 0;
}
