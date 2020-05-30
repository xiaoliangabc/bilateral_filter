#include <stdio.h>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

class Timer {
 public:
  // Start Timer
  void start() { start_time_ = std::chrono::steady_clock::now(); }

  // Get elapsed time
  double getElapsedTime() {
    auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start_time_);
    return double(elapsed_time.count()) *
           std::chrono::milliseconds::period::num /
           std::chrono::milliseconds::period::den;
  }

 private:
  std::chrono::_V2::steady_clock::time_point start_time_;
};

class BilateralFilter {
 public:
  void filter(const cv::Mat src, cv::Mat& dst, int diameter, float sigma_color,
              float sigma_space) {
    // Create dst image
    dst.create(src.size(), src.type());

    // Check sigma valid
    if (sigma_color <= 0.0) sigma_color = 1.0;
    if (sigma_space <= 0.0) sigma_space = 1.0;

    // Check diameter valid and compute radius
    int radius = diameter / 2;
    if (radius <= 0) radius = cvRound(sigma_space * 1.5);
    radius = std::max(radius, 1);
    diameter = radius * 2 + 1;

    // Make border for input image
    cv::Mat border_image;
    copyMakeBorder(src, border_image, radius, radius, radius, radius,
                   cv::BORDER_DEFAULT);

    // Initialize color-related bilateral filter coefficients
    std::vector<float> color_weights(256);
    ComputeColorGaussianLUT(sigma_color, &color_weights[0]);

    // Compute look up table for sapce gaussian
    std::vector<float> space_weights(diameter * diameter);
    std::vector<int> space_offset(diameter * diameter);
    int maxk = ComputeSpaceGaussianLUT(radius, sigma_space, border_image,
                                       &space_weights[0], &space_offset[0]);

    // Filter invoker
    cv::Size size = src.size();
    FilterInvoker(border_image, color_weights, space_weights, space_offset,
                  size, radius, maxk, &dst);
  }

 private:
  // Compute look up table for color gaussian
  void ComputeColorGaussianLUT(const float& sigma_color, float* color_weights) {
    // Fixed guassian coefficient
    float gauss_color_coeff = -0.5 / (sigma_color * sigma_color);
    // Compute look up table for every difference
    for (int i = 0; i < 256; ++i)
      color_weights[i] =
          static_cast<float>(std::exp(i * i * gauss_color_coeff));
  }

  // Compute look up table for sapce gaussian
  int ComputeSpaceGaussianLUT(const int& radius, const float& sigma_space,
                              const cv::Mat& border_image, float* space_weights,
                              int* space_offset) {
    // Fixed guassian coefficient
    float gauss_space_coeff = -0.5 / (sigma_space * sigma_space);
    // Maximum number of valid elements
    int maxk = 0;
    // Compute look up table for every difference
    for (int i = -radius; i <= radius; ++i) {
      for (int j = -radius; j <= radius; ++j) {
        // Compute distance to center
        float range = std::sqrt((float)i * i + (float)j * j);
        // Skip invalid element
        if (range > radius) continue;
        // Compute gaussian weights
        space_weights[maxk] =
            (float)std::exp(range * range * gauss_space_coeff);
        // Compute element offset
        space_offset[maxk++] = (int)(i * border_image.step + j);
      }
    }
    return maxk;
  }

  // Filter invoker
  void FilterInvoker(const cv::Mat& border_image,
                     const std::vector<float> color_weights,
                     const std::vector<float> space_weights,
                     const std::vector<int> space_offset, const cv::Size& size,
                     const int& radius, const int& maxk, cv::Mat* dst) {
    // Compute diameter
    int diameter = 2 * radius + 1;
    // Filter using opencv parallel framework
    cv::parallel_for_(cv::Range(0, size.height), [&](const cv::Range& range) {
      // Iterate over each row
      for (int i = range.start; i < range.end; i++) {
        // Point to the first column of this row
        const uchar* sptr = border_image.ptr(i + radius) + radius;  // Border
        uchar* dptr = dst->ptr(i);
        // Iterate over each column
        for (int j = 0; j < size.width; ++j) {
          // Get center point intensity
          int center_val = sptr[j];
          // Intensity sum and weight sum
          float sum = 0.0, weight_sum = 0.0;
          // Iterate over each point in the diameter range
          for (int k = 0; k < maxk; ++k) {
            // Get neighbor point intensity
            int neighbor_val = sptr[j + space_offset[k]];
            // Compute weight using both space and color
            float weight = space_weights[k] *
                           color_weights[abs(neighbor_val - center_val)];
            // Add intensity
            sum += weight * neighbor_val;
            // Add weights
            weight_sum += weight;
          }
          // Compute mean intensity
          dptr[j] = sum / weight_sum;
        }
      }
    });
  }
};

void PrintUsage() {
  printf("---------------------------------------------------------------\n");
  printf(
      "Usage: bilateral_filter image_path diameter sigma_color "
      "sigma_space\n");
  printf("bilateral_filter: bin file for current project\n");
  printf("image_path: raw image path for processing\n");
  printf("diameter: diameter of each pixel neighborhood\n");
  printf("sigma_color: filter sigma in the color space\n");
  printf("sigma_space: filter sigma in the coordinate space\n");
  printf("---------------------------------------------------------------\n");
}

int main(int argc, char** argv) {
  // Check out input
  if (argc != 5) {
    std::cerr << "Error: need input params.\n";
    PrintUsage();
    return EXIT_FAILURE;
  }

  // Parse input parameters
  const std::string raw_image_path = argv[1];
  int diameter = atoi(argv[2]);
  double sigma_color = atof(argv[3]);
  double sigma_space = atof(argv[4]);

  // Read raw image as gray sacle
  cv::Mat raw_image = cv::imread(raw_image_path, cv::IMREAD_GRAYSCALE);

  // Bilateral filter
  cv::Mat filter_image;
  Timer timer;
  timer.start();
  BilateralFilter bilateral_filter;
  bilateral_filter.filter(raw_image, filter_image, diameter, sigma_color,
                          sigma_space);
  printf("Bilateral filter took %f milliseconds\n", timer.getElapsedTime());

  // Save filtered image
  std::string filter_image_path =
      raw_image_path.substr(0, raw_image_path.length() - 4);
  filter_image_path += "_filtered.jpg";
  cv::imwrite(filter_image_path, filter_image);

  return 0;
}