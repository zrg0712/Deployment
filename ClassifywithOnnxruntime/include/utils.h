#pragma once

#include <opencv2/opencv.hpp>
#include <codecvt>
#include <fstream>

struct Result
{
	cv::Rect box;
	float conf;
	int classId;
};

namespace utils
{
	size_t vectorProduct(const std::vector<int64_t>& vector);
	std::wstring charToWstring(const char* str);
	std::vector<std::string> loadNames(const std::string& path);
	void visualizeDetection(cv::Mat& image, std::vector<Result>& detections, const std::vector<std::string>& classNames);

	void letterbox(const cv::Mat& image, cv::Mat& outImage,
		const cv::Size& newShape,
		const cv::Scalar& color,
		bool auto_,
		bool scaleFill,
		bool scaleUp,
		int stride);

	void scaleCoords(const cv::Size& imageShape, cv::Rect& box, const cv::Size& imageOriginalShape);

	template <typename T>
	T clip(const T& n, const T& lower, const T& upper);

}