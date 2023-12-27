#pragma once

#include <onnxruntime_cxx_api.h>
#include <utility>

#include "utils.h"

class Detector
{
public:
	explicit Detector(std::nullptr_t) {};
	Detector(const std::string& modelPath,
		const bool& isGPU,
		const cv::Size& inputSize);

	int detect(cv::Mat& image, const float& confThreshold, const float& iouThreshold);

private:
	Ort::Env env{ nullptr };
	Ort::SessionOptions session_options{ nullptr };
	Ort::Session session{ nullptr };

	void preprocessing(cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape);
	int postprocessing(std::vector<Ort::Value>& outputTensors);

	static void getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
		float& bestConf, int& bestClassId);

	std::vector<std::string> inputNames;
	std::vector<std::string> outputNames;
	bool isDynamicInputShape{};
	cv::Size2f inputImageShape;
};
