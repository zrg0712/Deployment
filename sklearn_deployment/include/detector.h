#pragma once
#include <onnxruntime_cxx_api.h>
#include <string>
#include <iostream>
#include <algorithm>


class Detector
{
public:
	Detector(const std::string& modelPath, const bool& isGPU);

	int detect(std::vector<float> inputs);

private:
	Ort::Env env{ nullptr };
	Ort::SessionOptions session_options{ nullptr };
	Ort::Session session{ nullptr };

	std::vector<std::string> inputNames;
	std::vector<std::string> outputNames;
};