#include "detector.h"
#include "utils.h"

Detector::Detector(const std::string& modelPath, const bool& isGPU)
{
	env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
	session_options = Ort::SessionOptions();
	std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
	auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
	OrtCUDAProviderOptions cudaOption;

	if (isGPU && (cudaAvailable == availableProviders.end()))
	{
		std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
		std::cout << "Inference device: CPU" << std::endl;
	}
	else if (isGPU && (cudaAvailable != availableProviders.end()))
	{
		std::cout << "Inference device: GPU" << std::endl;
		session_options.AppendExecutionProvider_CUDA(cudaOption);
	}
	else
	{
		std::cout << "Inference device: CPU" << std::endl;
	}

	std::wstring w_modelPath = utils::charToWstring(modelPath.c_str());
	session = Ort::Session(env, w_modelPath.c_str(), session_options);

	Ort::AllocatorWithDefaultOptions allocator;
	Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
	std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();

	std::cout << std::endl;
	inputNames.push_back(session.GetInputNameAllocated(0, allocator).get());
	outputNames.push_back(session.GetOutputNameAllocated(0, allocator).get());

	//std::cout << "Input name: " << inputNames[0] << std::endl;
	//std::cout << "Output name: " << outputNames[0] << std::endl;
}


int Detector::detect(std::vector<float> inputs)
{
	float* blob = inputs.data();
	std::vector<int64_t> inputTensorShape{ 1, 4 };

	size_t inputTensorSize = utils::vectorProduct(inputTensorShape);
	std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

	std::vector<Ort::Value> inputTensors;
	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
		OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault
	);
	inputTensors.push_back(Ort::Value::CreateTensor<float>(
		memoryInfo, inputs.data(), inputTensorSize,  // .data()返回一个指向向量内部使用的数组中第一个元素的指针
		inputTensorShape.data(), 2
		));

	std::vector<const char*> input_names_char(inputNames.size(), nullptr);
	std::transform(std::begin(inputNames), std::end(inputNames), std::begin(input_names_char),
		[&](const std::string& str) { return str.c_str(); });
	std::vector<const char*> output_names_char(outputNames.size(), nullptr);
	std::transform(std::begin(outputNames), std::end(outputNames), std::begin(output_names_char),
		[&](const std::string& str) { return str.c_str(); });

	std::vector<Ort::Value> outputTensors = this->session.Run(Ort::RunOptions{ nullptr },
		input_names_char.data(),
		inputTensors.data(),
		1,
		output_names_char.data(),
		1);

	auto* rawOutput = outputTensors[0].GetTensorData<int>();  // 数据所在指针 int数据类型跟要输出的结果有关，类别是整数故是int
	int final_res = int(*rawOutput);

	return final_res;

}