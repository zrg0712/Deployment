#include "detector.h"

Detector::Detector(const std::string& modelPath, const bool& isGPU = true,
    const cv::Size& inputSize = cv::Size(640, 640))
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

#ifdef _WIN32
    std::wstring w_modelPath = utils::charToWstring(modelPath.c_str());
    session = Ort::Session(env, w_modelPath.c_str(), session_options);
#else
    session = Ort::Session(env, modelPath.c_str(), session_options);
#endif // _WIN32

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    this->isDynamicInputShape = false;
    if (inputTensorShape[2] == -1 && inputTensorShape[3] == -1)
    {
        std::cout << "Dynamic input shape" << std::endl;
        this->isDynamicInputShape = true;
    }

    for (auto shape : inputTensorShape)
        std::cout << "Input shape: " << shape << std::endl;

    inputNames.push_back(session.GetInputNameAllocated(0, allocator).get());
    outputNames.push_back(session.GetOutputNameAllocated(0, allocator).get());

    //auto temp = inputNames[0];
    //auto temp2 = inputNames.data();
    std::cout << "Input name: " << inputNames[0] << std::endl;
    std::cout << "Output name: " << outputNames[0] << std::endl;

    this->inputImageShape = cv::Size2f(inputSize);
}

void Detector::getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
    float& bestConf, int& bestClassId)
{
    // first 5 element are box and obj confidence
    bestClassId = 5;
    bestConf = 0;

    for (int i = 5; i < numClasses + 5; i++)
    {
        if (it[i] > bestConf)
        {
            bestConf = it[i];
            bestClassId = i - 5;
        }
    }

}

void Detector::preprocessing(cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape)
{
    cv::Mat resizedImage, floatImage;
    cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);
    //utils::letterbox(resizedImage, resizedImage, this->inputImageShape,
    //    cv::Scalar(114, 114, 114), this->isDynamicInputShape,
    //    false, true, 32);
    cv::resize(resizedImage, resizedImage, this->inputImageShape);
    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;

    resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
    blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
    cv::Size floatImageSize{ floatImage.cols, floatImage.rows };

    // hwc->chw
    std::vector<cv::Mat> chw(floatImage.channels());
    for (int i = 0; i < floatImage.channels(); ++i)
    {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(floatImage, chw);
}

int Detector::postprocessing(std::vector<Ort::Value>& outputTensors)
{
    auto* rawOutput = outputTensors[0].GetTensorData<float>();  // 数据所在指针
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();  // detection result shape [1,1,5]
    size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();  // total number
    std::vector<float> output(rawOutput, rawOutput + count);  // 初始化一个大小为5的vector

	int max_index = 0;
    float max_num = float(*rawOutput);
	for(int i=1;i<output.size();i++)
	{
		rawOutput++;
		if(float(*rawOutput)>max_num)
		{
			max_index = i;
		}
	}

    return max_index;
}

int Detector::detect(cv::Mat& image, const float& confThreshold = 0.4,
    const float& iouThreshold = 0.45)
{
    float* blob = nullptr;
    std::vector<int64_t> inputTensorShape{ 1,3,-1,-1 };
    this->preprocessing(image, blob, inputTensorShape);

    size_t inputTensorSize = utils::vectorProduct(inputTensorShape);
    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

    std::vector<Ort::Value> inputTensors;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize,
        inputTensorShape.data(), inputTensorShape.size()
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

    int result = this->postprocessing(outputTensors);

    delete[] blob;

    return result;
}
