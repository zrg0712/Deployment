#include <iostream>
#include <opencv2/opencv.hpp>

#include "utils.h"
#include "detector.h"


int main(int argc, char* argv[])
{
    const float confThreshold = 0.3f;
    const float iouThreshold = 0.4f;

    //cmdline::parser cmd;
    //cmd.add<std::string>("model_path", 'm', "Path to onnx model.", true, "yolov5.onnx");
    //cmd.add<std::string>("image", 'i', "Image source to be detected.", true, "bus.jpg");
    //cmd.add<std::string>("class_names", 'c', "Path to class names file.", true, "coco.names");
    //cmd.add("gpu", '\0', "Inference on cuda device.");

    //cmd.parse_check(argc, argv);

    //bool isGPU = cmd.exist("gpu");
    //const std::string classNamesPath = cmd.get<std::string>("class_names");
    //const std::vector<std::string> classNames = utils::loadNames(classNamesPath);
    //const std::string imagePath = cmd.get<std::string>("image");
    //const std::string modelPath = cmd.get<std::string>("model_path");
    std::cout << "Hello world" << std::endl;
    bool isGPU = true;
    const std::string classNamesPath = "F:\\learning\\OnnxDeployment\\deployment - new\\deployment\\models\\coco.names";
    const std::vector<std::string> classNames = utils::loadNames(classNamesPath);
    const std::string imagePath = "F:\\learning\\OnnxDeployment\\deployment - new\\deployment\\images\\1702604875550.jpg";
    const std::string modelPath = "F:\\learning\\OnnxDeployment\\deployment - new\\deployment\\models\\AlexNet.onnx";

    if (classNames.empty())
    {
        std::cerr << "Error: Empty class names file." << std::endl;
        return -1;
    }

    Detector detector{ nullptr };
    cv::Mat image;
    int result;

    try
    {
        detector = Detector(modelPath, isGPU, cv::Size(224, 224));
        std::cout << "Model was initialized." << std::endl;

        image = cv::imread(imagePath);
        result = detector.detect(image, confThreshold, iouThreshold);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    std::cout << "class is " << classNames[result] << std::endl;

    return 0;
}