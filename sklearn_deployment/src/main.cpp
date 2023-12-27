#include <iostream>
#include <vector>
#include "utils.h"
#include "detector.h"

using namespace std;

int main()
{
	//float v[] = { 2.5f,2.1f,3.6f,4.0f };
	std::vector<float> v{ 5.1f, 3.5f, 1.4f, 0.2f };
	bool isGPU = false; // sklearn��GPU����ʾwarning
	std::string modelPath = "./model_file/gbdt_iris.onnx"; // ��ǰ·����.vcxproj���ڵ�·��
	Detector detector(modelPath, isGPU);
	int res = detector.detect(v);

	std::cout << "result: " << res << std::endl;
	return 0;
}
