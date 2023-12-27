#pragma once

#include <algorithm>
#include <ctime>
#include <memory>
#include <numeric>
#include <direct.h>
#include <sstream>
#include <string>
#include <utility>

#include <mutex>
//#include <atomic>
#include <vector>
#include <opencv2/opencv.hpp>
#include <thread>

namespace PaddleDetection
{
	struct ObjectResult
	{
		// Rectangle coordinates of detected object:left, right, top, down
		std::vector<int> rect;
		int class_id;
		float confidence;
		std::vector<int> mask;
	};

	void nms(std::vector<ObjectResult>& input_boxes, float nms_threshold);

	bool pathIsExist(const std::string& path);

	void mkDir(const std::string& path);
}
