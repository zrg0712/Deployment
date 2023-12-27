#include "utils.h"

namespace PaddleDetection
{
	void nms(std::vector<ObjectResult>& input_boxes, float nms_threshold)
	{
		sort(input_boxes.begin(), input_boxes.end(), [](ObjectResult a, ObjectResult b) {return a.confidence > b.confidence; });
		std::vector<float> vArea(input_boxes.size());
		
		for (int i = 0; i<int(input_boxes.size()); ++i)
		{
			vArea[i] = (input_boxes.at(i).rect[2] - input_boxes.at(i).rect[0] + 1) *
				(input_boxes.at(i).rect[3] - input_boxes.at(i).rect[1] + 1);
		}

		for (int i = 0; i<int(input_boxes.size()); ++i)
		{
			for (int j = i + 1; j<int(input_boxes.size());)
			{
				float xx1 = (std::max)(input_boxes[i].rect[0], input_boxes[j].rect[0]);
				float yy1 = (std::max)(input_boxes[i].rect[1], input_boxes[j].rect[1]);
				float xx2 = (std::max)(input_boxes[i].rect[2], input_boxes[j].rect[2]);
				float yy2 = (std::max)(input_boxes[i].rect[3], input_boxes[j].rect[3]);
				float w = (std::max)(float(0), xx2 - xx1 + 1);
				float h = (std::max)(float(0), yy2 - yy1 + 1);
				float inter = w * h;
				float ovr = inter / (vArea[i] + vArea[j] - inter);
				if (ovr >= nms_threshold)
				{
					input_boxes.erase(input_boxes.begin() + j);
					vArea.erase(vArea.begin() + j);
				}
				else
				{
					j++;
				}
			}
		}

	}

	bool pathIsExist(const std::string& path)
	{
		struct _stat buffer;
		return (_stat(path.c_str(), &buffer) == 0);
	}

	void mkDir(const std::string& path) {
		if (pathIsExist(path)) return;
		int ret = 0;
		ret = _mkdir(path.c_str());
		if (ret != 0) {
			std::string path_error(path);
			path_error += "mkdir failed";
			throw std::runtime_error(path_error);
		}
	}
}