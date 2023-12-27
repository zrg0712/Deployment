#include <iostream>
#include <string>
#include <Windows.h>
#include <io.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <numeric>
#include <fstream>
#include "object_detector.h"

#include <paddle_inference_api.h>

using namespace std;


int main(int argc, char** argv) {
	string output_dir = "result";
	if (!PaddleDetection::pathIsExist(output_dir)) {
		PaddleDetection::mkDir(output_dir);
	}
	std::cout << "use " << "GPU" << " to infer" << std::endl;
	string model_dir = "F:\\GitRepository\\Deployment\\PaddleInference\\PaddleInference\\PaddleInference\\model\\disc_classification";
	string img_dir = "F:\\GitRepository\\Deployment\\PaddleInference\\PaddleInference\\PaddleInference\\images";

	std::vector<std::string> img_path;
	std::vector<cv::String> cv_all_img_paths;
	cv::glob(img_dir, cv_all_img_paths);
	if (cv_all_img_paths.size() == 1)
	{
		for (const auto& image_path : cv_all_img_paths)
		{
			img_path.push_back(image_path);
		}
	}

	PaddleDetection::Detector detector(model_dir, "GPU");
	vector<cv::Mat> batch_imgs;
	for (const auto& path : img_path)
	{
		cv::Mat im = cv::imread(path, 1);
		batch_imgs.push_back(im);
	}
	vector<vector<PaddleDetection::ObjectResult>> result(img_path.size());
	vector<double> det_times;

	// threshold 0.5
	detector.predict(batch_imgs, 0.5, result, &det_times);
	auto labels = detector.getLabelList();
	vector<PaddleDetection::ObjectResult> im_result;
	string save_image_path(output_dir);

	for (int i = 0; i < result.size(); i++)
	{
		im_result.clear();
		for (int j = 0; j < result.at(i).size(); j++)
		{
			PaddleDetection::ObjectResult item = result.at(i).at(j);
			if (item.class_id == -1)
			{
				continue;
			}
			item.rect[0] = int(float(item.rect[0]) * 3072.0f / 640.0f);
			item.rect[1] = int(float(item.rect[1]) * 3072.0f / 640.0f);
			item.rect[2] = int(float(item.rect[2]) * 3072.0f / 640.0f);
			item.rect[3] = int(float(item.rect[3]) * 3072.0f / 640.0f);
			im_result.push_back(item);
		}

		cv::Mat vis_img = PaddleDetection::VisualizeResult(batch_imgs.at(i), im_result, labels.at(i));
		save_image_path = save_image_path + "\\" + to_string(i)+".jpg";
		cv::imwrite(save_image_path, vis_img);
		std::cout << "Visualized output saved as " << save_image_path.c_str() << std::endl;
	}


}