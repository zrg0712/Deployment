#pragma once

#include <vector>
#include <ctime>
#include <string>
#include <math.h>
#include <paddle_inference_api.h>
#include <opencv2/opencv.hpp>

#include "utils.h"
#include "preprocess.h"
#include "config_parser.h"

#define OS_PATH_SEP "\\"

namespace PaddleDetection
{
	class Detector
	{
	public:
		explicit Detector(const string& model_dir, const string device)
		{
			this->m_device = device;
			this->m_model_dir = model_dir;
			load_model();
			config_.load_config(model_dir);
			preprocessor_.Init(config_.preprocess_info_);
		}

		void load_model();

		void predict(const vector<cv::Mat> imgs, const double threshold, vector<vector<PaddleDetection::ObjectResult>>& resutl, vector<double>* times = nullptr);

		void preprocess_img(const cv::Mat& ori_im);

		void postprocess_img(const std::vector<cv::Mat> mats,
			std::vector<std::vector<PaddleDetection::ObjectResult>>& result,
			std::vector<int> bbox_num, std::vector<float> output_data_,
			std::vector<int> output_mask_data_, bool is_rbox);

		std::shared_ptr<paddle_infer::Predictor> predictor;

		// Get Model Label list
		const std::vector<std::string>& getLabelList() const {
			return config_.label_list_;
		}

	private:
		string m_device;
		string m_model_dir;
		ImageBlob inputs_;
		ConfigParser config_;
		Preprocessor preprocessor_;
	};
	cv::Mat VisualizeResult(const cv::Mat& img, const vector<PaddleDetection::ObjectResult>& results, const std::string lables);
}