#include "object_detector.h"

namespace PaddleDetection
{
	void Detector::load_model()
	{
		paddle_infer::Config config;
		string prog_file = m_model_dir + OS_PATH_SEP + "model.pdmodel";
		string params_file = m_model_dir + OS_PATH_SEP + "model.pdiparams";
		config.SetModel(prog_file, params_file);
		if (m_device == "GPU")
		{
			config.EnableUseGpu(3000, 0);
		}
		else
		{
			config.DisableGpu();
			config.EnableMKLDNN();
		}
		config.EnableMemoryOptim();
		config.SetCpuMathLibraryNumThreads(1);
		config.DisableGlogInfo();
		predictor = paddle_infer::CreatePredictor(config);
	}

	void Detector::predict(const vector<cv::Mat> imgs, const double threshold, vector<vector<PaddleDetection::ObjectResult>>& result, vector<double>* times)
	{
		auto preprocess_start = chrono::steady_clock::now();
		int batch_size = imgs.size();

		vector<float> in_data_all;
		vector<float> im_shape_all(static_cast<double>(batch_size) * 2.0);
		vector<float> scale_factor_all(static_cast<double>(batch_size) * 2.0f);
		vector<const float*> output_data_list_;
		vector<int> out_bbox_num_data_;
		vector<int> out_mask_data_;

		//in_net img for each batch
		vector<cv::Mat> in_net_img_all(batch_size);  // 存放batch_size大小的图像

		// preprocess image
		for (int bs_idx = 0; bs_idx < batch_size; bs_idx++)
		{
			cv::Mat im = imgs.at(bs_idx);
			preprocess_img(im);  // 获取了inputs_信息
			im_shape_all[bs_idx * 2] = inputs_.im_shape_[0];
			im_shape_all[bs_idx * 2 + 1] = inputs_.im_shape_[1];

			scale_factor_all[bs_idx * 2] = inputs_.scale_factor_[0];
			scale_factor_all[bs_idx * 2 + 1] = inputs_.scale_factor_[1];
			in_data_all.insert(in_data_all.end(), inputs_.im_data_.begin(),
				inputs_.im_data_.end());

			// collect in_net img
			in_net_img_all[bs_idx] = inputs_.in_net_im_;
		}
		auto preprocess_end = std::chrono::steady_clock::now();  // 结束时间
		// prepare input tensor
		auto input_names = predictor->GetInputNames();
		for (const auto& tensor_name : input_names)
		{
			auto in_tensor = predictor->GetInputHandle(tensor_name);
			if (tensor_name == "image" || tensor_name == "x0")
			{
				int rh = inputs_.in_net_shape_[0];
				int rw = inputs_.in_net_shape_[1];
				in_tensor->Reshape({ batch_size,3,rh,rw });
				in_tensor->CopyFromCpu(in_data_all.data());
			}
			else if (tensor_name == "im_shape")
			{
				in_tensor->Reshape({ batch_size,2 });
				in_tensor->CopyFromCpu(im_shape_all.data());
			}
			else if (tensor_name == "scale_factor") {
				in_tensor->Reshape({ batch_size, 2 });
				in_tensor->CopyFromCpu(scale_factor_all.data());
			}
		}
		// Run predictor
		vector<vector<float>> out_tensor_list;
		vector<vector<int>> output_shape_list;

		bool is_rbox = false;
		int reg_max = 7;
		int num_class = 1;

		auto inference_start = chrono::steady_clock::now();
		predictor->Run();
		// Get output tensor
		auto output_names = predictor->GetOutputNames();
		for (int j = 0; j < output_names.size(); j++)
		{
			auto output_tensor = predictor->GetOutputHandle(output_names[j]);
			vector<int> output_shape = output_tensor->shape();
			int out_num = std::accumulate(output_shape.begin(), output_shape.end(),
				1, std::multiplies<int>());
			output_shape_list.push_back(output_shape);

			std::vector<float> out_data;
			//std::vector<int> out_data;
			out_bbox_num_data_.resize(out_num);
			out_data.resize(out_num);
			output_tensor->CopyToCpu(out_data.data());
			out_tensor_list.push_back(out_data);
		}
		auto inference_end = std::chrono::steady_clock::now();
		// Postprocessing result
		auto postprocess_start = std::chrono::steady_clock::now();
		result.clear();
		is_rbox = output_shape_list[0][output_shape_list[0].size() - 1] % 10 == 0;
		for (int bs_idx = 0; bs_idx < batch_size; bs_idx++)
			postprocess_img(imgs, result, out_bbox_num_data_, out_tensor_list[0], out_mask_data_, is_rbox);

		auto postprocess_end = std::chrono::steady_clock::now();

		std::chrono::duration<float> preprocess_diff =
			preprocess_end - preprocess_start;
		times->push_back(static_cast<double>(preprocess_diff.count()) * 1000);
		std::chrono::duration<float> inference_diff = inference_end - inference_start;
		times->push_back(
			static_cast<double>(inference_diff.count()) * 1000);
		std::chrono::duration<float> postprocess_diff =
			postprocess_end - postprocess_start;
		times->push_back(static_cast<double>(postprocess_diff.count()) * 1000);

	}

	void Detector::preprocess_img(const cv::Mat& ori_im)
	{
		// Clone the image : keep the original mat for postprocess
		cv::Mat im = ori_im.clone();
		cv::cvtColor(im, im, cv::COLOR_BGR2RGB);  // 改变图像通道
		preprocessor_.Run(&im, &inputs_);
	}

	void Detector::postprocess_img(const std::vector<cv::Mat> mats,
		std::vector<std::vector<PaddleDetection::ObjectResult>>& result,
		std::vector<int> bbox_num, std::vector<float> output_data_,
		std::vector<int> output_mask_data_, bool is_rbox)
	{
		//result.clear();
		int start_idx = 0;
		//int total_num = std::accumulate(bbox_num.begin(), bbox_num.end(), 0);
		int total_num = bbox_num.size() / 6;
		int out_mask_dim = -1;
		if (config_.mask_) {
			out_mask_dim = output_mask_data_.size() / total_num;
		}

		for (int im_id = 0; im_id < mats.size(); im_id++)
		{
			for (int j = start_idx; j < total_num; j++)
			{
				// Class id
				int class_id = static_cast<int>(round(output_data_[5 + static_cast<long long>(j) * 6]));
				// Confidence score
				float score = output_data_[4 + static_cast<long long>(j) * 6];
				float cx = output_data_[0 + static_cast<long long>(j) * 6];
				float cy = output_data_[1 + static_cast<long long>(j) * 6];
				float width = output_data_[2 + static_cast<long long>(j) * 6];
				float height = output_data_[3 + static_cast<long long>(j) * 6];

				int xmin = cx - (width / 2);
				int ymin = cy - (height / 2);
				int xmax = cx + (width / 2);
				int ymax = cy + (height / 2);


				PaddleDetection::ObjectResult result_item;
				result_item.rect = { xmin, ymin, xmax, ymax };
				result_item.class_id = class_id;
				result_item.confidence = score;

				if (result_item.confidence >= 0)
				{
					result.at(im_id).push_back(result_item);
				}

			}
			PaddleDetection::nms(result.at(im_id), 0.4);
		}
	}

	cv::Mat VisualizeResult(const cv::Mat& img, const vector<PaddleDetection::ObjectResult>& results, const std::string lables)
	{
		cv::Mat vis_img = img.clone();
		int img_h = vis_img.rows;
		int img_w = vis_img.cols;

		int c1 = 0;
		int c2 = 0;
		int c3 = 255;
		cv::Scalar roi_color = cv::Scalar(c1, c2, c3);
		//int font_face = cv::FONT_HERSHEY_COMPLEX_SMALL;
		int font_face = cv::FONT_HERSHEY_DUPLEX;
		double font_scale = 2.0;
		float thickness = 2.0f;
		for (int i = 0; i < results.size(); ++i)
		{
			cv::Point origin;
			// Draw rectangle
			int w = results[i].rect[2] - results[i].rect[0];
			int h = results[i].rect[3] - results[i].rect[1];
			cv::Rect roi = cv::Rect(results[i].rect[0], results[i].rect[1], w, h);
			// Draw roi object
			cv::rectangle(vis_img, roi, roi_color, 2);

			//// Draw point
			//int x = (results[i].rect[0] + results[i].rect[2]) / 2;
			//int y = (results[i].rect[3] + results[i].rect[1]) / 2;
			//cv::Point pt = cv::Point(x, y);
			//cv::circle(vis_img, pt, 2, roi_color, thickness);
		}
		return vis_img;
	}
}