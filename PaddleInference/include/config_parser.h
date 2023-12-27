#pragma once

#include <string>
#include <iostream>
#include <vector>
#include <yaml-cpp/yaml.h>

#define OS_PATH_SEP "\\"

using namespace std;

namespace PaddleDetection
{
	class ConfigParser
	{
	public:
		bool load_config(const string& model_dir);
		string mode_;
		float draw_threshold_;
		string arch_;
		int min_subgraph_size_;
		YAML::Node preprocess_info_;
		YAML::Node nms_info_;
		vector<string> label_list_;
		vector<int> fpn_stride_;
		bool use_dynamic_shape_;
		float conf_thresh_;
		bool mask_ = false;
	};
}
