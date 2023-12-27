#include "config_parser.h"

namespace PaddleDetection
{
	bool ConfigParser::load_config(const string& model_dir)
	{
		YAML::Node config;
		string yml_file = model_dir + OS_PATH_SEP + "infer_cfg.yml";
		config = YAML::LoadFile(yml_file);
        if (config["mode"].IsDefined()) {
            mode_ = config["mode"].as<std::string>();
        }
        else {
            std::cerr << "Please set mode, support value : paddle/trt_fp16/trt_fp32." << std::endl;
            return false;
        }

        if (config["arch"].IsDefined()) {
            arch_ = config["arch"].as<std::string>();
        }
        else {
            std::cerr << "Please set model arch,"
                << "support value : YOLO, SSD, RetinaNet, RCNN, Face."
                << std::endl;
            return false;
        }

        // Get min_subgraph_size for tensorrt
        if (config["min_subgraph_size"].IsDefined()) {
            min_subgraph_size_ = config["min_subgraph_size"].as<int>();
        }
        else {
            std::cerr << "Please set min_subgraph_size." << std::endl;
            return false;
        }
        // Get draw_threshold for visualization
        if (config["draw_threshold"].IsDefined()) {
            draw_threshold_ = config["draw_threshold"].as<float>();
        }
        else {
            std::cerr << "Please set draw_threshold." << std::endl;
            return false;
        }
        // Get Preprocess for preprocessing
        if (config["Preprocess"].IsDefined()) {
            preprocess_info_ = config["Preprocess"];
        }
        else {
            std::cerr << "Please set Preprocess." << std::endl;
            return false;
        }
        // Get label_list for visualization
        if (config["label_list"].IsDefined()) {
            label_list_ = config["label_list"].as<std::vector<std::string>>();
        }
        else {
            std::cerr << "Please set label_list." << std::endl;
            return false;
        }

        // Get use_dynamic_shape for TensorRT
        if (config["use_dynamic_shape"].IsDefined()) {
            use_dynamic_shape_ = config["use_dynamic_shape"].as<bool>();
        }
        else {
            std::cerr << "Please set use_dynamic_shape." << std::endl;
            return false;
        }

        // Get conf_thresh for tracker
        if (config["tracker"].IsDefined()) {
            if (config["tracker"]["conf_thres"].IsDefined()) {
                conf_thresh_ = config["tracker"]["conf_thres"].as<float>();
            }
            else {
                std::cerr << "Please set conf_thres in tracker." << std::endl;
                return false;
            }
        }

        // Get NMS for postprocess
        if (config["NMS"].IsDefined()) {
            nms_info_ = config["NMS"];
        }
        // Get fpn_stride in PicoDet
        if (config["fpn_stride"].IsDefined()) {
            fpn_stride_.clear();
            for (auto item : config["fpn_stride"]) {
                fpn_stride_.emplace_back(item.as<int>());
            }
        }

        if (config["mask"].IsDefined()) {
            mask_ = config["mask"].as<bool>();
        }

        return true;
	}
}