#ifndef CLASS_DETECTOR_H_
#define CLASS_DETECTOR_H_

#include "API.h"
#include <iostream>
#include <opencv2/opencv.hpp>

struct Result
{
	int		 id		= -1;
	float	 prob	= 0.f;
	cv::Rect rect;
};

using BatchResult = std::vector<Result>;

enum ModelType
{
	YOLOV3,
	YOLOV3_TINY,
	YOLOV4,
	YOLOV4_TINY,
};

enum Precision
{
	INT8 = 0,
	FP16,
	FP32
};

struct Config
{
	std::string file_model_cfg					= "configs/yolov3.cfg";

	std::string file_model_weights				= "configs/yolov3.weights";

	float detect_thresh							= 0.9;

	ModelType	net_type						= YOLOV3;

	Precision	inference_precison				= FP32;
	
	int	gpu_id									= 0;
};

class API Detector
{
public:
	explicit Detector();

	~Detector();

	void init(const Config &config);

	void detect(const std::vector<cv::Mat> &mat_image, std::vector<BatchResult> &vec_batch_result);

private:
	
	Detector(const Detector &);
	const Detector &operator =(const Detector &);
	class Impl;
	Impl *_impl;
};

#endif // !CLASS_QH_DETECTOR_H_
