/*
* Project:Face Recognition, Human Recognition,
* Human Fall Detection, Anomly Detection, Human Following


* Programmer:Fikrat Gasimov

* Work: Embedded AI & Robotics Scientist
* 
  Date: 31/05/2022
*
*
*/






// Class Detector Utilis
#include "class_timer.hpp"
#include "class_detector.h"
#include "trt_utils.h"
// File Management
#include <fstream>
#include <sstream>
#include <iostream>
// Time Management
#include <chrono>

#include "client_socket.hpp"
// Json Management
#include "json.hpp"
// Memory Managment
#include <memory>
// Thread Managment
#include <thread>
// Map Dictionary Managment
#include <map>
// String
#include <string>
// Jetson Lib
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <l2norm_helper.h>

// OpenCV
#include <opencv2/highgui.hpp>
// Face Net
#include "faceNet.h"
#include "network.h"
// MTCNN
#include "mtcnn.h"

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

using namespace cv;
using namespace dnn;
using namespace std;

using namespace nvinfer1;
using namespace nvuffparser;


//Main
int main() {
        rs2::colorizer color_map;
	rs2::pipeline p;
        p.start();

        // Load configuration from json file
        std::ifstream config_file("../configs/config.json", std::ifstream::binary);
        json config = json::parse(config_file);

        //USER DEFINED VALUES
        const string uffFile="../Models/facenet/facenet.uff";
        const string engineFile="../Models/facenet/facenet.engine";
	nvinfer1::DataType dtype = nvinfer1::DataType::kHALF;
        
	//DataType dtype = DataType::kFLOAT;
        bool serializeEngine = true;
        int batchSize = 1;
       
       	int videoFrameWidth = config["width"];
        int videoFrameHeight = config["height"];
        int maxFacesPerScene = config["maxFacesPerScene"];
        float knownPersonThreshold = config["knownPersonThreshold"];
        const string pipe_width =  std::to_string(videoFrameWidth);
        const string pipe_height = std::to_string(videoFrameHeight);
	
	// Local Port and IP address for ALARM
	const string alarm_hostname=config["default_server_hostname"];
	const int alarm_port = config["default_server_port"];

	// Remote PC IP and Port for Video Streaming
        const string default_pipe_host = config["default_pipe_host"]; 
        const string default_pipe_port = config["default_pipe_port"];

        const string pipe_out = 
		" appsrc ! "
		" videoconvert ! "
		" video/x-raw, format=RGBA ! "
		" nvvidconv ! "
		" omxh264enc insert-vui=true insert-sps-pps=1 ! "
		" video/x-h264, width=" + pipe_width + 
		" , height=" + pipe_height + 
		" , stream-format=byte-stream, alignment=au ! "
		" h264parse ! "
		" rtph264pay name=pay0 pt=96 ! "
		" udpsink host=" + default_pipe_host +
		" port=" + default_pipe_port +
		" sync=false async=false ";

  
        bool streaming = config["streaming"];
        bool do_face_reco = config["do_face_reco"];
        bool do_obj_det = config["do_obj_det"];
        const float DETECTION_THRESHOLD = config["detection_threshold"];
        int cam_device = config["cam_device"];

        // WARNING NO FACE AND NO YOLO
        if (!do_face_reco and !do_obj_det) {
        	std::cout << "[WARNING] No models selected." << std::endl;
             	return 0;
        }

        //FACENET
	facenetLogger gLogger = facenetLogger();
        // Register default TRT plugins (e.g. LRelu_TRT)
        if (!initLibNvInferPlugins(&gLogger, "")) { 
		return 1; 
	}
        // init streaming
       cv::VideoWriter h264_shmsink
		(pipe_out, cv::CAP_GSTREAMER, 0, 20, cv::Size (videoFrameWidth,videoFrameHeight));
          
        if (!h264_shmsink.isOpened ()) {
        	std::cout << "Failed to open h264_shmsink writer." << std::endl;
             	return (-2);
        }

	// init facenet
        FaceNetClassifier faceNet = FaceNetClassifier(gLogger, dtype, uffFile, engineFile, batchSize, serializeEngine,
            knownPersonThreshold, maxFacesPerScene, videoFrameWidth, videoFrameHeight);
        
        // init mtCNN
        mtcnn mtCNN(videoFrameHeight, videoFrameWidth);

        //init Bbox and allocate memory for "maxFacesPerScene" faces per scene
        std::vector<struct Bbox> outputBbox;
        outputBbox.reserve(maxFacesPerScene);

        // get embeddings of known faces and save face names in a vector
        std::vector<std::string> persons;
        std::vector<struct Paths> paths;
        cv::Mat image;
     
     	getFilePaths("../configs/facenet/imgs", paths);
        for(int i=0; i < paths.size(); i++) {
		loadInputImage(paths[i].absPath, image, videoFrameWidth, videoFrameHeight);
		outputBbox = mtCNN.findFace(image);
		std::size_t index = paths[i].fileName.find_last_of(".");
		std::string rawName = paths[i].fileName.substr(0,index);
		faceNet.forwardAddFace(image, outputBbox, rawName);
		faceNet.resetVariables();
		persons.push_back(rawName);
        }
        outputBbox.clear();

	int m_counts_th = 5;
	std::vector<int> m_counts(persons.size(),0);
	std::vector<int> m_time_buffer(persons.size(),5);
	std::vector<int> m_alarm_interval(persons.size(),20);
	std::vector<int> m_start_count_time(persons.size(),0);
	std::vector<int> m_alarm_time(persons.size(),0);
	std::vector<int> m_last_count_time(persons.size(),0);
	int m_time_delay = 10;
	bool do_face_reco_in_frame = false;
	bool do_object_reco_in_frame = false;
	//END FACENET
	
	// START YOLO
	// Initialize the parameters
	vector<string> classes;
	
	vector<int> interested_objects = config["interested_objects"];
       
	// length of interested objects size-> 5
	int length = interested_objects.size();
        std::cout << " Length Objects Size:" << length << endl;
      
       	// count interested objects with length of objects. Start from 0
	std::vector<int> interested_objects_counts(length, 0);
       
	//Match Objects        
	std::vector<std::string> interested_matches(10);

	// define threshold from 5 to 10
	std::vector<int> interested_objects_th(length, 10);

	// define time buffer from length of objects 5 - to 10
	//
	std::vector<int> time_buffer(length, 10);

	// define alarm interval from 5 to 20
	std::vector<int> alarm_interval(length, 20);

	// define start count time from 0 to 5
	std::vector<int> start_count_time(length, 0);

	// define alarm sound time from 0 to 5
	std::vector<int> alarm_time(length, 0);

       // define last count from 0 to 5
	std::vector<int> last_count_time(length, 0);
        
	//maximum number of seconds between two subsequent counts of the same object before reset object count	
	int time_delay = 10;

	Config config_v4_tiny;
	config_v4_tiny.net_type = YOLOV4_TINY;
	config_v4_tiny.detect_thresh = DETECTION_THRESHOLD;
	config_v4_tiny.file_model_cfg = "../Models/yolo/yolov4-tiny.cfg";
	config_v4_tiny.file_model_weights = "../Models/yolo/yolov4-tiny.weights";
	config_v4_tiny.inference_precison = FP32;

	std::unique_ptr<Detector> detector(new Detector());
	detector->init(config_v4_tiny);


        // Load names of classes
        string classesFile = "../Models/yolo/coco.names";

	// check if file exist or not
	assert(fileExists_trt(classesFile));
        
	ifstream ifs(classesFile.c_str());

        // store 4 elements for loop
	string line;
      
        while (getline(ifs, line)) classes.push_back(line);

        //Open the camera stream
/*	VideoCapture cap;
	try {
		cap.open(cam_device);
		cap.set(cv::CAP_PROP_FRAME_WIDTH, videoFrameWidth);
		cap.set(cv::CAP_PROP_FRAME_HEIGHT, videoFrameHeight);
	}
        catch(...) {
          	cout << "Could not open the input video stream" << endl;
          	return 0;
        }*/

        // Create TCP CLient
	tcp_client c;
	// localhost and port 
	c.conn(alarm_hostname, alarm_port);


        // Process frames.
	Mat frame;
	std::vector<BatchResult> batch_res;
	Timer timer;
        bool check_object;
        const auto window_name = "Display Image";
        namedWindow(window_name, WINDOW_AUTOSIZE);

	while (waitKey(1) < 0 && getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
        {
		rs2::frameset frames = p.wait_for_frames();
		rs2::depth_frame depth = frames.get_depth_frame();
		// Get the depth frame's dimensions
          //      auto width = depth.get_width();
            //    auto height = depth.get_height();
        //	f
		// get frame from the video
          	//cap >> frame;

          	// Stop the program if reached end of video
          	
		/*if (frames.empty()) {
            		cout << "Done processing !!!" << endl;
            		waitKey(3000);
            		break;
	  	}*/
		 
            
                // this will return with milliseconds by default
          	auto frame_time = std::chrono::system_clock::now();
	       
		// we convert it to second with duration_cast	
  		int m_frame_time = int(std::chrono::duration_cast<chrono::seconds>(frame_time.time_since_epoch()).count());
         
	 
         
		// START OBJECT DETECTION
          	if (do_obj_det) {
                        // Query frame size (width and height)
                          const int w = depth.as<rs2::video_frame>().get_width();
                          const int h = depth.as<rs2::video_frame>().get_height();

                         // Create OpenCV matrix of size (w,h) from the colorized depth data
                         Mat image(Size(w, h), CV_8UC3, (void*)depth.get_data(), Mat::AUTO_STEP);

                         // Update the window with new data
                         imshow(window_name, image);

       	//		rs2::depth_frame depth = frames.get_depth_frame();
                         
			 // Get the depth frame's dimensions
        //                 auto width = depth.get_width();
      //                   auto height = depth.get_height();
                
	//		 float dist_to_center = depth.get_distance(width / 2, height / 2);
          //               std::cout << "The camera is facing an object " << dist_to_center << " meters away \r";
     	      		
			//prepare batch data
	   		std::vector<cv::Mat> batch_img;
	      		batch_img.push_back(frame);
       
	      		//detect
              		timer.reset();
              		detector->detect(batch_img, batch_res);
              		timer.out("detect");
			

               		//display 
          		for (int i=0;i<batch_img.size();++i)
          		{
              			for (const auto &r : batch_res[i])
              			{
					
		     			//if at least a person is detected in the frame, set true the bool variable to pass the frame to facenet
                     			if(int(r.id) == 0) do_face_reco_in_frame = true;
									  										
                                        for (int j =0; j<interested_objects.size(); j++){
                                        				                                                 
					      do_object_reco_in_frame = true;

				            if (int(r.id)==interested_objects[j]){

                                                 // Streaming Data
                                                std::stringstream stream_last;     
                                              
					//	std::stringstream stream_first;
					        if (classes[int(r.id)]== "suitcase") 
						{ 
						  check_object = true;
					          interested_matches.push_back(classes[int(r.id)]);
 
						  // create string Suitcase
						   std::stringstream stream_object;  
                                            
 					            stream_object << "SUITCASE" << endl;

                                                    // Display Title for image
                                                   cv::putText(batch_img[i], stream_object.str(), cv::Point(r.rect.x, r.rect.y - 5), 0, 0.5, cv::Scalar(0, 0, 255), 2);

						   cv::rectangle(batch_img[i], r.rect, cv::Scalar(0, 125, 255), 2);

						  // interested_matches.reserve(classes[int(r.id)]);
						 							   
						 				
						}
						
												  
						// Check if person width gets higher value than heigth

						// Person is falled down!
                                                if ((classes[int(r.id)]== "person") && (r.rect.width > r.rect.height)){
						  // set object frame to true
						  
					           std::stringstream stream_first;                            
                    
						//   time_t current_time = time(0);
                                                  
						   stream_first << "\n**** ALARM !!! PERSON FALL DOWN ****" << endl;
                                                
						   // if person fall down, chnage color to yellow
                                                   cv::rectangle(batch_img[i], r.rect, cv::Scalar(0, 255, 255), 2);
                                                   
						  // then put text in green color
						   cv::putText(batch_img[i], stream_first.str(), cv::Point(r.rect.x, r.rect.y - 5), 0, 0.5, cv::Scalar(0, 255, 0), 2);
						  
						   string fr_output_1 = "buzzer";
					          
						   // send created json
						   c.send_data(fr_output_1 + "\n");
						
					
						}
					   					 
					      // IF Person on foot, follow that person !
					      else if (classes[int(r.id)]== "person")
					       {
                                                                 
                                                   std::stringstream stream_last;
                                                   
 					           stream_last << "*** HUMAN ON FOOT *** " << endl;

                                                     // Display Title for image
                                                   cv::putText(batch_img[i], stream_last.str(), cv::Point(r.rect.x, r.rect.y - 5), 0, 0.5, cv::Scalar(0, 0, 255), 2);
						   cv::rectangle(batch_img[i], r.rect, cv::Scalar(255, 0, 0), 2);

                                             }

                                            
                                             else if (do_object_reco_in_frame && (classes[int(r.id)]!="suitcase")){
					           for(int k = 0; k < interested_matches.size(); k++){
						     if (!(classes[int(r.id)] == interested_matches[k])){
						        std::stringstream stream_2;
                                                   
 					                stream_2 << "ANOMALY DETECTION" << endl;

                                                     // Display Title for image
                                                        cv::putText(frame, stream_2.str(), cv::Point(60, 60), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(50, 205, 154), 2);						
						}
						} 
					            // increment
                                             }
						  
						
						
                                                  
                                            } // End last if
                                       } // End third for
	                                         
	      			} // end second for
          		} // end first for 
	  	} // END OBJECT DETECTION


          	// START FACE RECOGNITION
	  	if((do_face_reco_in_frame and do_face_reco) or (do_face_reco and !do_obj_det))
          	{
	    		auto startMTCNN = chrono::steady_clock::now();
            		outputBbox = mtCNN.findFace(frame);
           		auto endMTCNN = chrono::steady_clock::now();
            		auto startForward = chrono::steady_clock::now();
            		faceNet.forward(frame, outputBbox);
            		auto endForward = chrono::steady_clock::now();
            		auto startFeatM = chrono::steady_clock::now();
            		std::map<std::string, int> personRec = faceNet.featureMatching(frame);
            		auto endFeatM = chrono::steady_clock::now();
            		faceNet.resetVariables();

            		//threshold system to send alarms
  	    		for(auto& it : personRec)
	    		{
	       			if (personRec.size() == persons.size())
	       			{
                 			for(int p=0; p < persons.size(); p++)
	         			{
                    				if (persons[p] == it.first && it.second==1)
                    				{
		        				//person seen for the first time
		        				if (m_counts[p]==0)
	                				{
                           					m_counts[p] = 1;
 		  	   					m_start_count_time[p] = m_frame_time;
			   					m_last_count_time[p] = m_frame_time;
	                 				}
		        				//increment counters
		        				if (m_counts[p]>0 && ((m_frame_time - m_start_count_time[p]) < m_time_buffer[p]) && ((m_frame_time - m_last_count_time[p]) < m_time_delay))
	                				{
                           					m_counts[p]++;
                           					m_last_count_time[p] = m_frame_time;
		        				}
		        				if (m_counts[p]<m_counts_th && (((m_frame_time-m_start_count_time[p])>=m_time_buffer[p]) || ((m_frame_time-m_last_count_time[p])>m_time_delay)))
		        				{
		           					m_counts[p] = 1;
                           					m_start_count_time[p] = m_frame_time;
                           					m_last_count_time[p] = m_frame_time;
		        				}
		        				if (m_counts[p]>=m_counts_th && (m_frame_time-m_last_count_time[p])>m_time_delay)
	                				{
			   					m_counts[p] = 1;
			   					m_start_count_time[p] = m_frame_time;
                           					m_last_count_time[p] = m_frame_time;
		        				}
		        				if (m_counts[p]>=m_counts_th && ((m_frame_time-m_last_count_time[p])<m_time_delay) && ((m_frame_time-m_start_count_time[p])>=m_time_buffer[p]) && ((m_frame_time-m_alarm_time[p]) > m_alarm_interval[p]))
		        				{
                           					//send an alarm json and reset counters
                           					time_t current_time = time(0);
                           					json fr_output =
                           					{
                              						{"Person", it.first},
                              						{"Timestamp", ctime(&current_time)}
                           					};
                           		
  		 	   					m_counts[p] = 0;
		           					m_alarm_time[p] = m_frame_time;
		        				}
		    				}
	           			}
	       			}
            		}
	 	} //END FACE RECOGNITION
    
    	        if (streaming == true) {
         		h264_shmsink.write (frame);
    		}
    	 	else {
        		cv::imshow("VideoSource", frame);
    	 	}

    	 	outputBbox.clear();
    	 	frame.release();

	} //end while

    	cv::destroyAllWindows();
    	//cap.release();
    	return 0;

} //main
