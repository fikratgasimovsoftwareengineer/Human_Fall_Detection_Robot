/*
 * Sviluppatore: Fikrat Gasimov	
 * Company: EurolinkSystems
 * Date: 19.04.2022
 * Target: AI Models with Json 
 * */

// pacchetti utili
#include "class_timer.hpp"
#include "trt_utils.h"
#include "class_detector.h"

// c++ lib duration
#include <chrono>
// defines utilities to manage
// dyanmic memory
#include <memory>
// thread allocation
#include <thread>
// dictionary lib
#include <map>

#include <string>
// Jetson Pacchetti
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <l2norm_helper.h>
// AI pacchetti
#include "faceNet.h"
#include "network.h"
#include "mtcnn.h"
// pacchetti
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <opencv4/opencv2/highgui.hpp>

// read write lib
#include <fstream>
#include <sstream>
#include <iostream>
#include "client_socket.hpp"
// json lib
#include "json.hpp"
//#include <nlohmann/json.hpp>


using namespace std;
// TensorRT  Gst-nvinfer plugin does inferencing on input data using NVIDIA® TensorRT
using namespace nvinfer1;
using namespace nvuffparser;

// opencv
using namespace cv;	
//using namespace json;
//using json = nlohmann::json;

int main(int argc, char** argv){
 
 // Load configuration from json file
 
  std::ifstream config_file("/home/eurolink/doberBot/ai_doberBot/configs/config.json", std::ifstream::binary);
  json config = json::parse(config_file);

  // FaceNet  Defined Values
  const string uffFile = "/home/eurolink/doberBot/ai_doberBot/Models/facenet/facenet.uff";
  const string engileFile = "/home/eurolink/doberBot/ai_doberBot/Models/facenet/facenet.engine";
  // data type
  nvinfer1::DataType dtype = nvinfer1::DataType::kHALF;

  // check if json file content is empty or not!
  if (config.empty()){
    cout << "Json File is Empty !" << endl;
    exit(-1);
  }
 
 // json config = json::parse(config_file);      
  // video frame width and height with INT  
  int videoFrameWidth = config["width"];
  int videoFrameHeight = config["height"]; 
  // Max face initialization
  int maxFacesPerScene = config["maxFacesPerVideo"];
  bool do_face_reco = config["do_face_reco"]; 
  // Initialized threshold
  float knownPersonThreshold = config["knownPersonThreshold"];
  
  // Object Human Detection
  bool do_human_detection = config["do_human_detection"];
  // Human Detection Threshold
  const float DETECTION_THRESHOLD = config["detection_threshold"];

  // serialized engine
  bool serializeEngine = true;
  // pass image to gpu one by one
  int batchSize = 1;

  // video frame width and height with String
  const string pipe_width =  to_string(videoFrameWidth);
  const string pipe_height = to_string(videoFrameHeight);
  const string pipe_host = config["default_pipe_host"];
  const string pipe_port = config["default_pipe_port"]; 
  // streaming set to true
  bool streaming = config["streaming"];
  
  // cam device set to 2
  int cam_device = config["cam_device"];


  //  bool streaming = true;
const string pipe_out =
	  " appsrc ! "
	  " videoconvert ! "
	  "video/x-raw, format=RGBA !"
	  " nvvidconv !"
	  " omxh264enc insert-vui=true insert-sps-pps=1 ! "
	  " video/x-h264, width =" + pipe_width + 
	  " , height= " + pipe_height + 
	  " , stream-format=byte-stream, alignment=au ! "
	  " h264parse ! "
	  " rtph264pay name=pay0 pt=96 ! "
	  " udpsink host =" + pipe_host + 
	  " port=" + pipe_port + 
 	  " sync=false async = false ";


  /*FaceNet Implementation*/
  // check if model is chosen
  if(!do_face_reco and !do_human_detection){
    std::cout << "[WARNING] No Models selected!" << std::endl;
    return 0;
  
  }

 /*FaceNet Logger*/
  facenetLogger glogger = facenetLogger();
  // Register Defaults TRT plugins(LRelu TRT)
  if(!initLibNvInferPlugins(&glogger, "")){
    return 1;
  }

          
  VideoWriter h264_shmsink
		(pipe_out, CAP_GSTREAMER, 0, 20, cv::Size (videoFrameWidth ,videoFrameHeight), true);

  if (!h264_shmsink.isOpened()){
    cout << "Failed to open h264 writer "<< endl;
    return (-1);
  };

  // Facenet Initialization
  FaceNetClassifier faceNet = FaceNetClassifier(glogger, dtype, uffFile,engileFile, batchSize, serializeEngine, knownPersonThreshold, maxFacesPerScene,videoFrameWidth, videoFrameHeight);

  // MTCNN Init
  //
  mtcnn mtCNN(videoFrameHeight, videoFrameWidth);

  // create Vector for allocation detected faces
  std::vector<struct Bbox> outputBbox;
  outputBbox.reserve(maxFacesPerScene);

  // get embeddings of known faces and save face names in a vector
  //
  std::vector<std::string> persons;
  // create struct vector for saving images from drc
  std::vector<struct Paths> paths;
 // cv Mat Image matrix 
  cv::Mat image;
  // Load Images frpm configs/imgs
  getFilePaths("/home/eurolink/doberBot/ai_doberBot/configs/facenet/imgs", paths);

  // load one by one
  //
  for (int i =0; i<paths.size(); i++){
    // loadinput image
    loadInputImage(paths[i].absPath, image, videoFrameWidth, videoFrameHeight);
    cout << "I Passed images " << endl;
    // detect face on image
    // load matrix images in vector
    outputBbox = mtCNN.findFace(image);
    // find image file with name extension
    std::size_t index = paths[i].fileName.find_last_of(".");
    // filename substr
    std::string rawName = paths[i].fileName.substr(0, index);
   
    //pass those above to facenet
    faceNet.forwardAddFace(image, outputBbox, rawName);

    //reset variables
    faceNet.resetVariables();
    // push Names of persons;
    persons.push_back(rawName);

  } // loop finish
  outputBbox.clear();

 // set number of people Max detected
  int m_counts_th = 5;
  std::vector<int>m_counts(persons.size(), 0);
  std::vector<int>m_time_buffer(persons.size(), 5);
  std::vector<int>m_alarm_interval(persons.size(), 20);
  std::vector<int>m_start_count_time(persons.size(), 0);

  std::vector<int>m_alarm_time(persons.size(), 0);
  std::vector<int>m_last_count_time(persons.size(), 0);

  int m_time_delay = 10;
  bool do_face_reco_in_frame = true;
  // FInish Face Recognition
 
  // Start Human Detection
  std::vector<std::string> classes;
  
   // load models yolo4-tiny
  Config config_v4_tiny;
  config_v4_tiny.net_type=YOLOV4_TINY;
  config_v4_tiny.detect_thresh = DETECTION_THRESHOLD;
  config_v4_tiny.file_model_cfg= "/home/eurolink/doberBot/ai_doberBot/Models/yolo/yolo4-tiny.cfg";
  config_v4_tiny.file_model_weights = "/home/eurolink/doberBot/ai_doberBot/Models/yolo/yolov4-tiny.weights";
  // Model Precision Floating Point 32
  config_v4_tiny.inference_precison=FP32;
   
  
  // boost library unique
  std::unique_ptr<Detector> detector(new Detector());
  // init yolov4-tiny parameters with smart pointer
  detector->init(config_v4_tiny);

  // load names of classes 
  std::string classesFile = "/home/eurolink/doberBot/ai_doberBot/Models/yolo/coco.names";
  // confirm that file exists
  assert(fileExists_trt(classesFile));
  
  // open file 
 
  std::ifstream ifs(classesFile.c_str());
  if(ifs.is_open()){
    
       // read by lien
    string line;
    // get by string and fill buffer
    while(std::getline(ifs,line)) {
      printf("%s",line.c_str());  
      classes.push_back(line);
    
    }
  }
  
  // end Human Object Recognition

  VideoCapture cap;

  try{
   cap.open(cam_device);
   cap.set(CAP_PROP_FRAME_WIDTH, 640);
   cap.set(CAP_PROP_FRAME_HEIGHT, 480);
  
  }

  catch(...){
    cout<< "Could not open iput video stream " << endl;
    return 0;
  }

  Mat frame;
  // Process Batch Results
  std::vector<BatchResult> batch_res;
  // set timer 
  Timer timer;

  while(waitKey(1) < 0){
    cap>>frame;

    if(frame.empty()){
      cout << "Done Processing !" << endl;
      waitKey(3000);
      break;
    
   }

    // set time detected frame
   auto frame_time = std::chrono::system_clock::now();
    
   int m_frame_time= int(std::chrono::duration_cast<chrono::seconds>(frame_time.time_since_epoch()).count());
   // Start Object Detection
   if (do_human_detection){
   
     //prepare batch data
     std::vector<cv::Mat> batch_img;
     batch_img.push_back(frame);
     
     // reset time
     timer.reset();
     detector->detect(batch_img, batch_res);
     timer.out("detect");


     for(int i=0; i<batch_img.size(); ++i){
     
       for(const auto &r: batch_res[i]){
       
       if (int (r.id)==0) do_face_reco_in_frame = true;

       cv::rectangle(batch_img[i], r.rect, cv::Scalar(255, 0, 0), 2);  

       // stream title of the detection
       std::stringstream stream;

       stream << std::fixed << std::setprecision(2) << "HUMAN: " << classes[int(r.id)];
       cv::putText(batch_img[i], stream.str(), cv::Point(r.rect.x, r.rect.y-5), 0, 0.5, 
			      cv::Scalar(0,0,255), 2);

       
       
       }
     
     
     }  
   
   }// ENd HUman Detection

  /*
   *Start Face Recognition Loop

   * */
   if((do_face_reco_in_frame and do_face_reco) or (do_face_reco and !do_human_detection)){
     // assign start time to mtcnn
     auto startMTCNN = chrono::steady_clock::now();
     // find face in the meantime of initiliaztion
     outputBbox = mtCNN.findFace(frame);
     // assign end time to mtcnn 
     auto endMTCNN = chrono::steady_clock::now();
     // continue time calculation
     auto startForward = chrono::steady_clock::now();
    
     faceNet.forward(frame, outputBbox);
     // end continuation
     auto endForward = chrono::steady_clock::now();

    
     auto startFeatM = chrono::steady_clock::now();
     // match detected face and load image from DB
     std::map<std::string, int>personRec = faceNet.featureMatching(frame);     
     auto endFeatM = chrono::steady_clock::now();
  
     faceNet.resetVariables();

     for(auto& it : personRec){
     
       if (personRec.size() == persons.size()){
       
         for (int p=0; p < persons.size(); p++){
	 
	   if(persons[p] == it.first && it.second == 1){
             
            // person seen for first time
	    //
	    if (m_counts[p] == 0){
	    
	       m_counts[p]=1;
	       m_start_count_time[p] = m_frame_time;

	       m_last_count_time[p] = m_frame_time;
	    
	    }


	    // increment counters
	    //
	    if (m_counts[p] > 0 && ((m_frame_time - m_start_count_time[p]) < m_time_buffer[p]) && ((m_frame_time - m_last_count_time[p]) < m_time_delay)){
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

} // End Face Recognition
     


  if (streaming == true){
//     cout << " *** Streaming is being written ..." << endl;
     h264_shmsink.write(frame);
 //    imshow("UDP Streaming", frame);

  }
  else{
    cout << "Streaming Remote Failed, Video Local Streaming!";	  
    imshow("VideoSource", frame);
  
  }
  frame.release();

 
  
  } //end while

  destroyAllWindows();
  cap.release();
  return 0;
} // Main finish
