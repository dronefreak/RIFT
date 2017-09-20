#include <stdio.h>
#include <string>
#include <mutex>

// ROS
#include <ros/ros.h>
#include <std_msgs/String.h> 
#include <std_msgs/Float64.h> 
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

// OpenCV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

// ALGORITHM
#include "process_image.h"
#include "system_parameters.h"

// Namespaces
using namespace std;
using namespace cv;
using namespace ros;

int s32_headerSequenceCount = 0;
std::mutex mutexObject;

// Control Parameters
float f32_linearVelocityMagnitude = 1.5f;
float f32_angularVelocityMagnitudeMax = 0.5f;

// Video File Writer
cv::VideoWriter cvVR_outVideo;
int codec = CV_FOURCC('M', 'J', 'P', 'G');
double f64_fps = 10.0;
string pc_outVideoFilename = "./output.avi";

cv::Mat frame_0;
static const std::string OUTPUT_WINDOW = "Detected Road Patch";

// Process Video File
const string pc_hardInputVideoFilename = "./bebop.mp4";
cv::VideoCapture capture;

string pc_InputFilePath = "./image_0/";
string pc_OutputFolderPath = "./image_0/out/";

cv::Mat frame;

void processFrame(cv::Mat frame, ros::Publisher twist_pub_, geometry_msgs::TwistStamped msgVel);

class ImageConverter
{
	ros::NodeHandle nh_;
	ros::Publisher twist_pub_;
	image_transport::ImageTransport it_;
	image_transport::Subscriber image_sub_;
	geometry_msgs::TwistStamped msgVel;

public:
	ImageConverter(): it_(nh_)
	{
		twist_pub_ = nh_.advertise<geometry_msgs::TwistStamped>("/mavros/setpoint_velocity/cmd_vel", 100);
		
#ifdef PROCESS_USB_WEB_CAM_FRAMES
		image_sub_ = it_.subscribe("camera/image", 1, &ImageConverter::imageCb, this);
#endif
		
		cv::namedWindow(OUTPUT_WINDOW);
	}

	~ImageConverter()
	{
		cv::destroyWindow(OUTPUT_WINDOW);
	}
	
	void imageCb(const sensor_msgs::ImageConstPtr& msg)
	{
		// Computation Time
		struct timeval start, end;
		long mtime, seconds, useconds;
		gettimeofday(&start, NULL);
		
		// Assign defaults
		msgVel.header.stamp = ros::Time::now();
		mutexObject.lock();
			msgVel.header.seq = ++s32_headerSequenceCount;
		mutexObject.unlock();
		msgVel.header.frame_id = 1;
		msgVel.twist.linear.x = 0.f;
	  	msgVel.twist.linear.y = 0.f;
	  	msgVel.twist.linear.z = 0.f;
	  	msgVel.twist.angular.x = 0.f;
	  	msgVel.twist.angular.y = 0.f;
	  	msgVel.twist.angular.z = 0.f;
		
		cv_bridge::CvImagePtr cv_ptr;
		try
		{
		  cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
		}
		catch (cv_bridge::Exception& e)
		{

		  ROS_ERROR("cv_bridge exception: %s", e.what());
		  return;
		}
		
		frame = cv_ptr-> image;
		
		if (frame.empty())
		{
			twist_pub_.publish(msgVel);
			return;
		}
		
	   	float f32_steer;
		cv::Mat cvMat_outputImgBGRresized = cv::Mat::zeros(TESTING_OUTPUT_IMAGE_HEIGHT, TESTING_OUTPUT_IMAGE_WIDTH, CV_8UC3);
		ProcessImage(frame, cvMat_outputImgBGRresized, f32_steer);
		
		msgVel.twist.linear.x = +f32_linearVelocityMagnitude;
	   	msgVel.twist.angular.z = f32_steer * f32_angularVelocityMagnitudeMax;
		twist_pub_.publish(msgVel);
		
		cv::imshow(OUTPUT_WINDOW, cvMat_outputImgBGRresized);
		cv::waitKey(30);
		
		// Computation Time
		gettimeofday(&end, NULL);
		seconds  = end.tv_sec  - start.tv_sec;
		useconds = end.tv_usec - start.tv_usec;
		mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
		printf("Elapsed time: %ld milliseconds\n", mtime);
	}
	
	void processFrame(cv::Mat frame_0)
	{
		// Computation Time
		struct timeval start, end;
		long mtime, seconds, useconds;
		gettimeofday(&start, NULL);
		
		// Assign defaults
		msgVel.header.stamp = ros::Time::now();
		mutexObject.lock();
			msgVel.header.seq = ++s32_headerSequenceCount;
		mutexObject.unlock();
		msgVel.header.frame_id = 1;
		msgVel.twist.linear.x = 0.f;
	  	msgVel.twist.linear.y = 0.f;
	  	msgVel.twist.linear.z = 0.f;
	  	msgVel.twist.angular.x = 0.f;
	  	msgVel.twist.angular.y = 0.f;
	  	msgVel.twist.angular.z = 0.f;
		
		if (frame_0.empty())
		{
			twist_pub_.publish(msgVel);
			return;
		}
		
	   	float f32_steer;
		cv::Mat cvMat_outputImgBGRresized = cv::Mat::zeros(TESTING_OUTPUT_IMAGE_HEIGHT, TESTING_OUTPUT_IMAGE_WIDTH, CV_8UC3);
		ProcessImage(frame_0, cvMat_outputImgBGRresized, f32_steer);
		
		msgVel.twist.linear.x = +f32_linearVelocityMagnitude;
	   	msgVel.twist.angular.z = f32_steer * f32_angularVelocityMagnitudeMax;
		twist_pub_.publish(msgVel);
		
		cv::imshow(OUTPUT_WINDOW, cvMat_outputImgBGRresized);
		cv::waitKey(30);
		
		// Computation Time
		gettimeofday(&end, NULL);
		seconds  = end.tv_sec  - start.tv_sec;
		useconds = end.tv_usec - start.tv_usec;
		mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
		printf("Elapsed time: %ld milliseconds\n", mtime);
	}
};
 
int main(int argc, char **argv)
{
   ros::init(argc, argv, "RoadFollowing");
   
#ifndef PROCESS_USB_WEB_CAM_FRAMES
   // Process Video File
   capture.open(pc_hardInputVideoFilename);
#endif
   
   ImageConverter ic;
   
#ifndef PROCESS_USB_WEB_CAM_FRAMES
   capture >> frame;
   while (!frame.empty())
   {
   		ic.processFrame(frame);
		capture >> frame;
   }
#endif
   
   ros::spin();
   
   return 0;
}
