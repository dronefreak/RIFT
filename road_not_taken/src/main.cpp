/**
 * @file main.cpp
 * @brief ROS node for road detection and autonomous navigation
 *
 * This node processes camera images to detect roads and publishes velocity
 * commands for autonomous navigation. It supports both live camera feeds
 * and pre-recorded video files.
 */

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

// Global counters and synchronization
int s32_headerSequenceCount = 0;
std::mutex mutexObject;

// Control Parameters
const float DEFAULT_LINEAR_VELOCITY = 1.5f;
const float DEFAULT_ANGULAR_VELOCITY_MAX = 0.5f;

// Video File Writer Parameters
const double DEFAULT_FPS = 10.0;
const std::string DEFAULT_OUTPUT_VIDEO = "./output.avi";

// Video input parameters
const std::string DEFAULT_INPUT_VIDEO = "./bebop.mp4";
const std::string DEFAULT_INPUT_PATH = "./image_0/";
const std::string DEFAULT_OUTPUT_PATH = "./image_0/out/";

// Window name
static const std::string OUTPUT_WINDOW = "Detected Road Patch";

/**
 * @class ImageConverter
 * @brief Handles image processing and velocity command publication for road following
 */
class ImageConverter
{
private:
	ros::NodeHandle nh_;
	ros::Publisher twist_pub_;
	image_transport::ImageTransport it_;
	image_transport::Subscriber image_sub_;
	geometry_msgs::TwistStamped msgVel;
	float linearVelocity;
	float angularVelocityMax;

	/**
	 * @brief Initialize velocity message with default values
	 */
	void initializeVelocityMessage()
	{
		msgVel.header.stamp = ros::Time::now();
		mutexObject.lock();
		msgVel.header.seq = ++s32_headerSequenceCount;
		mutexObject.unlock();
		msgVel.header.frame_id = 1;
		msgVel.twist.linear.x = 0.0f;
		msgVel.twist.linear.y = 0.0f;
		msgVel.twist.linear.z = 0.0f;
		msgVel.twist.angular.x = 0.0f;
		msgVel.twist.angular.y = 0.0f;
		msgVel.twist.angular.z = 0.0f;
	}

public:
	/**
	 * @brief Constructor - initializes ROS node, publishers, and subscribers
	 */
	ImageConverter() : it_(nh_)
	{
		// Get parameters from ROS parameter server with defaults
		nh_.param("linear_velocity", linearVelocity, DEFAULT_LINEAR_VELOCITY);
		nh_.param("angular_velocity_max", angularVelocityMax, DEFAULT_ANGULAR_VELOCITY_MAX);

		ROS_INFO("Road Following Node starting with:");
		ROS_INFO("  Linear velocity: %.2f m/s", linearVelocity);
		ROS_INFO("  Max angular velocity: %.2f rad/s", angularVelocityMax);

		// Setup publisher
		twist_pub_ = nh_.advertise<geometry_msgs::TwistStamped>("/mavros/setpoint_velocity/cmd_vel", 100);

#ifdef PROCESS_USB_WEB_CAM_FRAMES
		// Subscribe to camera topic
		image_sub_ = it_.subscribe("camera/image", 1, &ImageConverter::imageCb, this);
		ROS_INFO("Subscribed to camera/image topic");
#else
		ROS_INFO("Processing video file mode");
#endif

		// Create display window
		cv::namedWindow(OUTPUT_WINDOW);
	}

	/**
	 * @brief Destructor - cleanup resources
	 */
	~ImageConverter()
	{
		cv::destroyWindow(OUTPUT_WINDOW);
	}
	
	/**
	 * @brief Process image frame and compute control commands
	 * @param frame Input image frame
	 */
	void processImageFrame(const cv::Mat& frame)
	{
		// Start timing
		struct timeval start, end;
		gettimeofday(&start, NULL);

		// Initialize velocity message
		initializeVelocityMessage();

		// Check if frame is valid
		if (frame.empty())
		{
			ROS_WARN_THROTTLE(1.0, "Received empty frame");
			twist_pub_.publish(msgVel);
			return;
		}

		// Process image to detect road and compute steering
		float steer = 0.0f;
		cv::Mat outputImage = cv::Mat::zeros(TESTING_OUTPUT_IMAGE_HEIGHT,
		                                     TESTING_OUTPUT_IMAGE_WIDTH,
		                                     CV_8UC3);

		ProcessImage(frame, outputImage, steer);

		// Set velocity commands
		msgVel.twist.linear.x = linearVelocity;
		msgVel.twist.angular.z = steer * angularVelocityMax;

		// Publish velocity command
		twist_pub_.publish(msgVel);

		// Display result
		cv::imshow(OUTPUT_WINDOW, outputImage);
		cv::waitKey(30);

		// Compute and log processing time
		gettimeofday(&end, NULL);
		long seconds = end.tv_sec - start.tv_sec;
		long useconds = end.tv_usec - start.tv_usec;
		long mtime = (seconds * 1000 + useconds / 1000.0) + 0.5;
		ROS_DEBUG("Processing time: %ld ms", mtime);
	}

	/**
	 * @brief Callback for camera image messages
	 * @param msg ROS image message
	 */
	void imageCb(const sensor_msgs::ImageConstPtr& msg)
	{
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

		if (!cv_ptr)
		{
			ROS_ERROR("Failed to convert image");
			return;
		}

		processImageFrame(cv_ptr->image);
	}
	
	/**
	 * @brief Process a single video frame (for video file input)
	 * @param frame Input frame from video file
	 */
	void processFrame(const cv::Mat& frame)
	{
		processImageFrame(frame);
	}
};
 
/**
 * @brief Main function - initializes node and starts processing
 */
int main(int argc, char **argv)
{
	// Initialize ROS node
	ros::init(argc, argv, "RoadFollowing");
	ros::NodeHandle nh;

	ROS_INFO("==========================================================");
	ROS_INFO("Road Following Node - Starting");
	ROS_INFO("==========================================================");

#ifndef PROCESS_USB_WEB_CAM_FRAMES
	// Video file processing mode
	std::string video_filename = DEFAULT_INPUT_VIDEO;
	nh.param("video_file", video_filename, DEFAULT_INPUT_VIDEO);

	ROS_INFO("Opening video file: %s", video_filename.c_str());

	cv::VideoCapture capture;
	capture.open(video_filename);

	if (!capture.isOpened())
	{
		ROS_ERROR("Failed to open video file: %s", video_filename.c_str());
		ROS_ERROR("Please check:");
		ROS_ERROR("  1. File exists");
		ROS_ERROR("  2. File path is correct");
		ROS_ERROR("  3. OpenCV supports the video format");
		return 1;
	}

	ROS_INFO("Video file opened successfully");
	ROS_INFO("  Frame width: %.0f", capture.get(cv::CAP_PROP_FRAME_WIDTH));
	ROS_INFO("  Frame height: %.0f", capture.get(cv::CAP_PROP_FRAME_HEIGHT));
	ROS_INFO("  FPS: %.0f", capture.get(cv::CAP_PROP_FPS));
	ROS_INFO("  Total frames: %.0f", capture.get(cv::CAP_PROP_FRAME_COUNT));
#endif

	// Create image converter/processor
	ImageConverter ic;

#ifndef PROCESS_USB_WEB_CAM_FRAMES
	// Process all frames from video file
	cv::Mat frame;
	int frame_count = 0;
	capture >> frame;

	while (!frame.empty() && ros::ok())
	{
		ic.processFrame(frame);
		capture >> frame;
		frame_count++;

		if (frame_count % 100 == 0)
		{
			ROS_INFO("Processed %d frames", frame_count);
		}

		// Allow ROS to process callbacks
		ros::spinOnce();
	}

	ROS_INFO("Finished processing %d frames", frame_count);
	capture.release();
#else
	// Live camera mode - use ROS spin
	ROS_INFO("Running in live camera mode");
	ros::spin();
#endif

	ROS_INFO("Road Following Node - Shutting down");
	return 0;
}

