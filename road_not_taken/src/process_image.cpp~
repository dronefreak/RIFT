#include <stdio.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <sys/time.h>

#include "system_parameters.h"

using namespace std;

// Algorithm Settings
#define PROCESSED_IMAGE_HEIGHT	100
#define PROCESSED_IMAGE_WIDTH	100

// Initialization
cv::Size processedImgSize(PROCESSED_IMAGE_WIDTH, PROCESSED_IMAGE_HEIGHT);
cv::Mat cvMat_imgBGRresized = cv::Mat::zeros(PROCESSED_IMAGE_HEIGHT, PROCESSED_IMAGE_WIDTH, CV_8UC3);
cv::Mat cvMat_outputImgBGRBuffer = cv::Mat::zeros(PROCESSED_IMAGE_HEIGHT, PROCESSED_IMAGE_WIDTH, CV_8UC3);
cv::Mat cvMat_YChannel = cv::Mat::zeros(PROCESSED_IMAGE_HEIGHT, PROCESSED_IMAGE_WIDTH, CV_8UC1);
cv::Mat cvMat_CbChannel = cv::Mat::zeros(PROCESSED_IMAGE_HEIGHT, PROCESSED_IMAGE_WIDTH, CV_8UC1);
cv::Mat cvMat_CrChannel = cv::Mat::zeros(PROCESSED_IMAGE_HEIGHT, PROCESSED_IMAGE_WIDTH, CV_8UC1);

// Flood Fill
cv::Mat cvMat_roadLikePixels = cv::Mat::zeros(PROCESSED_IMAGE_HEIGHT, PROCESSED_IMAGE_WIDTH, CV_8UC3);
cv::Mat cvMat_roadLikePixelsClone = cv::Mat::zeros(PROCESSED_IMAGE_HEIGHT, PROCESSED_IMAGE_WIDTH, CV_8UC3);

// Hybrid
cv::Mat cvMat_imgEdgesClone = cv::Mat::zeros(PROCESSED_IMAGE_HEIGHT, PROCESSED_IMAGE_WIDTH, CV_8UC1);

// Canny
cv::Mat cvMat_imgGrayResized = cv::Mat::zeros(PROCESSED_IMAGE_HEIGHT, PROCESSED_IMAGE_WIDTH, CV_8UC1);
cv::Mat cvMat_imgEdges = cv::Mat::zeros(PROCESSED_IMAGE_HEIGHT, PROCESSED_IMAGE_WIDTH, CV_8UC1);
int edgeThresh = 1;
int lowThreshold = 20;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;


int ProcessImage(cv::Mat cvMat_imgBGR, cv::Mat cvMat_outputImgBGRresized, float &f32_steer)
{
	// Read & Resize
	cv::resize(cvMat_imgBGR, cvMat_imgBGRresized, processedImgSize, 0, 0, cv::INTER_LINEAR); // Bilinear Interpolation
	
	// Keep a clone of the input
	cvMat_outputImgBGRBuffer = cvMat_imgBGRresized.clone();
	
	// Split Input Image to RGB Channels
	std::vector <cv::Mat> BGRChannels(3);
	cv::split(cvMat_imgBGRresized, BGRChannels);
	cv::Mat cvMat_RChannel = BGRChannels[2];
	cv::Mat cvMat_GChannel = BGRChannels[1];
	cv::Mat cvMat_BChannel = BGRChannels[0];
	
	// Split Output Image to RGB Channels
	std::vector <cv::Mat> BGRChannelsOutput(3); // Check for memory alloc
	cv::split(cvMat_outputImgBGRBuffer, BGRChannelsOutput);
	cv::Mat cvMat_RChannelOutput = BGRChannelsOutput[2];
	cv::Mat cvMat_GChannelOutput = BGRChannelsOutput[1];
	cv::Mat cvMat_BChannelOutput = BGRChannelsOutput[0];
	
	// cvMat_roadLikePixels
	std::vector <cv::Mat> BGRChannelsRoadLikePixels(3); // Check for memory alloc
	cv::split(cvMat_roadLikePixels, BGRChannelsRoadLikePixels);
	cv::Mat cvMat_RChannelRoadLikePixels = BGRChannelsRoadLikePixels[2];
	cv::Mat cvMat_GChannelRoadLikePixels = BGRChannelsRoadLikePixels[1];
	cv::Mat cvMat_BChannelRoadLikePixels = BGRChannelsRoadLikePixels[0];
	
	//////////////////////////////////////////////////////////////////////
	// Canny Edge Detection
	
	cv::cvtColor(cvMat_imgBGRresized, cvMat_imgGrayResized, CV_BGR2GRAY);
	cv::blur(cvMat_imgGrayResized, cvMat_imgEdges, cv::Size(3,3));
  	cv::Canny(cvMat_imgEdges, cvMat_imgEdges, lowThreshold, lowThreshold*ratio, kernel_size);
	
	//imshow("Canny Edge Detection", cvMat_imgEdges);
	//////////////////////////////////////////////////////////////////////
	
	//////////////////////////////////////////////////////////////////////
	// Spectral Processing
	
	for (int rowIndex = 0; rowIndex < PROCESSED_IMAGE_HEIGHT; rowIndex++)
	{
		for (int columnIndex = 0; columnIndex < PROCESSED_IMAGE_WIDTH; columnIndex++)
		{			
			cvMat_RChannelRoadLikePixels.at<unsigned char>(rowIndex, columnIndex) = (unsigned char)0;
			cvMat_GChannelRoadLikePixels.at<unsigned char>(rowIndex, columnIndex) = (unsigned char)0;
			cvMat_BChannelRoadLikePixels.at<unsigned char>(rowIndex, columnIndex) = (unsigned char)0;
			
			// Eliminating shades of grey close to pure black and pure white
			if ((cvMat_RChannel.at<unsigned char>(rowIndex, columnIndex) >= (unsigned char)25) && (cvMat_RChannel.at<unsigned char>(rowIndex, columnIndex) <= (unsigned char)235))
			{
				if ((cvMat_GChannel.at<unsigned char>(rowIndex, columnIndex) >= (unsigned char)25) && (cvMat_GChannel.at<unsigned char>(rowIndex, columnIndex) <= (unsigned char)235))
				{
					if ((cvMat_BChannel.at<unsigned char>(rowIndex, columnIndex) >= (unsigned char)25) && (cvMat_BChannel.at<unsigned char>(rowIndex, columnIndex) <= (unsigned char)235))
					{
						// Color Saturation - Red or Blue must stay within their limits to avoid expulsion
						float f32_RGBMean = (float)(cvMat_RChannel.at<unsigned char>(rowIndex, columnIndex) + cvMat_BChannel.at<unsigned char>(rowIndex, columnIndex) + cvMat_GChannel.at<unsigned char>(rowIndex, columnIndex)) / 3.f;
						float f32_DeviationR = (cvMat_RChannel.at<unsigned char>(rowIndex, columnIndex) - f32_RGBMean);
						float f32_DeviationG = (cvMat_GChannel.at<unsigned char>(rowIndex, columnIndex) - f32_RGBMean);
						float f32_DeviationB = (cvMat_BChannel.at<unsigned char>(rowIndex, columnIndex) - f32_RGBMean);
						
						// Color Temperature - Redness - Sunlight, Greenness - Sunlight passing through thin leaves, Blueness - Cloudy/DimLight
						if (f32_DeviationR <= 17)
						{
							if (f32_DeviationG <= 11)
							{
								if (f32_DeviationB <= 13)
								{
									cvMat_RChannelRoadLikePixels.at<unsigned char>(rowIndex, columnIndex) = (unsigned char)0;
									cvMat_GChannelRoadLikePixels.at<unsigned char>(rowIndex, columnIndex) = (unsigned char)0;
									cvMat_BChannelRoadLikePixels.at<unsigned char>(rowIndex, columnIndex) = (unsigned char)255;
								}
							}
						}
					}
				}
			}			
		}
	}
	
	// Merge RGB Channels to Output Image
	std::vector <cv::Mat> TempBGRChannelsRoadLikePixels;
	TempBGRChannelsRoadLikePixels.push_back(cvMat_BChannelRoadLikePixels);
	TempBGRChannelsRoadLikePixels.push_back(cvMat_GChannelRoadLikePixels);
	TempBGRChannelsRoadLikePixels.push_back(cvMat_RChannelRoadLikePixels);
	cv::merge(TempBGRChannelsRoadLikePixels, cvMat_roadLikePixels);
	
	//imshow("Spectral Filtering", cvMat_roadLikePixels);
	///////////////////////////////////////////////////////////////////////
	
	///////////////////////////////////////////////////////////////////////
	// Integration of Edge Detection Output with Spectral Filtering Output
	
	for (int rowIndex = 0; rowIndex < PROCESSED_IMAGE_HEIGHT; rowIndex++)
	{
		for (int columnIndex = 0; columnIndex < PROCESSED_IMAGE_WIDTH; columnIndex++)
		{
			if (cvMat_BChannelRoadLikePixels.at<unsigned char>(rowIndex, columnIndex) != (unsigned char)255)
			{
				cvMat_imgEdges.at<unsigned char>(rowIndex, columnIndex) = (unsigned char)255;
			}
		}
	}
	//imshow("Composite Edge Output", cvMat_imgEdges);
	///////////////////////////////////////////////////////////////////////
	
	///////////////////////////////////////////////////////////////////////
	// Filtering based on Composite Edge Energy
	int s32_windowSize = 5;
	int s32_offset = (s32_windowSize - 1) / 2;
	for (int rowIndex = 0 + s32_offset; rowIndex < PROCESSED_IMAGE_HEIGHT - s32_offset; rowIndex = rowIndex + s32_windowSize)
	{
		for (int columnIndex = 0 + s32_offset; columnIndex < PROCESSED_IMAGE_WIDTH - s32_offset; columnIndex = columnIndex + s32_windowSize)
		{
			// Edge Energy Window
			int s32_energySum = 0;
			bool b_flag = false;
			for (int i = -s32_offset; i <= +s32_offset; i++)
			{
				for (int j = -s32_offset; j <= +s32_offset; j++)
				{
					if (cvMat_imgEdges.at<unsigned char>(rowIndex + i, columnIndex + j) == (unsigned char)255)
					{
						s32_energySum += 255;
					}
				}
			}
			
			if (s32_energySum >= (255 * 5))
			{
				for (int i = -s32_offset; i <= +s32_offset; i++)
				{
					for (int j = -s32_offset; j <= +s32_offset; j++)
					{
						cvMat_imgEdges.at<unsigned char>(rowIndex + i, columnIndex + j) = (unsigned char)255;
					}
				}
			}
		}
	}
	
	//imshow("Edge-based Filtering", cvMat_imgEdges);
	///////////////////////////////////////////////////////////////////////
	
	///////////////////////////////////////////////////////////////////////
	// Flood Fill
	
	// Parameters
	cv::Point cvPt_seedPoint = cv::Point((PROCESSED_IMAGE_WIDTH / 2) - 1, PROCESSED_IMAGE_HEIGHT - 1);
	cv::Scalar cvSclr_newColor = cv::Scalar(0, 255, 0);
	cv::Rect cvRect_enigmaRect;
	cv::Scalar cvSclr_loDiff = cv::Scalar(20, 20, 20);
	cv::Scalar cvSclr_upDiff = cv::Scalar(20, 20, 20);
	int s32_flags = 8 | ( 180 << 8 );
	
	cv::Point cvPtList_bestSeeds[3];
	int ps32_bestPixelsList[3];
	int s32_bestSeedsIndex = 0;
	int s32_currentMaxPixels = -1;
	int s32_prevMaxPixels = 100000000;
	
	for (int i = 0; i < 3; i++)
	{
		cvPtList_bestSeeds[i].x = -1;
		cvPtList_bestSeeds[i].y = -1;
		ps32_bestPixelsList[i] = -1;
	}
	
	for (int i = 0; i < 3; i++)
	{
		for (int rowIndex = (int)((1.f) * (float)cvMat_imgEdges.rows) - 1; rowIndex >= (int)((1.f/2.f) * (float)cvMat_imgEdges.rows) - 1; rowIndex = rowIndex - ((int)((1.f/16.f) * (float)PROCESSED_IMAGE_HEIGHT) - 1))
		{
			for (int columnIndex = (int)((1.f/4.f) * (float)cvMat_imgEdges.cols) - 1; columnIndex <= (int)((3.f/4.f) * (float)cvMat_imgEdges.cols) - 1; columnIndex = columnIndex + ((int)((1.f/16.f) * (float)PROCESSED_IMAGE_WIDTH) - 1))
			{
				if (cvMat_imgEdges.at<unsigned char>(rowIndex, columnIndex) == (unsigned char)0)
				{
					cvMat_imgEdgesClone = cvMat_imgEdges.clone();
					int s32_pixels = cv::floodFill(cvMat_imgEdgesClone, cv::Point(columnIndex, rowIndex), cv::Scalar(30), &cvRect_enigmaRect, cvSclr_loDiff, cvSclr_upDiff, 8);
					
					if ((s32_pixels > s32_currentMaxPixels) && (s32_pixels < s32_prevMaxPixels))
					{
						s32_currentMaxPixels = s32_pixels;
						cvPtList_bestSeeds[s32_bestSeedsIndex] = cv::Point(columnIndex, rowIndex);
						ps32_bestPixelsList[s32_bestSeedsIndex] = s32_pixels;
					}
				}
			}
		}
		
		if (ps32_bestPixelsList[s32_bestSeedsIndex] > -1)
		{
			s32_prevMaxPixels = s32_currentMaxPixels;
			s32_currentMaxPixels = -1;
			s32_bestSeedsIndex++;
		}
		else
		{
			break;
		}
	}
	
	int s32_numberOfPixelsRepainted = 0;
	for (int i = 0; i < 1; i++)
	{
		//if (ps32_bestPixelsList[i] >= 0/*400*/)	// > -1 is automatically taken care of
		if (ps32_bestPixelsList[i] >= (int)(0.30f * (float)ps32_bestPixelsList[0]))
		{
			s32_numberOfPixelsRepainted = cv::floodFill(cvMat_imgEdges, cvPtList_bestSeeds[i], cv::Scalar(30), &cvRect_enigmaRect, cvSclr_loDiff, cvSclr_upDiff, 8);
		}
		else
		{
			break;
		}
	}
	
	//imshow("pre-final flood", cvMat_imgEdges);
	///////////////////////////////////////////////////////////////////////
	
	// POST PROC
	// Edge Energy Based Filtering (ACTUALLY, INVERSE ENERGY)
	/*int */s32_windowSize = 5;//5;	// ODD *****
	/*int */s32_offset = (s32_windowSize - 1) / 2;
	for (int rowIndex = 0 + s32_offset; rowIndex < PROCESSED_IMAGE_HEIGHT - s32_offset; rowIndex = rowIndex + s32_windowSize)
	{
		for (int columnIndex = 0 + s32_offset; columnIndex < PROCESSED_IMAGE_WIDTH - s32_offset; columnIndex = columnIndex + s32_windowSize)
		{
			// Edge Energy Window
			int s32_energySum = 0;
			bool b_flag = false;
			for (int i = -s32_offset; i <= +s32_offset; i++)
			{
				for (int j = -s32_offset; j <= +s32_offset; j++)
				{
					if (cvMat_imgEdges.at<unsigned char>(rowIndex + i, columnIndex + j) == (unsigned char)30)
					{
						s32_energySum += 255;
					}
				}
			}
			
			if (s32_energySum >= (255 * 3/*5*/))//0)		// *****
			{
				for (int i = -s32_offset; i <= +s32_offset; i++)
				{
					for (int j = -s32_offset; j <= +s32_offset; j++)
					{
						cvMat_imgEdges.at<unsigned char>(rowIndex + i, columnIndex + j) = (unsigned char)30;
					}
				}
			}
		}
	}
	
	//imshow("Pre-final output", cvMat_imgEdges);
	///////////////////////////////////////////////////////////////////////
	
	///////////////////////////////////////////////////////////////////////
	// Overlay output on input image
	
	for (int rowIndex = 0; rowIndex < PROCESSED_IMAGE_HEIGHT; rowIndex++)
	{
		for (int columnIndex = 0; columnIndex < PROCESSED_IMAGE_WIDTH; columnIndex++)
		{
			if (cvMat_imgEdges.at<unsigned char>(rowIndex + 1, columnIndex + 1) == (unsigned char)30)
			{
				cvMat_RChannelOutput.at<unsigned char>(rowIndex, columnIndex) = (unsigned char)0;
				cvMat_GChannelOutput.at<unsigned char>(rowIndex, columnIndex) = (unsigned char)255;
				cvMat_BChannelOutput.at<unsigned char>(rowIndex, columnIndex) = (unsigned char)0;
			}
		}
	}
	
	///////////////////////////////////////////////////////////////////////
	
	// Compute Road Centroid
	float f32_meanX = 0.f;
	float f32_meanY = 0.f;
	int s32_pixelCounter = 0;
	
	for (int rowIndex = PROCESSED_IMAGE_HEIGHT - 1; rowIndex >= 0; rowIndex--)
	{
		for (int columnIndex = 0; columnIndex < PROCESSED_IMAGE_WIDTH; columnIndex++)
		{
			if ((cvMat_RChannelOutput.at<unsigned char>(rowIndex, columnIndex) == (unsigned char)0) && (cvMat_GChannelOutput.at<unsigned char>(rowIndex, columnIndex) == (unsigned char)255) && (cvMat_BChannelOutput.at<unsigned char>(rowIndex, columnIndex) == (unsigned char)0))
			{
				f32_meanX += columnIndex;
				f32_meanY += rowIndex;
				s32_pixelCounter++;
			}
		}
	}
	
	// Handle divide_by_0
	if (s32_pixelCounter == 0)
	{
		s32_pixelCounter = 1;
	}
	
	f32_meanX /= s32_pixelCounter;
	f32_meanY /= s32_pixelCounter;
	int s32_centroidX1 = (int)f32_meanX;
	int s32_centroidY1 = (int)f32_meanY;
	
	// Indicate Centroid on Image
	if (((s32_centroidY1 > 0) && (s32_centroidY1 < PROCESSED_IMAGE_HEIGHT)) && ((s32_centroidX1 > 0) && (s32_centroidX1 < PROCESSED_IMAGE_WIDTH)))
	{
		for (int i = -2; i <= +2; i++)
		{
			for (int j = -2; j <= +2; j++)
			{
				cvMat_RChannelOutput.at<unsigned char>(s32_centroidY1 + i, s32_centroidX1 + j) = (unsigned char)0;
				cvMat_GChannelOutput.at<unsigned char>(s32_centroidY1 + i, s32_centroidX1 + j) = (unsigned char)0;
				cvMat_BChannelOutput.at<unsigned char>(s32_centroidY1 + i, s32_centroidX1 + j) = (unsigned char)255;
			}	
		}
	}
	
	// Computing Normalized Angular Velocity Magnitude and indicating it on image
	float f32_targetCentroid = s32_centroidX1;
	float f32_xDeviation = f32_targetCentroid - (float)((PROCESSED_IMAGE_WIDTH / 2) - 1);
	float f32_xDeviationNormalized = f32_xDeviation / ((float)PROCESSED_IMAGE_WIDTH / 2.f);
	float f32_xDeviationNormalizedThreshold = 0.f;
	
	// Indicate Angular Z Magnitude
	if (f32_xDeviationNormalized < 0.f)
	{
		f32_steer = -f32_xDeviationNormalized;
	
		// Indicate Angular Z (+) Magnitude -> Left Rotation
		int s32_horOffset = 2 * (PROCESSED_IMAGE_HEIGHT / 8);//(PROCESSED_IMAGE_WIDTH / 2);
		int s32_vertOffset = (PROCESSED_IMAGE_HEIGHT / 8);
		for (int horIndex = 1; horIndex < (PROCESSED_IMAGE_HEIGHT / 8); horIndex++)
		{
			for (int vertIndex = 1; vertIndex < (PROCESSED_IMAGE_HEIGHT / 8); vertIndex++)
			{
				if (vertIndex > horIndex)
				{
					cvMat_RChannelOutput.at<unsigned char>(s32_vertOffset + vertIndex, s32_horOffset - horIndex) = (unsigned char)255;
					cvMat_GChannelOutput.at<unsigned char>(s32_vertOffset + vertIndex, s32_horOffset - horIndex) = (unsigned char)0;
					cvMat_BChannelOutput.at<unsigned char>(s32_vertOffset + vertIndex, s32_horOffset - horIndex) = (unsigned char)(0);// * -f32_xDeviationNormalized);
				}
			}
		}	
	
		s32_vertOffset = (PROCESSED_IMAGE_HEIGHT / 8) * 3 - 1;
		for (int horIndex = 1; horIndex < (PROCESSED_IMAGE_HEIGHT / 8); horIndex++)
		{
			for (int vertIndex = 1; vertIndex < (PROCESSED_IMAGE_HEIGHT / 8); vertIndex++)
			{
				if (vertIndex > horIndex)
				{
					cvMat_RChannelOutput.at<unsigned char>(s32_vertOffset - vertIndex, s32_horOffset - horIndex) = (unsigned char)255;
					cvMat_GChannelOutput.at<unsigned char>(s32_vertOffset - vertIndex, s32_horOffset - horIndex) = (unsigned char)0;
					cvMat_BChannelOutput.at<unsigned char>(s32_vertOffset - vertIndex, s32_horOffset - horIndex) = (unsigned char)(0);// * -f32_xDeviationNormalized);
				}
			}
		}
	}
	else if (f32_xDeviationNormalized > 0.f)
	{
		f32_steer = -f32_xDeviationNormalized;
	
		// Indicate Angular Z (-) Magnitude -> Right Rotation
		int s32_horOffset = PROCESSED_IMAGE_WIDTH - ((PROCESSED_IMAGE_HEIGHT / 8) * 2);
		int s32_vertOffset = (PROCESSED_IMAGE_HEIGHT / 8);
		for (int horIndex = 1; horIndex < (PROCESSED_IMAGE_HEIGHT / 8); horIndex++)
		{
			for (int vertIndex = 1; vertIndex < (PROCESSED_IMAGE_HEIGHT / 8); vertIndex++)
			{
				if (vertIndex > horIndex)
				{
					cvMat_RChannelOutput.at<unsigned char>(s32_vertOffset + vertIndex, s32_horOffset + horIndex) = (unsigned char)255;
					cvMat_GChannelOutput.at<unsigned char>(s32_vertOffset + vertIndex, s32_horOffset + horIndex) = (unsigned char)0;
					cvMat_BChannelOutput.at<unsigned char>(s32_vertOffset + vertIndex, s32_horOffset + horIndex) = (unsigned char)(0);// * -f32_xDeviationNormalized);
				}
			}
		}
		s32_vertOffset = (PROCESSED_IMAGE_HEIGHT / 8) * 3 - 1;
		for (int horIndex = 1; horIndex < (PROCESSED_IMAGE_HEIGHT / 8); horIndex++)
		{
			for (int vertIndex = 1; vertIndex < (PROCESSED_IMAGE_HEIGHT / 8); vertIndex++)
			{
				if (vertIndex > horIndex)
				{
					cvMat_RChannelOutput.at<unsigned char>(s32_vertOffset - vertIndex, s32_horOffset + horIndex) = (unsigned char)255;
					cvMat_GChannelOutput.at<unsigned char>(s32_vertOffset - vertIndex, s32_horOffset + horIndex) = (unsigned char)0;
					cvMat_BChannelOutput.at<unsigned char>(s32_vertOffset - vertIndex, s32_horOffset + horIndex) = (unsigned char)(0);// * -f32_xDeviationNormalized);
				}
			}
		}
	}
	else
	{
		f32_steer = 0.f;
	}
	
	// Merge RGB Channels to Output Image
	std::vector <cv::Mat> outputBGRChannels;
	outputBGRChannels.push_back(cvMat_BChannelOutput);
	outputBGRChannels.push_back(cvMat_GChannelOutput);
	outputBGRChannels.push_back(cvMat_RChannelOutput);
	cv::merge(outputBGRChannels, cvMat_outputImgBGRBuffer);
	
	cv::resize(cvMat_outputImgBGRBuffer, cvMat_outputImgBGRresized, cvMat_outputImgBGRresized.size(), 0, 0, cv::INTER_LINEAR); // Bilinear Interpolation
	
	return 0;
}
