#pragma once
#include "headfile.h"
#include"MatHelper.h"
#include"ForExp.h"
#include "rgbd2vec.h"
class Confirm{
public:
	static Mat getVector(const Mat &depthFrame, const Mat &rgbFrame, const Mat &pointCloud, int x, int y, int pixelNum) {

		Mat RectDepth, RectRGB, RectCloud;
		//if (int(x - 1.5*pixelNum - 1) < 0 || int(x + 1.5*pixelNum + 1) > 640 || int(y + 3.3*pixelNum + 1) > 480 || int(y - 0.7*pixelNum - 1) < 0)
		//{
		//	return false;
		//}
		//设定输出参数
		
		RectDepth = Mat(4 * pixelNum + 2, 3 * pixelNum + 2, CV_16U);
		RectRGB = Mat(4 * pixelNum + 2, 3 * pixelNum + 2, CV_8UC3);
		Mat tempDepth;
		Mat tempRGB;
		depthFrame.copyTo(tempDepth);
		rgbFrame.copyTo(tempRGB);
		MatHelper::GetRectDepthMat(tempDepth, RectDepth, int(x - 1.5*pixelNum - 1), int(y - 0.7*pixelNum - 1), 3 * pixelNum + 2, 4 * pixelNum + 2);//width height 
		MatHelper::GetRectMat(tempRGB, RectRGB, int(x - 1.5*pixelNum - 1), int(y - 0.7*pixelNum - 1), 3 * pixelNum + 2, 4 * pixelNum + 2);//width height 
																																		  //获取点云RECT
		RectCloud = Mat(4 * pixelNum + 2, 3 * pixelNum + 2, CV_32FC3);
		Mat tempCloud;
		pointCloud.copyTo(tempCloud);
		MatHelper::GetRectMatF(tempCloud, RectCloud, int(x - 1.5*pixelNum - 1), int(y - 0.7*pixelNum - 1), 3 * pixelNum + 2, 4 * pixelNum + 2);//width height 
																																			   //确定头顶点位置
		Point2i rectHeadPoint = Point2i(x - int(x - 1.5*pixelNum - 1), y - int(y - 0.7*pixelNum - 1));
		RectDepth.at<unsigned short>(0, 0) = rectHeadPoint.x;
		RectDepth.at<unsigned short>(0, 1) = rectHeadPoint.y;
		//cout << RectDepth.at<unsigned short>(0, 0) << " " << RectDepth.at<unsigned short>(0, 1) << endl;
		return rgbd2vec::rgb2vector(RectDepth, RectRGB, RectCloud);
		

	}


	
};