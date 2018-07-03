#pragma once
#include "headfile.h"
#include"MatHelper.h"
static class UtilFeatureAccumulator
{
public:
	static Mat hotDectector() {}

	static Mat hotMapBuilder(Mat &depthMap,int dx1,int dy1,int dx2,int dy2) {
		Mat featureMap(depthMap.rows, depthMap.cols,CV_16U,Scalar::all(0));
		//featureMap()
		//imshow("test", featureMap);
		if (depthMap.depth() == CV_32FC1) {
			for (int i = 1; i < depthMap.rows - 1; i++) {
				for (int j = 1; j < depthMap.cols - 1; j++) {
					if (depthMap.at<float>(i, j)>depthMap.at<float>(i + dx1, j + dy1) + 0.5 && depthMap.at<float>(i, j)>depthMap.at<float>(i + dx2, j + dy2) + 0.5) {
						featureMap.at<unsigned short>(i, j) = 65535;
					}
				}
			}
		}
		if (depthMap.depth() == CV_16U)
		for (int i = 1; i < depthMap.rows-1; i++) {
			for (int j = 1; j < depthMap.cols-1; j++){
				if ((int)depthMap.at<unsigned short>(i, j)>(int)depthMap.at<unsigned short>(i + dx1, j + dy1)+ 200 && (int)depthMap.at<unsigned short>(i, j)>(int)depthMap.at<unsigned short>(i + dx2, j + dy2) + 200) {
					featureMap.at<unsigned short>(i, j) = 65535;
				}
			}
		}
		if (depthMap.depth() == CV_8UC1) {
			for (int i = 1; i < depthMap.rows - 1; i++) {
				for (int j = 1; j < depthMap.cols - 1; j++) {
					if ((int)depthMap.at<char>(i, j)>(int)depthMap.at<char>(i + dx1, j + dy1) + 5 && (int)depthMap.at<char>(i, j)>(int)depthMap.at<char>(i + dx2, j + dy2) + 5) {
						featureMap.at<unsigned short>(i, j) = 65535;
					}
				}
			}
		}
		return featureMap;
	}




	static Mat hotMapBuilderGround(Mat &depthMap, int dx1, int dy1, int dx2, int dy2) {
		Mat featureMap(depthMap.rows, depthMap.cols, CV_16U, Scalar::all(0));
		//featureMap()
		//imshow("test", featureMap);
		if (depthMap.depth() == CV_32FC1) {
			for (int i = 1; i < depthMap.rows - 1; i++) {
				for (int j = 1; j < depthMap.cols - 1; j++) {
					if (depthMap.at<float>(i, j)>depthMap.at<float>(i + dx1, j + dy1)+5 && depthMap.at<float>(i, j)>depthMap.at<float>(i + dx2, j + dy2)+5) {
						featureMap.at<unsigned short>(i, j) = 65535;
					}
				}
			}
		}
		if (depthMap.depth() == CV_16U)
			for (int i = 1; i < depthMap.rows - 1; i++) {
				for (int j = 1; j < depthMap.cols - 1; j++) {
					if ((int)depthMap.at<unsigned short>(i, j)>(int)depthMap.at<unsigned short>(i + dx1, j + dy1)  && (int)depthMap.at<unsigned short>(i, j)>(int)depthMap.at<unsigned short>(i + dx2, j + dy2)) {
						featureMap.at<unsigned short>(i, j) = 65535;
					}
				}
			}
		if (depthMap.depth() == CV_8UC1) {
			for (int i = 1; i < depthMap.rows - 1; i++) {
				for (int j = 1; j < depthMap.cols - 1; j++) {
					if ((int)depthMap.at<char>(i, j)>(int)depthMap.at<char>(i + dx1, j + dy1) + 3 && (int)depthMap.at<char>(i, j)>(int)depthMap.at<char>(i + dx2, j + dy2) + 3) {
						featureMap.at<unsigned short>(i, j) = 65535;
					}
				}
			}
		}
		return featureMap;
	}



	static Mat hotMapAccumulator(vector<Mat> &mapSet) {
		Mat featureMap(mapSet[0].rows, mapSet[0].cols, CV_16U, Scalar::all(1));
		int dis = 64000 / mapSet.size();
		for (Mat depthMap : mapSet) {
			for (int i = 1; i < depthMap.rows; i++) {
				for (int j = 1; j < depthMap.cols; j++) {
					if (depthMap.at<unsigned short>(i, j) >5) {
						featureMap.at<unsigned short>(i, j)=featureMap.at<unsigned short>(i, j)>64000? featureMap.at<unsigned short>(i, j): featureMap.at<unsigned short>(i, j) + dis;
					}
				}
			}
		}
		return featureMap;
	}
	static Mat hotIgnoreUnderMean(Mat &imgDepth,int width,int height) {
		vector<vector<int64>> sum;
		sum.resize(imgDepth.rows);
		Mat ret = imgDepth.clone();
		for (int i = 0; i < imgDepth.rows;i++) {
			sum[i].resize(imgDepth.cols);
		}
		for (int i = 1; i < imgDepth.rows; i++) {
			for (int j = 1; j < imgDepth.cols; j++) {
				sum[i][j] = sum[i - 1][j] + sum[i][j - 1] - sum[i - 1][j - 1] + (int)imgDepth.at<ushort>(i, j);
			}
		}
		double mean = ((double)sum[imgDepth.rows - 1][imgDepth.cols - 1]) / ((double)(imgDepth.rows - 1)*(imgDepth.cols - 1));

		for (int i = height+1; i < imgDepth.rows; i++) {
			for (int j = width+1; j < imgDepth.cols; j++) {
				double block = sum[i][j] - sum[i - height][j] - sum[i][j - width] + sum[i - height][j - width];
				double blockmean = block / ((double)(height*width));
				for (int x = i - height + 1; x <= i; x++) {
					for (int y = i - width + 1; y <= i; y++) {
						if (blockmean < mean) {
							ret.at<ushort>(i, j) = 0;
						}
						else {
							ret.at<ushort>(i, j) = imgDepth.at<ushort>(i,j);
						}
					}
				}
				//sum[i][j] = sum[i - 1][j] + sum[i][j - 1] - sum[i - 1][j - 1] + (int)imgDepth.at<ushort>(i, j);
			}
		}
		return ret;
	}

private:

};
