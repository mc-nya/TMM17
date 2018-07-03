#pragma once
#include"headfile.h"
#include<io.h>
#include"MyImageLoader.h"
#include"C2Vec.h"
#include"UtilFeatureAccumulator.h"


class rgbd2vec {
public:
	// convert small head rect to vector
	static Mat rgb2vector(Mat imDepth,Mat imRGB,Mat pointCloud ) {
		

		Mat imRGBGray;
		Mat imDepth8;
		imDepth.convertTo(imDepth8, CV_8U,255.0/8000.0);
		cvtColor(imRGB, imRGBGray, CV_RGB2GRAY);
		Mat imgGra = ForExp::getGraB(imDepth8);
		Mat imgGra2 = ForExp::getGraF(imgGra);
		//Mat RGBGra = ForExp::getGraB(imRGBGray);
		//Mat RGBGra2 = ForExp::getGraF(RGBGra);

		// build feature map for RGB image, different gradient orders
		// NOT USED IN PAPER
		//************************************************************************************************************************
		//vector<Mat> RGBGrayMap;
		UtilFeatureAccumulator util;
		Mat temp;
		int count;
		int flag[3][3][3][3];

		//memset(flag, 0, sizeof(flag));

		//count = 0;

		//for (int i = -1; i < 2; i++) {
		//	for (int j = -1; j < 2; j++) {
		//		for (int ii = -1; ii < 2; ii++) {
		//			for (int jj = -1; jj < 2; jj++) {
		//				int dis = abs(i - ii) + abs(j - jj);
		//				if (dis < 2) continue;
		//				if ((i == 0 && j == 0)) continue;
		//				if ((ii == 0 && jj == 0))continue;
		//				if ((i == ii) && (j == jj)) continue;
		//				if (flag[i + 1][j + 1][ii + 1][jj + 1] == 1) continue;
		//				flag[i + 1][j + 1][ii + 1][jj + 1] = 1;
		//				flag[ii + 1][jj + 1][i + 1][j + 1] = 1;
		//				count++;
		//				if (!(count == 1 || count == 3 || count == 6 || count == 7 || count == 14 || count == 16 || count == 19 || count == 20)) continue;
		//				Mat temp = util.hotMapBuilder(imRGBGray, i, j, ii, jj);
		//				RGBGrayMap.push_back(temp);
		//				//cvWaitKey(2);
		//			}
		//		}
		//	}
		//}

		//memset(flag, 0, sizeof(flag));

		//count = 0;
		//for (int i = -1; i < 2; i++) {
		//	for (int j = -1; j < 2; j++) {
		//		for (int ii = -1; ii < 2; ii++) {
		//			for (int jj = -1; jj < 2; jj++) {
		//				int dis = abs(i - ii) + abs(j - jj);
		//				if (dis < 2) continue;
		//				if ((i == 0 && j == 0)) continue;
		//				if ((ii == 0 && jj == 0))continue;
		//				if ((i == ii) && (j == jj)) continue;
		//				if (flag[i + 1][j + 1][ii + 1][jj + 1] == 1) continue;
		//				flag[i + 1][j + 1][ii + 1][jj + 1] = 1;
		//				flag[ii + 1][jj + 1][i + 1][j + 1] = 1;
		//				count++;
		//				if (!(count == 1 || count == 3 || count == 6 || count == 7 || count == 14 || count == 16 || count == 19 || count == 20)) continue;
		//				Mat temp = util.hotMapBuilderGround(RGBGra, i, j, ii, jj);
		//				RGBGrayMap.push_back(temp);
		//				//cvWaitKey(2);
		//			}
		//		}
		//	}
		//}


		//memset(flag, 0, sizeof(flag));


		//count = 0;
		//for (int i = -1; i < 2; i++) {
		//	for (int j = -1; j < 2; j++) {
		//		for (int ii = -1; ii < 2; ii++) {
		//			for (int jj = -1; jj < 2; jj++) {
		//				int dis = abs(i - ii) + abs(j - jj);
		//				if (dis < 2) continue;
		//				if ((i == 0 && j == 0)) continue;
		//				if ((ii == 0 && jj == 0))continue;
		//				if ((i == ii) && (j == jj)) continue;
		//				if (flag[i + 1][j + 1][ii + 1][jj + 1] == 1) continue;
		//				flag[i + 1][j + 1][ii + 1][jj + 1] = 1;
		//				flag[ii + 1][jj + 1][i + 1][j + 1] = 1;
		//				count++;
		//				if (!(count == 1 || count == 3 || count == 6 || count == 7 || count == 14 || count == 16 || count == 19 || count == 20)) continue;
		//				Mat temp = util.hotMapBuilderGround(RGBGra2, i, j, ii, jj);
		//				RGBGrayMap.push_back(temp);
		//				//cvWaitKey(2);
		//			}
		//		}
		//	}
		//}

		//Mat RGBGrayAcc = util.hotMapAccumulator(RGBGrayMap);


		// build feature map for depth image, different gradient orders
		// CODE BELOW USES ZERO AND FIRST ORDER
		//************************************************************************************************************************
		vector<Mat> DepthMap;
		memset(flag, 0, sizeof(flag));
		count = 0;

		for (int i = -1; i < 2; i++) {
			for (int j = -1; j < 2; j++) {
				for (int ii = -1; ii < 2; ii++) {
					for (int jj = -1; jj < 2; jj++) {
						int dis = abs(i - ii) + abs(j - jj);
						if (dis < 2) continue;
						if ((i == 0 && j == 0)) continue;
						if ((ii == 0 && jj == 0))continue;
						if ((i == ii) && (j == jj)) continue;
						if (flag[i + 1][j + 1][ii + 1][jj + 1] == 1) continue;
						flag[i + 1][j + 1][ii + 1][jj + 1] = 1;
						flag[ii + 1][jj + 1][i + 1][j + 1] = 1;
						count++;
						if (!(count == 1 || count == 3 || count == 6 || count == 7 || count == 14 || count == 16 || count == 19 || count == 20)) continue;
						Mat temp = util.hotMapBuilder(imDepth, i, j, ii, jj);
						DepthMap.push_back(temp);
						//cvWaitKey(2);
					}
				}
			}
		}
		memset(flag, 0, sizeof(flag));
		count = 0;

		//imshow("gra1", imgGra);
		for (int i = -1; i < 2; i++) {
			for (int j = -1; j < 2; j++) {
				for (int ii = -1; ii < 2; ii++) {
					for (int jj = -1; jj < 2; jj++) {
						int dis = abs(i - ii) + abs(j - jj);
						if (dis < 2) continue;
						if ((i == 0 && j == 0)) continue;
						if ((ii == 0 && jj == 0))continue;
						if ((i == ii) && (j == jj)) continue;
						if (flag[i + 1][j + 1][ii + 1][jj + 1] == 1) continue;
						flag[i + 1][j + 1][ii + 1][jj + 1] = 1;
						flag[ii + 1][jj + 1][i + 1][j + 1] = 1;
						count++;
						if (!(count == 1 || count == 3 || count == 6 || count == 7 || count == 14 || count == 16 || count == 19 || count == 20)) continue;
						Mat temp = util.hotMapBuilder(imgGra, i, j, ii, jj);
						DepthMap.push_back(temp);
						//cvWaitKey(2);
					}
				}
			}
		}
		//memset(flag, 0, sizeof(flag));
		//count = 0;

		//for (int i = -1; i < 2; i++) {
		//	for (int j = -1; j < 2; j++) {
		//		for (int ii = -1; ii < 2; ii++) {
		//			for (int jj = -1; jj < 2; jj++) {
		//				int dis = abs(i - ii) + abs(j - jj);
		//				if (dis < 2) continue;
		//				if ((i == 0 && j == 0)) continue;
		//				if ((ii == 0 && jj == 0))continue;
		//				if ((i == ii) && (j == jj)) continue;
		//				if (flag[i + 1][j + 1][ii + 1][jj + 1] == 1) continue;
		//				flag[i + 1][j + 1][ii + 1][jj + 1] = 1;
		//				flag[ii + 1][jj + 1][i + 1][j + 1] = 1;
		//				count++;
		//				if (!(count == 1 || count == 3 || count == 6 || count == 7 || count == 14 || count == 16 || count == 19 || count == 20)) continue;
		//				Mat temp = util.hotMapBuilder(imgGra2, i, j, ii, jj);
		//				DepthMap.push_back(temp);
		//				//cvWaitKey(2);
		//			}
		//		}
		//	}
		//}
		Mat DepthAcc = util.hotMapAccumulator(DepthMap);


		//************************************************************************************************************************

		//************************************************************************************************************************



		Mat result = imRGB.clone();

		int hx = imDepth.at<ushort>(0, 0);
		int hy = imDepth.at<ushort>(0, 1);

		C2Vec::getHeight(pointCloud, result, hx, hy);

		vector<Mat> channels;
		split(result, channels);
		channels[0] = imDepth8.clone();
		//channels[1] = imDepth8;
		/*{
			int statmax = -1;
			for (int i = 1; i < RGBGrayAcc.rows; i++) {
				for (int j = 1; j < RGBGrayAcc.cols; j++) {
					if (RGBGrayAcc.at<ushort>(i, j)>statmax)
						statmax = RGBGrayAcc.at<ushort>(i, j);
				}
			}
			RGBGrayAcc.convertTo(channels[1], CV_8U, 255.0 / (float)statmax);
		}*/
		
		// scaling to 0~1
		{
			int statmax = -1;
			for (int i = 1; i < DepthAcc.rows; i++) {
				for (int j = 1; j < DepthAcc.cols; j++) {
					if (DepthAcc.at<ushort>(i, j)>statmax)
						statmax = DepthAcc.at<ushort>(i, j);
				}
			}
			DepthAcc.convertTo(channels[1], CV_8U, 255.0 / (float)statmax);
		}

		
		merge(channels, result);
		//if (result.rows > 80) {
		//	imshow("0", channels.at(0));
		//	imshow("1", channels.at(1));
		//	imshow("2", channels.at(2));
		//	imshow("result", result);
		//	cvWaitKey();
		//}
		
		return result;
	}
};