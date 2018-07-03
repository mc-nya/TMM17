#pragma once
#include "headfile.h"
#include"MatHelper.h"
#define pi 3.1415926535
class C2Vec {
public:
	static Mat getGra(const Mat &image) {
		Mat Gra = Mat(image.rows, image.cols, CV_32FC1);
		double gx, gy;
		for (int j = 1; j < image.rows - 1; j++) {
			for (int i = 1; i < image.cols - 1; i++) {
				gx = image.at<unsigned short>(j, i + 1) - image.at<unsigned short>(j, i - 1);
				gy = image.at<unsigned short>(j - 1, i) - image.at<unsigned short>(j + 1, i);
				Gra.at<float>(j, i) = sqrt(gx*gx + gy*gy);
				//cout << "gra" << endl;
			}
		}
		return Gra;
	}
	static Mat getGraB(const Mat &image) {
		Mat Gra = Mat(image.rows, image.cols, CV_32FC1);
		double gx, gy;
		for (int j = 1; j < image.rows - 1; j++) {
			for (int i = 1; i < image.cols - 1; i++) {
				gx = image.at<unsigned char>(j, i + 1) - image.at<unsigned char>(j, i - 1);
				gy = image.at<unsigned char>(j - 1, i) - image.at<unsigned char>(j + 1, i);
				Gra.at<float>(j, i) = sqrt(gx*gx + gy*gy);
				//cout << "gra" << endl;
			}
		}
		return Gra;
	}	
	static Mat getGraF(const Mat &image) {
		Mat Gra = Mat(image.rows, image.cols, CV_32FC1);
		double gx, gy;
		for (int j = 1; j < image.rows - 1; j++) {
			for (int i = 1; i < image.cols - 1; i++) {
				gx = image.at<float>(j, i + 1) - image.at<float>(j, i - 1);
				gy = image.at<float>(j - 1, i) - image.at<float>(j + 1, i);
				Gra.at<float>(j, i) = sqrt(gx*gx + gy*gy);
				//cout << "gra" << endl;
			}
		}
		return Gra;
	}
	static void computeBin(
		vector<float> &ret, 
		const Mat &rawDepth, 
		const Mat &depthGra, 
		const Mat &rgbGra, 
		const Mat &rgbGra2, 
		int binNum, 
		int y, int x, 
		int j1, int i1, 
		int j2, int i2,
		double &maxD,double &maxDG,double &maxIG,double &maxIG2,
		int &TmaxD,int &TmaxDG,int &TmaxIG,int &TmaxIG2) 
	{
		if (rawDepth.at<unsigned short>(y, x)>rawDepth.at<unsigned short>(y + j1, x + i1) && rawDepth.at<unsigned short>(y, x) > rawDepth.at<unsigned short>(y + j2, x + i2))
		{
			ret[binNum]++;
		}

		if (depthGra.at<float>(y, x) > depthGra.at<float>(y + j1, x + i1) && depthGra.at<float>(y, x) > depthGra.at<float>(y + j2, x + i2)) {
			ret[16 + binNum]++;
		}

		if (rgbGra.at<float>(y, x) > rgbGra.at<float>(y + j1, x + i1) && rgbGra.at<float>(y, x) > rgbGra.at<float>(y + j2, x + i2)) {
			ret[32 + binNum]++;
		}

		if (rgbGra2.at<float>(y, x) > rgbGra2.at<float>(y + j1, x + i1) && rgbGra2.at<float>(y, x) > rgbGra2.at<float>(y + j2, x + i2)) {
			ret[48 + binNum]++;
		}
		//double sumI = image.at<unsigned short>(y, x) + image.at<unsigned short>(y + j1, x + i1) + image.at<unsigned short>(y + j2, x + i2);
		double sumD = rawDepth.at<unsigned short>(y, x) + rawDepth.at<unsigned short>(y + j1, x + i1) + rawDepth.at<unsigned short>(y + j2, x + i2);
		double sumDG = depthGra.at<float>(y, x) + depthGra.at<float>(y + j1, x + i1) + depthGra.at<float>(y + j2, x + i2);
		double sumIG = rgbGra.at<float>(y, x) + rgbGra.at<float>(y + j1, x + i1) + rgbGra.at<float>(y + j2, x + i2);
		double sumIG2 = rgbGra2.at<float>(y, x) + rgbGra2.at<float>(y + j1, x + i1) + rgbGra2.at<float>(y + j2, x + i2);
		if (sumD >= maxD) {
			maxD = sumD;
			TmaxD = binNum;
		}
		if (sumDG >= maxDG) {
			maxDG = sumDG;
			TmaxDG = binNum;
		}
		if (sumIG >= maxIG) {
			maxIG = sumIG;
			TmaxIG = binNum;
		}
		if (sumIG2 >= maxIG2) {
			maxIG2 = sumIG2;
			TmaxIG2 = binNum;
		}
	}
	static void addArgMax(
		vector<float> &ret,
		const Mat &rawDepth,
		const Mat &depthGra,
		const Mat &rgbGra,
		const Mat &rgbGra2,
		int binNum,
		int y, int x,
		int j1, int i1,
		int j2, int i2,
		double &maxD, double &maxDG, double &maxIG, double &maxIG2,
		int &TmaxD, int &TmaxDG, int &TmaxIG, int &TmaxIG2)
	{
		//double sumI = image.at<unsigned short>(y, x) + image.at<unsigned short>(y + j1, x + i1) + image.at<unsigned short>(y + j2, x + i2);
		double sumD = rawDepth.at<unsigned short>(y, x) + rawDepth.at<unsigned short>(y + j1, x + i1) + rawDepth.at<unsigned short>(y + j2, x + i2);
		double sumDG = depthGra.at<float>(y, x) + depthGra.at<float>(y + j1, x + i1) + depthGra.at<float>(y + j2, x + i2);
		double sumIG = rgbGra.at<float>(y, x) + rgbGra.at<float>(y + j1, x + i1) + rgbGra.at<float>(y + j2, x + i2);
		double sumIG2 = rgbGra2.at<float>(y, x) + rgbGra2.at<float>(y + j1, x + i1) + rgbGra2.at<float>(y + j2, x + i2);
		if (sumD == maxD) {
			ret[binNum+8]++;
		}
		if (sumDG == maxDG) {
			ret[binNum+24]++;
		}
		if (sumIG == maxIG) {
			ret[binNum+40]++;
		}
		if (sumIG2 == maxIG2) {
			ret[binNum + 56]++;
		}
	}

	// block feature, collect bin features
	static vector<float> blockfeature(const Mat &rawDepth, const Mat &depthGra, const Mat &rgbGra, const Mat &rgbGra2, int startj, int starti, int height, int width) {
		vector<float> ret;
		ret.resize(64);
		double maxD, maxDG, maxIG, maxIG2;
		int TmaxD, TmaxDG, TmaxIG, TmaxIG2;
	
		for (int j = startj; j < startj + height; j++) {
			for (int i = starti; i < starti + width; i++) {
				maxD=-1, maxDG=-1, maxIG=-1, maxIG2=-1;
				TmaxD=-1, TmaxDG=-1, TmaxIG=-1, TmaxIG2=-1;
				//bin0
				computeBin(ret, rawDepth, depthGra, rgbGra, rgbGra2, 0, j, i, -1, -1, 1, 1, maxD, maxDG, maxIG, maxIG2, TmaxD, TmaxDG, TmaxIG, TmaxIG2);
				//bin1
				computeBin(ret, rawDepth, depthGra, rgbGra, rgbGra2, 1, j, i, -1, 1, 1, -1, maxD, maxDG, maxIG, maxIG2, TmaxD, TmaxDG, TmaxIG, TmaxIG2);
				//bin2			
				computeBin(ret, rawDepth, depthGra, rgbGra, rgbGra2, 2, j, i, 0, -1, 0, 1, maxD, maxDG, maxIG, maxIG2, TmaxD, TmaxDG, TmaxIG, TmaxIG2);
				//bin3			
				computeBin(ret, rawDepth, depthGra, rgbGra, rgbGra2, 3, j, i, -1, 0, 1, 0, maxD, maxDG, maxIG, maxIG2, TmaxD, TmaxDG, TmaxIG, TmaxIG2);
				//bin4			
				computeBin(ret, rawDepth, depthGra, rgbGra, rgbGra2, 4, j, i, 0, -1, -1, 0, maxD, maxDG, maxIG, maxIG2, TmaxD, TmaxDG, TmaxIG, TmaxIG2);
				//bin5			
				computeBin(ret, rawDepth, depthGra, rgbGra, rgbGra2, 5, j, i, -1, 0, 0, 1, maxD, maxDG, maxIG, maxIG2, TmaxD, TmaxDG, TmaxIG, TmaxIG2);
				//bin6			
				computeBin(ret, rawDepth, depthGra, rgbGra, rgbGra2, 6, j, i, 0, 1, 1, 0, maxD, maxDG, maxIG, maxIG2, TmaxD, TmaxDG, TmaxIG, TmaxIG2);
				//bin7			
				addArgMax(ret, rawDepth, depthGra, rgbGra, rgbGra2, 7, j, i, 1, 0, 0, -1, maxD, maxDG, maxIG, maxIG2, TmaxD, TmaxDG, TmaxIG, TmaxIG2);
				addArgMax(ret, rawDepth, depthGra, rgbGra, rgbGra2, 0, j, i, -1, -1, 1, 1, maxD, maxDG, maxIG, maxIG2, TmaxD, TmaxDG, TmaxIG, TmaxIG2);
				//bin1
				addArgMax(ret, rawDepth, depthGra, rgbGra, rgbGra2, 1, j, i, -1, 1, 1, -1, maxD, maxDG, maxIG, maxIG2, TmaxD, TmaxDG, TmaxIG, TmaxIG2);
				//bin2			
				addArgMax(ret, rawDepth, depthGra, rgbGra, rgbGra2, 2, j, i, 0, -1, 0, 1, maxD, maxDG, maxIG, maxIG2, TmaxD, TmaxDG, TmaxIG, TmaxIG2);
				//bin3			
				addArgMax(ret, rawDepth, depthGra, rgbGra, rgbGra2, 3, j, i, -1, 0, 1, 0, maxD, maxDG, maxIG, maxIG2, TmaxD, TmaxDG, TmaxIG, TmaxIG2);
				//bin4			
				addArgMax(ret, rawDepth, depthGra, rgbGra, rgbGra2, 4, j, i, 0, -1, -1, 0, maxD, maxDG, maxIG, maxIG2, TmaxD, TmaxDG, TmaxIG, TmaxIG2);
				//bin5			
				addArgMax(ret, rawDepth, depthGra, rgbGra, rgbGra2, 5, j, i, -1, 0, 0, 1, maxD, maxDG, maxIG, maxIG2, TmaxD, TmaxDG, TmaxIG, TmaxIG2);
				//bin6			
				addArgMax(ret, rawDepth, depthGra, rgbGra, rgbGra2, 6, j, i, 0, 1, 1, 0, maxD, maxDG, maxIG, maxIG2, TmaxD, TmaxDG, TmaxIG, TmaxIG2);
				//bin7			
				addArgMax(ret, rawDepth, depthGra, rgbGra, rgbGra2, 7, j, i, 1, 0, 0, -1, maxD, maxDG, maxIG, maxIG2, TmaxD, TmaxDG, TmaxIG, TmaxIG2);
				
			}
		}
		return ret;
	}
	
	// NOT USE
	static void getAngle(const Mat &rawDepth, Mat &ret) {
		//Mat gaussianDepth;
		//GaussianBlur(rawDepth, gaussianDepth, Size(5, 5), 0, 0);
		for (int i = 3; i < rawDepth.rows-3; i++) {
			for (int j = 3; j < rawDepth.cols - 3; j++) {
				//cout << "out  " << j << "  " << i << endl;
				signed short Gradient_X, Gradient_Y;
				if (rawDepth.depth() == CV_8U) {
					Gradient_X = 0.5*(rawDepth.at<char>(i, j + 1) - rawDepth.at<char>(i, j - 1));
					Gradient_Y = 0.5*(rawDepth.at<char>(i + 1, j) - rawDepth.at<char>(i - 1, j));
				}
				else {
					Gradient_X = 0.5*(rawDepth.at<unsigned short>(i, j + 1) - rawDepth.at<unsigned short>(i, j - 1));
					Gradient_Y = 0.5*(rawDepth.at<unsigned short>(i + 1, j) - rawDepth.at<unsigned short>(i - 1, j));
				}
				
				double theta1, theta2;
				if (Gradient_X != 0)
					theta1 = atan(Gradient_Y / Gradient_X) / pi * 180;
				else if (Gradient_Y > 0)
					theta1 = 90;
				else
					theta1 = -90;
				//cout << "out  " << sqrt(Gradient_Y*Gradient_Y + Gradient_X*Gradient_X) << endl;
				theta2 = atan(sqrt(Gradient_Y*Gradient_Y + Gradient_X*Gradient_X)) / pi * 180;
				//cout << "out  " << theta1 << "  " << theta2 << endl;
				if (theta1 < 0) {
					theta1 += 180;
				}
				if (theta2 < 0) {
					theta2 += 180;
				}
				
				//ret.at<Vec3b>(i, j)[2] = (char)((int)round(theta1) % 180);
				ret.at<Vec3b>(i, j)[2] = (char)((int)round(theta2) % 180);
				//ret.at<Vec3b>(i, j)[2] = (unsigned char)(theta1);
				//ret.at<Vec3b>(i, j)[2] = (unsigned char)(theta2);
			}
		}
	
	}

	// normal vector  NOT USE
	static Mat getNormalVector(const Mat &rawDepth) {
		Mat ret(rawDepth.rows, rawDepth.cols,CV_16SC3);
	
		for (int i = 10; i < rawDepth.rows - 10; i++) {
			for (int j = 10; j < rawDepth.cols - 10; j++) {
				//cout << "out  " << j << "  " << i << endl;
				signed short Gradient_X, Gradient_Y;
				Gradient_X = 0.05*(rawDepth.at<unsigned short>(i, j + 10) - rawDepth.at<unsigned short>(i, j - 10));
				Gradient_Y = 0.05*(rawDepth.at<unsigned short>(i + 10, j) - rawDepth.at<unsigned short>(i - 10, j));

				ret.at<Vec3s>(i, j)[0] = -Gradient_X;
				ret.at<Vec3s>(i, j)[1] = -Gradient_Y;
				ret.at<Vec3s>(i, j)[2] = 1;
				//ret.at<Vec3b>(i, j)[2] = (char)((int)round(theta1) % 180);
				//ret.at<Vec3b>(i, j)[2] = (char)((int)round(theta2) % 180);
				//ret.at<Vec3b>(i, j)[2] = (unsigned char)(theta1);
				//ret.at<Vec3b>(i, j)[2] = (unsigned char)(theta2);
			}
		}
		return ret;
	}

	static void getLab(const Mat &rawRGB, Mat &ret) {
		Mat lab = rawRGB.clone();
		cvtColor(rawRGB, lab, CV_BGR2Lab);
		for (int i = 2; i < rawRGB.rows - 2; i++) {
			for (int j = 2; j < rawRGB.cols - 2; j++) {
				ret.at<Vec3b>(i, j)[2] = lab.at<Vec3b>(i, j)[0];
			}
		}
	}
	
	//hot
	static void getHoTfeature(const Mat &rawRGB, const Mat &rawDepth,Mat &ret) {
		Mat grayImage;
		Mat depthGra, rgbGra, rgbGra2;
		cvtColor(rawRGB, grayImage, CV_RGB2GRAY);
		depthGra = getGra(rawDepth);
		rgbGra = getGraB(grayImage);
		rgbGra2 = getGraF(rgbGra);
		//imshow("test", rawRGB);
		//imshow("test2", rawDepth);
		//imshow("test3", rgbGra2);
		//cvWaitKey(200);
		vector<float>feature;
		vector<float>temp;
		feature.resize(1121);
		for (int j = 1; j < (rawRGB.rows)-2; j = j + 8) {
			for (int i = 1; i < (rawRGB.cols)-2; i = i + 8) {
				//cout << rawRGB.rows << "  " << rawRGB.cols << endl;
				//cout <<"out  "<< j << "  " << i << endl;
				if (rawRGB.rows - j <= 16 || rawRGB.cols - i <= 16) {
					temp = blockfeature(rawDepth, depthGra, rgbGra, rgbGra2, j, i, 8, 8);
					for (int kj = 0; kj < 8; kj++) {
						for (int ki = 0; ki < 8; ki++) {
							ret.at<Vec3b>(j + kj, i + ki)[0] = temp[kj * 8 + ki] >= 128 ? 255 : (temp[kj * 8 + ki] * 4);
						}
					}
				}
				else {
					temp = blockfeature(rawDepth, depthGra, rgbGra, rgbGra2, j, i, 16, 16);
					for (int kj = 0; kj < 8; kj++) {
						for (int ki = 0; ki < 8; ki++) {
							ret.at<Vec3b>(j + kj, i + ki)[0] = temp[kj * 8 + ki] >= 256 ? 255 : temp[kj * 8 + ki];
						}
					}
				}
				

				
			}
		}
		//return feature;
	}
	

	// height map
	static void getHeight(Mat &pointCloud, Mat &ret,int hx,int hy) {
		Mat temp(pointCloud.rows, pointCloud.cols, CV_32FC1,cvScalar(0.0));
		double cap=700;

		for (int i = 0; i < pointCloud.rows; i++) {
			for (int j = 0; j < pointCloud.cols; j++) {
				if (MatHelper::GetDist(pointCloud, Point2i(j, i), Point2i(hx, hy))>cap) continue;
				temp.at<float>(i, j) = (pointCloud.at<Vec3f>(hy, hx)[1] - pointCloud.at<Vec3f>(i, j)[1])/cap *255;
				if (temp.at<float>(i, j) < 0) {
					temp.at<float>(i, j) = 0;
				}
			}
		}
		for (int i = 0; i < ret.rows; i++) {
			for (int j = 0; j < ret.cols; j++) {
				ret.at<Vec3b>(i, j)[2] = temp.at<float>(i, j)>255?255:(char)temp.at<float>(i, j);
			}
		}
	}
};