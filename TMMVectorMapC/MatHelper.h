#pragma once
#ifndef MAT_HELPER_H
#define MAT_HELPER_H

#include <fstream>
#include "headfile.h"

class MatHelper
{
public:
	// ��ӡһ�������ļ���;
	static void PrintMat(const std::string sMatType, const cv::Mat& mat, const string fileName)
	{
		ofstream fout(fileName.c_str());

		for (int i = 0; i < mat.rows; i++)
		{
			for (int j = 0; j < mat.cols; j++)
			{
				if (sMatType == "32SC1")
				{
					fout << mat.at<long>(i, j) << "\t";
				}
				else if (sMatType == "16UC1")
				{
					fout << mat.at<ushort>(i, j) << "\t";
				}
				else if (sMatType == "8UC1")
				{
					fout << mat.at<uchar>(i, j) << "\t";
				}
				else if (sMatType == "32F")
				{
					fout << mat.at<float>(i, j) << "\t";
				}
				else if (sMatType == "64F")
				{
					fout << mat.at<double>(i, j) << "\t";
				}

			}
			fout << "\n";
		}

		fout.close();
	}

	// ���ļ��ж�ȡһ������;
	static void ReadMat(const std::string sMatType, cv::Mat& mat, const string fileName)
	{
		ifstream fin(fileName.c_str());

		int index = 0;

		while (!fin.eof() && index < mat.rows*mat.cols)
		{
			if (sMatType == "32SC1")
			{
				fin >> mat.at<long>(index / mat.cols, index%mat.cols);
			}
			else if (sMatType == "16UC1")
			{
				fin >> mat.at<ushort>(index / mat.cols, index%mat.cols);
			}
			else if (sMatType == "8UC1")
			{
				fin >> mat.at<uchar>(index / mat.cols, index%mat.cols);
			}
			else if (sMatType == "32F")
			{
				fin >> mat.at<float>(index / mat.cols, index%mat.cols);
			}
			else if (sMatType == "64F")
			{
				fin >> mat.at<double>(index / mat.cols, index%mat.cols);
			}

			index++;
		}

		fin.close();
	}

	// Ϊ�˱����ڽ���ֱ����������ʾ����8UC1��Matת��Ϊ8UC3��Mat��ת������Ϊ��3��ͨ����ֵȫ��д��ͬһ��ԭʼֵ;
	static void ImageMat8UC1To8UC3(const cv::Mat& inMat_8UC1, cv::Mat& outMat_8UC3)
	{
		for (int i = 0; i < inMat_8UC1.rows; i++)
		{
			for (int j = 0; j < inMat_8UC1.cols; j++)
			{
				outMat_8UC3.at<cv::Vec3b>(i, j)[0] = inMat_8UC1.at<uchar>(i, j);
				outMat_8UC3.at<cv::Vec3b>(i, j)[1] = inMat_8UC1.at<uchar>(i, j);
				outMat_8UC3.at<cv::Vec3b>(i, j)[2] = inMat_8UC1.at<uchar>(i, j);
			}
		}
	}


	// �Ƚ�������ͬ�ߴ�Ĺ�һ���ľ��󣬻�����ƶȣ�0~1֮�䣩;
	static float GetMatSimilarity(const cv::Mat& mat1_32F, const cv::Mat& mat2_32F) // mat1_32F��mat2_32F�Ѿ���һ��
	{
		double dDistance = 0;

		float fSimilarity = 0;

		for (int i = 0; i < mat1_32F.rows; i++)
		{
			for (int j = 0; j < mat1_32F.cols; j++)
			{
				dDistance += sqrt(mat1_32F.at<float>(i, j) * mat2_32F.at<float>(i, j));
			}
		}

		dDistance = 1 - dDistance;
		fSimilarity = exp((-30.0) * dDistance * dDistance);

		return fSimilarity;
	}


	//��Mat��ȡһ��СMat,��ɫͼ��;
	static void GetRectMat(cv::Mat& img, cv::Mat& Rectimg, int x, int y, int width, int height)
	{

		for (int j = 0; j < height; j++)
		{
			for (int i = 0; i < width; i++)
			{
				if ((y + j >= img.rows) || (y + j <= 0) || (x + i >= img.cols) || (x + i <= 0)) {
					Rectimg.at<cv::Vec3b>(j, i)[0] = 1;
					Rectimg.at<cv::Vec3b>(j, i)[1] = 1;
					Rectimg.at<cv::Vec3b>(j, i)[2] = 1;
				}
				else {
					Rectimg.at<cv::Vec3b>(j, i)[0] = img.at<cv::Vec3b>(y + j, x + i)[0];
					Rectimg.at<cv::Vec3b>(j, i)[1] = img.at<cv::Vec3b>(y + j, x + i)[1];
					Rectimg.at<cv::Vec3b>(j, i)[2] = img.at<cv::Vec3b>(y + j, x + i)[2];
				}
			}
		}

	}
	static void GetRectMatF(cv::Mat& img, cv::Mat& Rectimg, int x, int y, int width, int height)
	{

		for (int j = 0; j < height; j++)
		{
			for (int i = 0; i < width; i++)
			{
				if ((y + j >= img.rows) || (y + j <= 0) || (x + i >= img.cols) || (x + i <= 0)) {
					Rectimg.at<cv::Vec3b>(j, i)[0] = 0;
					Rectimg.at<cv::Vec3b>(j, i)[1] = 0;
					Rectimg.at<cv::Vec3b>(j, i)[2] = 0;
				}
				else {
					Rectimg.at<cv::Vec3f>(j, i)[0] = img.at<cv::Vec3f>(y + j, x + i)[0];
					Rectimg.at<cv::Vec3f>(j, i)[1] = img.at<cv::Vec3f>(y + j, x + i)[1];
					Rectimg.at<cv::Vec3f>(j, i)[2] = img.at<cv::Vec3f>(y + j, x + i)[2];
				}
				
			}
		}

	}
	//��Mat��ȡһ��СMat,���ͼ;
	static void GetRectDepthMat(cv::Mat& img, cv::Mat& Rectimg, int x, int y, int width, int height)
	{
		for (int j = 0; j < height; j++)
		{
			for (int i = 0; i < width; i++)
			{
				if (y + j >= img.rows || y + j < 0 || x + i >= img.cols || x + i < 0) {
					Rectimg.at<unsigned short>(j, i) = 0;
				}
				else {
					Rectimg.at<unsigned short>(j, i) = img.at<unsigned short>(y + j, x + i);
				}
			}
		}

	}
	static void GetRectDepthMatB(cv::Mat& img, cv::Mat& Rectimg, int x, int y, int width, int height)
	{
		for (int j = 0; j < height; j++)
		{
			for (int i = 0; i < width; i++)
			{
				Rectimg.at<unsigned char>(j, i) = img.at<unsigned char>(y + j, x + i);
			}
		}

	}

	//Mat�ӿ�;
	static void GetedgeDepthMat(cv::Mat& img, cv::Mat& Edgeimg)
	{
		for (int j = 1; j < img.rows + 1; j++)
		{
			for (int i = 1; i < img.cols + 1; i++)
			{
				Edgeimg.at<unsigned short>(j, i) = img.at<unsigned short>(j - 1, i - 1);
			}
		}

		for (int i = 0; i < img.cols + 2; i++)//��+��;
		{
			Edgeimg.at<unsigned short>(0, i) = Edgeimg.at<unsigned short>(1, i);
			Edgeimg.at<unsigned short>(img.rows + 1, i) = Edgeimg.at<unsigned short>(img.rows, i);
		}

		for (int j = 1; j < img.rows + 1; j++)//��+��;
		{
			Edgeimg.at<unsigned short>(j, 0) = Edgeimg.at<unsigned short>(j, 1);
			Edgeimg.at<unsigned short>(j, img.cols + 1) = Edgeimg.at<unsigned short>(j, img.cols);
		}

	}


	static void ResizedDepthMat(cv::Mat& img1, cv::Mat& img2, int width1, int height1, int width2, int height2)//�����ͼ�е����ֵ���ŵ��̶��ĳߴ�;
	{
		int j1 = 0, i1 = 0, j2, i2;
		float multiwidth, multiheight;


		for (j2 = 0; j2 < height2; j2++)
		{
			for (i2 = 0; i2 < width2; i2++)
			{
				multiwidth = (float)width1 / width2;
				multiheight = (float)height1 / height2;
				j1 = multiheight * j2;
				i1 = multiwidth * i2;

				img2.at<unsigned short>(j2, i2) = img1.at<unsigned short>(j1, i1);
			}
		}
	}
	static float GetDist(cv::Mat &img, Point2i p1, Point2i p2) {		//���������	���룺poing2i(i,j);
		float dist = 0;
		float x1 = img.at<cv::Vec3f>(p1.y, p1.x)[0];
		float y1 = img.at<cv::Vec3f>(p1.y, p1.x)[1];
		float z1 = img.at<cv::Vec3f>(p1.y, p1.x)[2];
		float x2 = img.at<cv::Vec3f>(p2.y, p2.x)[0];
		float y2 = img.at<cv::Vec3f>(p2.y, p2.x)[1];
		float z2 = img.at<cv::Vec3f>(p2.y, p2.x)[2];
		dist = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2) *(z1 - z2));
		return dist;
	}
	//static void GetCurvature(cv::Mat &Cloud, cv::Mat &result, Point2i headPoint){
	//	float dist = 0;
	//	for (int j = 1; j < Cloud.rows; j++){
	//		for (int i = 1; i < Cloud.cols; i++)
	//		{
	//			dist = MatHelper::GetDist(Cloud, Point2i(i, j), headPoint);
	//			
	//			if (dist < 400){
	//				if (GetDist(Cloud, Point2i(i, j), Point2i(i - 1, j)) < 80 && GetDist(Cloud, Point2i(i, j), Point2i(i - 1, j - 1)) < 80 && GetDist(Cloud, Point2i(i, j), Point2i(i, j - 1)) < 80)
	//				{
	//					
	//				}
	//			}
	//		}
	//	}
	//}
};

#endif