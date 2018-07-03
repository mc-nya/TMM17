#pragma once
#include "MyImageLoader.h"
class DarkImLoader :public MyImageLoader {
public:
	DarkImLoader(string datapath, string preDFilename, string sufDFilename, string preCfilename, string sufCfilename, int initcount, int endcount)
		{
			count = initcount - 1;
			maxcount = endcount;
			path = datapath;
			preDepthFilename = preDFilename;
			preColorFilename = preCfilename;
			sufDepthFilename = sufDFilename;
			sufColorFilename = sufCfilename;
			isCloud = 0;
			next();
		
	};
	//加载图片 返回1加载成功  0加载失败（文件不存在，标号超出上限）

	virtual bool load(){		//返回1加载成功
		if (count > maxcount) {		//超出上限
			return 0;
		}
		//cout << count << endl;
		string s_num = to_string(count);
		//cout << s_num << endl;
		while (s_num.size() < 4) {
			s_num = "0" + s_num;
			
		}
		string dfilename = path + preDepthFilename + s_num + sufDepthFilename;
		string cfilename = path + preColorFilename + s_num + sufColorFilename;	//文件名生成
		
		existFlag = testExist(dfilename, cfilename);
		if (!existFlag) {		//文件不存在
			return 0;
		}
		depthImage.release();
		colorImage.release();
		depthImage = imread(dfilename, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
		colorImage = imread(cfilename, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
		if (isCloud) {
			string pfilename = path + preCloudFilename + s_num + sufCloudFilename;
			existFlag = testExist(dfilename, cfilename, pfilename);
			if (!existFlag) {		//文件不存在
				return 0;
			}
			pointCloud.release();
			pointCloud = Mat(depthImage.rows, depthImage.rows, CV_32FC3);
			fstream fin;
			fin.open(pfilename, ios::in);
			for (int i = 0; i < depthImage.rows; i++) {
				for (int j = 0; j < depthImage.cols; j++) {
					fin >> pointCloud.at<Vec3f>(i, j)[0] >> pointCloud.at<Vec3f>(i, j)[1] >> pointCloud.at<Vec3f>(i, j)[2];
				}
			}
			fin.close();
		}
		return 1;
	}
};