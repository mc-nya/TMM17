#ifndef MY_LOADER
#define MY_LOADER
#include"headfile.h"
#include<io.h>
#pragma once
class MyImageLoader{
protected:
	string path = "";		//数据集根路径
	string preDepthFilename, preColorFilename,preCloudFilename;		//深度及彩色图片名前缀
	string sufDepthFilename, sufColorFilename,sufCloudFilename;		//深度及彩色图片名后缀
	int count;				//文件表示为path+前缀+count+后缀
	int maxcount;
	Mat depthImage;
	Mat colorImage;		//深度及彩色图像
	Mat pointCloud;
	int existFlag;
	int isCloud;
public:
	/*构造函数 参数依次为（ string 数据集根路径
							string 深度文件名前缀
							string 深度文件名后缀
							string 彩色文件名前缀
							string 彩色文件名后缀
							int 初始序号
							int 最大序号)	*/
	MyImageLoader(string datapath,string preDFilename,string sufDFilename,string preCfilename,string sufCfilename,int initcount,int endcount){
		count = initcount-1;
		maxcount = endcount;
		path = datapath;
		preDepthFilename = preDFilename;
		preColorFilename = preCfilename;
		sufDepthFilename = sufDFilename;
		sufColorFilename = sufCfilename;
		isCloud = 0;
		next();
		
	}
	MyImageLoader() {


	}
	/*构造函数 参数依次为（ string 数据集根路径
	string 深度文件名前缀
	string 深度文件名后缀
	string 彩色文件名前缀
	string 彩色文件名后缀
	string 点云文件名前缀
	string 点云文件名后缀
	int 初始序号
	int 最大序号)	*/
	MyImageLoader(string datapath, string preDFilename, string sufDFilename, string preCfilename, string sufCfilename,string prePFilename,string sufPFilename, int initcount, int endcount) {
		count = initcount - 1;
		maxcount = endcount;
		path = datapath;
		preDepthFilename = preDFilename;
		preColorFilename = preCfilename;
		preCloudFilename = prePFilename;
		sufDepthFilename = sufDFilename;
		sufColorFilename = sufCfilename;
		sufCloudFilename = sufPFilename;
		isCloud = 1;
		next();

	}


	//取可用的下一张图片	返回值=1 成功加载	返回0 超出上限  
	virtual int next(){   
		count++;		
		while (load() == 0){
			
			if (count > maxcount){
				return 0;
			}
			count++;
		}	
		return 1;
	}
	//加载图片 返回1加载成功  0加载失败（文件不存在，标号超出上限）
	virtual ~MyImageLoader() {

	}
	virtual bool load(){		//返回1加载成功
		
		if (count > maxcount){		//超出上限
			return 0;
		}
		
		string dfilename = path + preDepthFilename + to_string(count) + sufDepthFilename;
		string cfilename = path + preColorFilename + to_string(count) + sufColorFilename;	//文件名生成
		existFlag = testExist(dfilename, cfilename);
		if (!existFlag){		//文件不存在
			return 0;
		}
		depthImage.release();
		colorImage.release();
		depthImage = imread(dfilename, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
		colorImage = imread(cfilename, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
		if (isCloud) {
			string pfilename= path + preCloudFilename + to_string(count) + sufCloudFilename;
			existFlag = testExist(dfilename, cfilename,pfilename);
			if (!existFlag) {		//文件不存在
				return 0;
			}
			pointCloud.release();
			pointCloud=Mat(depthImage.rows, depthImage.rows, CV_32FC3);
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
	//判断文件是否存在，输入深度和彩色文件名的完整路径，返回1则存在
	bool testExist(string dfilename, string cfilename){		
		char cfile1[200], cfile2[200];
		strncpy(cfile1, dfilename.c_str(), dfilename.length());
		strncpy(cfile2, cfilename.c_str(), cfilename.length());
		cfile1[dfilename.length()] = '\0';
		cfile2[cfilename.length()] = '\0';
		if (_access(cfile1, 0) == -1 || _access(cfile2, 0) == -1){
			return 0;
		}
		return 1;
	}

	//判断文件是否存在，输入深度和彩色文件名的完整路径，返回1则存在
	bool testExist(string dfilename, string cfilename,string pfilename) {
		char cfile1[200], cfile2[200], cfile3[200];
		strncpy(cfile1, dfilename.c_str(), dfilename.length());
		strncpy(cfile2, cfilename.c_str(), cfilename.length());
		strncpy(cfile3, pfilename.c_str(), pfilename.length());
		cfile1[dfilename.length()] = '\0';
		cfile2[cfilename.length()] = '\0';
		cfile3[pfilename.length()] = '\0';
		if (_access(cfile1, 0) == -1 || _access(cfile2, 0) == -1 || _access(cfile3, 0) == -1) {
			return 0;
		}
		return 1;
	}

	//获取深度mat
	Mat getDepthImage(){
		return depthImage;
	}
	//获取彩色图mat
	Mat getColorImage(){
		return colorImage;
	}
	Mat getPointCloud() {
		if (isCloud) {
			return pointCloud;
		}
		return colorImage;
	}
	//获取当前标号
	int getCount(){
		return count;
	}
};

#endif