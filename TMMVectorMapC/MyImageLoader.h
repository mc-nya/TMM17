#ifndef MY_LOADER
#define MY_LOADER
#include"headfile.h"
#include<io.h>
#pragma once
class MyImageLoader{
protected:
	string path = "";		//���ݼ���·��
	string preDepthFilename, preColorFilename,preCloudFilename;		//��ȼ���ɫͼƬ��ǰ׺
	string sufDepthFilename, sufColorFilename,sufCloudFilename;		//��ȼ���ɫͼƬ����׺
	int count;				//�ļ���ʾΪpath+ǰ׺+count+��׺
	int maxcount;
	Mat depthImage;
	Mat colorImage;		//��ȼ���ɫͼ��
	Mat pointCloud;
	int existFlag;
	int isCloud;
public:
	/*���캯�� ��������Ϊ�� string ���ݼ���·��
							string ����ļ���ǰ׺
							string ����ļ�����׺
							string ��ɫ�ļ���ǰ׺
							string ��ɫ�ļ�����׺
							int ��ʼ���
							int ������)	*/
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
	/*���캯�� ��������Ϊ�� string ���ݼ���·��
	string ����ļ���ǰ׺
	string ����ļ�����׺
	string ��ɫ�ļ���ǰ׺
	string ��ɫ�ļ�����׺
	string �����ļ���ǰ׺
	string �����ļ�����׺
	int ��ʼ���
	int ������)	*/
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


	//ȡ���õ���һ��ͼƬ	����ֵ=1 �ɹ�����	����0 ��������  
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
	//����ͼƬ ����1���سɹ�  0����ʧ�ܣ��ļ������ڣ���ų������ޣ�
	virtual ~MyImageLoader() {

	}
	virtual bool load(){		//����1���سɹ�
		
		if (count > maxcount){		//��������
			return 0;
		}
		
		string dfilename = path + preDepthFilename + to_string(count) + sufDepthFilename;
		string cfilename = path + preColorFilename + to_string(count) + sufColorFilename;	//�ļ�������
		existFlag = testExist(dfilename, cfilename);
		if (!existFlag){		//�ļ�������
			return 0;
		}
		depthImage.release();
		colorImage.release();
		depthImage = imread(dfilename, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
		colorImage = imread(cfilename, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
		if (isCloud) {
			string pfilename= path + preCloudFilename + to_string(count) + sufCloudFilename;
			existFlag = testExist(dfilename, cfilename,pfilename);
			if (!existFlag) {		//�ļ�������
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
	//�ж��ļ��Ƿ���ڣ�������ȺͲ�ɫ�ļ���������·��������1�����
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

	//�ж��ļ��Ƿ���ڣ�������ȺͲ�ɫ�ļ���������·��������1�����
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

	//��ȡ���mat
	Mat getDepthImage(){
		return depthImage;
	}
	//��ȡ��ɫͼmat
	Mat getColorImage(){
		return colorImage;
	}
	Mat getPointCloud() {
		if (isCloud) {
			return pointCloud;
		}
		return colorImage;
	}
	//��ȡ��ǰ���
	int getCount(){
		return count;
	}
};

#endif