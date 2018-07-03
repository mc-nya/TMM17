#include"headfile.h";
#include"Confirm.h"
#include<io.h>
#include"MyImageLoader.h"
float GetDist(cv::Mat &img, Point2i p1, Point2i p2){		//输出：毫米	输入：poing2i(i,j);
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
int GetSizeInImageBySizeIn3D(const int iSizeIn3D, const int iDistance)
{
	if (iDistance == 0 || iSizeIn3D == 0)
	{
		return 0;
	}

	static double dConstFactor = 0.0; // 常数，表示空间中？长度的距离相机为1毫米的直线将投影成一个像素宽
	static bool bIsFactorComputed = false; // 因为只想在第一次计算一次常数，以后不再计算，本变量保存计算状态

	if (!bIsFactorComputed)
	{
		// 在深度图上假定两点
		//pPoint3D[2] = { { 0, 0, 1000 }, { 100, 0, 1000 } }; // {列, 行, 深度}
		//float X1, X2, Y1, Y2, D1, D2;
		//float rX1, rX2, rY1, rY2, rZ1, rZ2;
		//X1 = 0; //1的列
		//Y1 = 0; //1的行
		//D1 = 1000;
		//X2 = 100; //2的列
		//Y2 = 0; //2的行
		//D2 = 1000;
		//g_openNi.m_DepthGenerator.ConvertProjectiveToRealWorld(2, pPoint3D, pPoint3D); // 可以得到点在空间中的实际坐标
		//CoordinateConverter::convertDepthToWorld(g_openNi.streamDepth, X1, Y1, D1, &rX1, &rY1, &rZ1);
		//CoordinateConverter::convertDepthToWorld(g_openNi.streamDepth, X2, Y2, D2, &rX2, &rY2, &rZ2);
		// 两点在空间中的实际距离
		double d3DDistance = 173.66668978426750;//std::sqrt((long double)((rX1 - rX2) * (rX1 - rX2) + (rY1 - rY2) * (rY1 - rY2) + (rZ1 - rZ2) * (rZ1 - rZ2)));
		dConstFactor = d3DDistance / (1000 * 100);
		bIsFactorComputed = true;
	}

	return (int)(((double)iSizeIn3D / (double)iDistance) / dConstFactor);

}

int quei[500000], quej[500000];
int seti[500000], setj[500000], setnum[500000];		//flood时使用队列que  集合set
int flag[480][640];		//集合标志
int h, t, setcount;		//队列标志

int findz(){

	MyImageLoader imLoader = MyImageLoader("E:\\dataset\\47\\", "depth\\depth_", ".png", "rgb\\rgb_", ".png", 34889, 53000);
	//ImageLoader imLoader = ImageLoader("E:\\实验室\\dataset\\38\\", "depth\\depth_", ".png", "rgb\\rgb_", ".png", 2030, 3000);
	//ImageLoader imLoader = ImageLoader("E:\\实验室\\dataset\\18D\\", "depth\\png\\1 (", ")_16UC1.png", "rgb\\1 (", ").png", 3600, 10401);
	//ImageLoader imLoader = ImageLoader("E:\\实验室\\dataset\\10D\\", "depth\\png\\1 (", ")_16UC1.png", "rgb\\1 (", ").png", 1, 700);
	//ImageLoader imLoader = ImageLoader("E:\\实验室\\dataset\\1\\", "depth\\png\\1 (", ")_8UC1_From1File.png", "rgb\\1 (", ").png", 1, 1999);
	//ImageLoader imLoader = ImageLoader("E:\\实验室\\dataset\\11\\", "depth\\png\\1 (", ")_16UC1.png", "rgb\\1 (", ").png", 240, 900);
	while (true) {
		Mat rawRGB = imLoader.getColorImage();
		Mat rawDepth = imLoader.getDepthImage();
		Mat properRGB = rawRGB.clone();
		Mat headCloud;			//生成点云
		headCloud = Mat(480, 640, CV_32FC3);
		double pixelLength = (240.0 / tan(21.5*PI / 180.0) + 320.0 / tan(28.5*PI / 180.0)) / 2.0;
		double l, ang, depth, x, y, z;
		for (int j = 1; j < rawDepth.rows; ++j) {
			for (int i = 1; i < rawDepth.cols; ++i) {
				if (rawDepth.at<unsigned short>(j, i)<800 || rawDepth.at<unsigned short>(j, i)>10000) {
					headCloud.at<cv::Vec3f>(j, i)[0] = 0;
					headCloud.at<cv::Vec3f>(j, i)[1] = 0;
					headCloud.at<cv::Vec3f>(j, i)[2] = 0;
					continue;
				}
				if (j>240 && i < 320) {
					int count = 0;
				}
				l = sqrt((j - 240)*(j - 240) + (i - 320) *(i - 320));
				ang = atan(l / pixelLength);
				depth = rawDepth.at<unsigned short>(j, i)*cos(ang);
				x = ((i - 320) / pixelLength)*depth;
				y = ((240 - j) / pixelLength)*depth;
				headCloud.at<cv::Vec3f>(j, i)[0] = x;
				headCloud.at<cv::Vec3f>(j, i)[1] = y;
				headCloud.at<cv::Vec3f>(j, i)[2] = depth;
			}
		}

		//for (int j = 0; j < 480; j++){
		//	for (int i = 0; i < 640; i++){
		//		flag[j][i] = 0;
		//	}
		//}
		memset(flag, 0, sizeof(flag));

		setcount = 0;
		for (int j = 0; j < rawDepth.rows; j++) {
			for (int i = 0; i < rawDepth.cols; i++) {
				if (flag[j][i] == 0 && rawDepth.at<unsigned short>(j, i)>800 && rawDepth.at<unsigned short>(j, i)<8000) {
					h = 1; t = 1;
					setcount++;
					quei[1] = i;	quej[1] = j;
					seti[setcount] = i; setj[setcount] = j;
					setnum[setcount] = 1;
					flag[j][i] = setcount;
					while (h <= t) {
						int tempi = quei[h];
						int tempj = quej[h];
						int pixelNum = GetSizeInImageBySizeIn3D(150, rawDepth.at<unsigned short>(j, i));

						pixelNum = 110;
						if (tempi>0 && flag[tempj][tempi - 1] == 0) {
							double dist = GetDist(headCloud, Point2i(tempi, tempj), Point2i(tempi - 1, tempj));
							if (dist < pixelNum) {
								t++;
								quei[t] = tempi - 1;
								quej[t] = tempj;
								flag[tempj][tempi - 1] = setcount;
								setnum[setcount]++;
							}
						}

						if (tempj<479 && flag[tempj + 1][tempi] == 0) {
							double dist = GetDist(headCloud, Point2i(tempi, tempj), Point2i(tempi, tempj + 1));
							if (dist < pixelNum) {
								t++;
								quei[t] = tempi;
								quej[t] = tempj + 1;
								flag[tempj + 1][tempi] = setcount;
								setnum[setcount]++;
							}
						}

						if (tempi<639 && flag[tempj][tempi + 1] == 0) {
							double dist = GetDist(headCloud, Point2i(tempi, tempj), Point2i(tempi + 1, tempj));
							if (dist < pixelNum) {
								t++;
								quei[t] = tempi + 1;
								quej[t] = tempj;
								flag[tempj][tempi + 1] = setcount;
								setnum[setcount]++;
							}
						}

						h++;

					}
				}
			}
		}
		Mat drawRGB = rawRGB.clone();
		int ccount = 0;
		int ncount = 0;
		for (int k = 1; k <= setcount; k++) {
			int i = seti[k];
			int j = setj[k];
			int pixelNum = GetSizeInImageBySizeIn3D(150, rawDepth.at<unsigned short>(j, i));
			if (j<70 || j>320)continue;
			if ((setnum[k] / (pixelNum ^ 2)  > 80) || setnum[k]>700) {

				Mat vec = Confirm::getVector(rawDepth, rawRGB, headCloud, i, j, pixelNum);
				imshow("vec", vec);
				cvWaitKey(100);
				cout << setnum[k] / pixelNum << endl;
				circle(drawRGB, Point(i, j), 3, Scalar(0, 69, 255), -1, 8);
				rectangle(drawRGB, cv::Rect(i - 1.5*pixelNum, j - 0.7*pixelNum, 3 * pixelNum, 4 * pixelNum), cv::Scalar(47, 230, 173), 5, 8);
				putText(drawRGB, to_string(ccount), cvPoint(i, j), CV_FONT_HERSHEY_DUPLEX, 0.5, CV_RGB(255, 255, 0));
			}
		}


		imshow("test1", drawRGB);
		if (!imLoader.next()) break;
	}
	
	

	return 0;
}