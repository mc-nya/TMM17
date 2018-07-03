#include "caffe/caffe.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "boost/make_shared.hpp"
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "layeradd.h"

#include <time.h>
#include"Confirm.h"
#include "DarkImLoader.h"
#include "GTruthReader.h"
#include <thread>
#include <mutex>
std::timed_mutex mtx;
//#include"MyImageLoader.h"

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
public:
	Classifier(const string& model_file,
		const string& trained_file,
		const string& mean_file,
		const string& label_file);

	std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

	std::vector<float> Predict(const cv::Mat& img);

private:
	void SetMean(const string& mean_file);

	

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);

private:
	boost::shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
	std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file,
	const string& trained_file,
	const string& mean_file,
	const string& label_file) {
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif

	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	/* Load the binaryproto mean file. */
	SetMean(mean_file);

	/* Load labels. */
	std::ifstream labels(label_file.c_str());
	CHECK(labels) << "Unable to open labels file " << label_file;
	string line;
	while (std::getline(labels, line))
		labels_.push_back(string(line));

	Blob<float>* output_layer = net_->output_blobs()[0];
	CHECK_EQ(labels_.size(), output_layer->channels())
		<< "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
	const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], static_cast<int>(i)));
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
	std::vector<float> output = Predict(img);

	N = std::min<int>(labels_.size(), N);
	std::vector<int> maxN = Argmax(output, N);
	std::vector<Prediction> predictions;
	for (int i = 0; i < N; ++i) {
		int idx = maxN[i];
		predictions.push_back(std::make_pair(labels_[idx], output[idx]));
	}

	return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

	/* Convert from BlobProto to Blob<float> */
	Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	CHECK_EQ(mean_blob.channels(), num_channels_)
		<< "Number of channels of mean file doesn't match input layer.";

	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	std::vector<cv::Mat> channels;
	float* data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < num_channels_; ++i) {
		/* Extract an individual channel. */
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}

	/* Merge the separate channels into a single image. */
	cv::Mat mean;
	cv::merge(channels, mean);

	/* Compute the global mean pixel value and create a mean image
	* filled with this value. */
	cv::Scalar channel_mean = cv::mean(mean);
	mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	net_->Forward();

	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net_->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();
	return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Classifier::Preprocess(const cv::Mat& img,
	std::vector<cv::Mat>* input_channels) {
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);

	/* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the cv::Mat
	* objects in input_channels. */
	cv::split(sample_normalized, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}



float GetDist(cv::Mat &img, Point2i p1, Point2i p2) {		//输出：毫米	输入：poing2i(i,j);
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
int posCount = 0;


void th_test(const int &start, const int &end,const double &threshold, const Mat &rawDepth,const Mat &rawRGB,
	const Mat &headCloud,const boost::shared_ptr<Classifier> &clsfr_ptr,Point ret3[], Point ret5[], Point ret7[]) {
	for (int k = start; k < end; k++) {
		int i = seti[k];
		int j = setj[k];
		int pixelNum = GetSizeInImageBySizeIn3D(150, rawDepth.at<unsigned short>(j, i));
		if (setnum[k]>150) {
		//if (setnum[k]>1200000.0 / (double)rawDepth.at<unsigned short>(j, i)) {

			Mat vec = Confirm::getVector(rawDepth, rawRGB, headCloud, i, j, pixelNum);

				std::vector<float> output = clsfr_ptr->Predict(vec);

				//different threshold
				if (output[0] >  0.05)
				{
					Point temp;
					temp.x = i;
					temp.y = j;
					//cout << output[1] << endl;
					ret3[0].x += 1;
					ret3[ret3[0].x] = temp;

				}
				if (output[0]>0.1)
				{
					Point temp;
					temp.x = i;
					temp.y = j;
					//cout << output[1] << endl;
					ret5[0].x += 1;
					ret5[ret5[0].x] = temp;

				}
				if (output[0]>0.2)
				{
					Point temp;
					temp.x = i;
					temp.y = j;
					//cout << output[1] << endl;
					ret7[0].x += 1;
					ret7[ret7[0].x] = temp;

				}
			//}
			
			

			
		}
	}

}


int main(int argc, char** argv) {
	//string workpath = "E:\\CaffeNet\\TMM_3Channel_4.9\\";
	//string model_file = workpath + "deploy.prototxt";
	//string trained_file = workpath + "TMM_3Channel_4.9_caffenet_finetune_iter_2000.caffemodel";
	//string mean_file = workpath + "mean.binaryproto";
	//string label_file = workpath+"labels.txt";

	// path of caffmodels and mean
	string workpath = "E:\\CaffeNet\\1_pixelMean\\";
	string model_file = workpath + "deploy.prototxt";
	string trained_file = workpath + "caffenet_fientune_TitanX_total1000_1st__iter_3000.caffemodel";
	string mean_file = workpath + "mean.pt";
	string label_file = workpath + "labels.txt";

	::google::InitGoogleLogging(argv[0]);
	
	// init 4 classifier for 4 threads
	boost::shared_ptr<Classifier> clsfr_ptr1 = boost::make_shared<Classifier>(model_file, trained_file, mean_file, label_file);
	boost::shared_ptr<Classifier> clsfr_ptr2 = boost::make_shared<Classifier>(model_file, trained_file, mean_file, label_file);
	boost::shared_ptr<Classifier> clsfr_ptr3 = boost::make_shared<Classifier>(model_file, trained_file, mean_file, label_file);
	boost::shared_ptr<Classifier> clsfr_ptr4 = boost::make_shared<Classifier>(model_file, trained_file, mean_file, label_file);


	// init data reader
	//MyImageLoader imLoader = MyImageLoader("E:\\dataset\\47\\", "depth\\depth_", ".png", "rgb\\rgb_", ".png", 34700, 52200);
	//MyImageLoader imLoader = MyImageLoader("E:\\dataset\\38\\", "depth\\depth_", ".png", "rgb\\rgb_", ".png", 2000, 10400);
	//MyImageLoader imLoader = MyImageLoader("E:\\dataset\\18D\\", "depth\\png\\1 (", ")_16UC1.png", "rgb\\1 (", ").png", 1, 4485);
	//MyImageLoader imLoader = MyImageLoader("E:\\dataset\\10D\\", "depth\\png\\1 (", ")_16UC1.png", "rgb\\1 (", ").png", 1, 550);
	MyImageLoader imLoader = MyImageLoader("E:\\dataset\\1\\", "depth\\png\\1 (", ")_8UC1_From1File.png", "rgb\\1 (", ").png", 1, 1999);
	//MyImageLoader imLoader = MyImageLoader("E:\\dataset\\8\\", "depth\\png\\1 (", ")_8UC1_From8File.png", "rgb\\1 (", ").png", 400, 1439);
	//MyImageLoader imLoader = MyImageLoader("E:\\dataset\\11\\", "depth\\png\\1 (", ")_16UC1.png", "rgb\\1 (", ").png", 240, 1100);

	//DarkImLoader imLoader = DarkImLoader("E:\\dataset\\Outdoor\\dark\\56\\", "depth\\depth_", ".png", "rgb\\img_", ".png",0, 1439);
	//GTruthReader gReader = GTruthReader("E:\\dataset\\Outdoor\\dark\\56.txt");
	//DarkImLoader imLoader = DarkImLoader("E:\\dataset\\Outdoor\\dark\\31\\", "depth\\depth_", ".png", "rgb\\img_", ".png", 0, 1439);
	//GTruthReader gReader = GTruthReader("E:\\dataset\\Outdoor\\dark\\31.txt");
	//DarkImLoader imLoader = DarkImLoader("E:\\dataset\\Outdoor\\dark\\54\\", "depth\\depth_", ".png", "rgb\\img_", ".png", 0, 1439);
	//GTruthReader gReader = GTruthReader("E:\\dataset\\Outdoor\\dark\\54.txt");

	string s1 = "D:\\result\\";
	string dataset = "1/";
	double threshold = 0.5;
	while (true) {
		//if ((imLoader.getCount() + 5) % 1 != 0) {
		//	imLoader.next();
		//	continue;
		//}
		//cout << imLoader.getCount() << endl;
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

						pixelNum = 80;
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
		//Mat drawRGB = rawRGB.clone();
		Mat drawRGB03= rawRGB.clone();
		Mat drawRGB05 = rawRGB.clone();
		Mat drawRGB07 = rawRGB.clone();
        //Mat colorMap;
        //rawDepth.convertTo(colorMap, CV_8U, 255.0 / 8000.0);
        //applyColorMap(colorMap, colorMap, cv::COLORMAP_JET);
       // Mat drawColorMap = colorMap.clone();
		//int ccount = 0;
		//int ncount = 0;

        //int t_start = clock();

		{
			l = setcount / 4;
			//Point results[4][100];
			Point results3[4][100];
			Point results5[4][100];
			Point results7[4][100];

			// using 4 thread to calculate 4 different head-top regions
			std::thread t1(th_test, 1, l, threshold, rawDepth, rawRGB,headCloud, clsfr_ptr1,  results3[0], results5[0], results7[0]);
			std::thread t2(th_test, l, 2 * l, threshold, rawDepth, rawRGB,headCloud, clsfr_ptr2, results3[1], results5[1], results7[1]);
			std::thread t3(th_test, 2*l, 3*l, threshold, rawDepth, rawRGB,headCloud, clsfr_ptr3,  results3[2], results5[2], results7[2]);
			std::thread t4(th_test, 3*l, setcount, threshold, rawDepth, rawRGB,headCloud, clsfr_ptr4,  results3[3], results5[3], results7[3]);
			int st1 = clock();
			
			t1.join();
			t2.join();
			t3.join();
			t4.join();


			for (int k = 0; k < 4; k++) {
				cout << results3[k][0].x << endl;
				for (int p = 0; p < results3[k][0].x; p++) {
					int i = results3[k][p].x;
					int j = results3[k][p].y;
					if (j<20 || j>400) continue;
					//cout << i << " " << j << endl;
					int pixelNum = GetSizeInImageBySizeIn3D(150, rawDepth.at<unsigned short>(j, i));
					circle(drawRGB03, Point(i, j), 3, Scalar(0, 69, 255), -1, 8);
					rectangle(drawRGB03, cv::Rect(i - 1.5*pixelNum, j - 0.7*pixelNum, 3 * pixelNum, 4 * pixelNum), cv::Scalar(47, 230, 173), 5, 8);
				}
			}
			for (int k = 0; k < 4; k++) {
				cout << results5[k][0].x << endl;
				for (int p = 0; p < results5[k][0].x; p++) {
					int i = results5[k][p].x;
					int j = results5[k][p].y;
					if (j<20 || j>400) continue;
					//cout << i << " " << j << endl;
					int pixelNum = GetSizeInImageBySizeIn3D(150, rawDepth.at<unsigned short>(j, i));
					circle(drawRGB05, Point(i, j), 3, Scalar(0, 69, 255), -1, 8);
					rectangle(drawRGB05, cv::Rect(i - 1.5*pixelNum, j - 0.7*pixelNum, 3 * pixelNum, 4 * pixelNum), cv::Scalar(47, 230, 173), 5, 8);
				}
			}
			for (int k = 0; k < 4; k++) {
				cout << results7[k][0].x << endl;
				for (int p = 0; p < results7[k][0].x; p++) {
					int i = results7[k][p].x;
					int j = results7[k][p].y;
					if (j<20 || j>400) continue;
					//cout << i << " " << j << endl;
					int pixelNum = GetSizeInImageBySizeIn3D(150, rawDepth.at<unsigned short>(j, i));
					circle(drawRGB07, Point(i, j), 3, Scalar(0, 69, 255), -1, 8);
					rectangle(drawRGB07, cv::Rect(i - 1.5*pixelNum, j - 0.7*pixelNum, 3 * pixelNum, 4 * pixelNum), cv::Scalar(47, 230, 173), 5, 8);
				}
			}

			
		}
		imshow("test1", drawRGB03);
		imshow("test2", drawRGB05);
		imshow("test3", drawRGB07);
		cv::imwrite(s1+"3/"+dataset+ to_string(imLoader.getCount()) + ".png", drawRGB03);
		cv::imwrite(s1 + "5/" + dataset + to_string(imLoader.getCount()) + ".png", drawRGB05);
		cv::imwrite(s1 + "7/" + dataset + to_string(imLoader.getCount()) + ".png", drawRGB07);

		cvWaitKey(1);
		
		if (!imLoader.next()) break;
	}

}

