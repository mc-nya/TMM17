#pragma once
#include<string>
#include<fstream>
#include<vector>
#include<opencv2/core.hpp>
#include<iostream>
#include <boost/algorithm/string.hpp>
//read ground truth for DARK dataset
class GTruthReader {
public:
	GTruthReader(std::string infile) {
		using namespace std;
		fin.open(infile, ios::in);
	}
	std::vector<cv::Rect> getTruthSet() {
		using namespace std;
		std::vector<cv::Rect> ret;
		getline(fin, currLine);
		vector<string> split_result;
		boost::split(split_result, currLine, boost::is_any_of("("));
		auto str=split_result.begin()+1;
		for (; str != split_result.end(); str++) {
			vector<string> vec_num;
			boost::split(vec_num, (*str), boost::is_any_of(","));
			cv::Rect rec;
			auto str2 = vec_num.begin();
			rec.x = atoi((*str2++).c_str());
			rec.y = atoi((*str2++).c_str());
			rec.width = atoi((*str2++).c_str())- rec.x;
			rec.height = atoi((*str2).substr(0,str2->size()-2).c_str())- rec.y;
			ret.push_back(rec);
		}
		return ret;
	}
protected:
	std::fstream fin;
	std::string currLine;
};