#include <windows.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <ml.h>

using namespace cv;
using namespace std;

char temp[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' };
char *kChars[] = {
	"´¨", "¶õ", "¸Ó", "¸Ê", "¹ó", "¹ð", "ºÚ", "»¦", "¼½", "½ò", "¾©", "¼ª", "ÁÉ", "Â³", "ÃÉ",
	"Ãö", "Äþ", "Çà", "Çí", "ÉÂ", "ËÕ", "½ú", "Íî", "Ïæ", "ÐÂ", "Ô¥", "Óå", "ÔÁ", "ÔÆ", "²Ø",
	"Õã" };

Mat preprocessChar(Mat in);
Mat ProjectedHistogram(Mat img, int t);
Mat charFeatures2(Mat in, int sizeData);
Mat charFeatures(Mat in, int sizeData);

float countOfBigValue(Mat &mat, int iValue) {
	float iCount = 0.0;
	if (mat.rows > 1) {
		for (int i = 0; i < mat.rows; ++i) {
			if (mat.data[i * mat.step[0]] > iValue) {
				iCount += 1.0;
			}
		}
		return iCount;

	}
	else {
		for (int i = 0; i < mat.cols; ++i) {
			if (mat.data[i] > iValue) {
				iCount += 1.0;
			}
		}

		return iCount;
	}
}

Mat ProjectedHistogram(Mat img, int t) {
	int sz = (t) ? img.rows : img.cols;
	Mat mhist = Mat::zeros(1, sz, CV_32F);

	for (int j = 0; j < sz; j++) {
		Mat data = (t) ? img.row(j) : img.col(j);

		mhist.at<float>(j) = countOfBigValue(data, 20);
	}

	// Normalize histogram
	double min, max;
	minMaxLoc(mhist, &min, &max);

	if (max > 0)
		mhist.convertTo(mhist, -1, 1.0f / max, 0);

	return mhist;
}

Mat preprocessChar(Mat in, int char_size) {
	// Remap image
	int h = in.rows;
	int w = in.cols;

	int charSize = char_size;

	Mat transformMat = Mat::eye(2, 3, CV_32F);
	int m = max(w, h);
	transformMat.at<float>(0, 2) = float(m / 2 - w / 2);
	transformMat.at<float>(1, 2) = float(m / 2 - h / 2);

	Mat warpImage(m, m, in.type());
	warpAffine(in, warpImage, transformMat, warpImage.size(), INTER_LINEAR,
		BORDER_CONSTANT, Scalar(0));

	Mat out;
	cv::resize(warpImage, out, Size(charSize, charSize));

	return out;
}

Rect GetCenterRect(Mat &in) {
	Rect _rect;

	int top = 0;
	int bottom = in.rows - 1;

	// find the center rect

	for (int i = 0; i < in.rows; ++i) {
		bool bFind = false;
		for (int j = 0; j < in.cols; ++j) {
			if (in.data[i * in.step[0] + j] > 20) {
				top = i;
				bFind = true;
				break;
			}
		}
		if (bFind) {
			break;
		}

	}
	for (int i = in.rows - 1;
		i >= 0;
		--i) {
		bool bFind = false;
		for (int j = 0; j < in.cols; ++j) {
			if (in.data[i * in.step[0] + j] > 20) {
				bottom = i;
				bFind = true;
				break;
			}
		}
		if (bFind) {
			break;
		}

	}


	int left = 0;
	int right = in.cols - 1;
	for (int j = 0; j < in.cols; ++j) {
		bool bFind = false;
		for (int i = 0; i < in.rows; ++i) {
			if (in.data[i * in.step[0] + j] > 20) {
				left = j;
				bFind = true;
				break;
			}
		}
		if (bFind) {
			break;
		}

	}
	for (int j = in.cols - 1;
		j >= 0;
		--j) {
		bool bFind = false;
		for (int i = 0; i < in.rows; ++i) {
			if (in.data[i * in.step[0] + j] > 20) {
				right = j;
				bFind = true;

				break;
			}
		}
		if (bFind) {
			break;
		}
	}

	_rect.x = left;
	_rect.y = top;
	_rect.width = right - left + 1;
	_rect.height = bottom - top + 1;

	return _rect;
}

Mat CutTheRect(Mat &in, Rect &rect) {
	int size = in.cols;  // (rect.width>rect.height)?rect.width:rect.height;
	Mat dstMat(size, size, CV_8UC1);
	dstMat.setTo(Scalar(0, 0, 0));

	int x = (int)floor((float)(size - rect.width) / 2.0f);
	int y = (int)floor((float)(size - rect.height) / 2.0f);

	for (int i = 0; i < rect.height; ++i) {

		for (int j = 0; j < rect.width; ++j) {
			dstMat.data[dstMat.step[0] * (i + y) + j + x] =
				in.data[in.step[0] * (i + rect.y) + j + rect.x];
		}
	}

	//
	return dstMat;
}

Mat charFeatures(Mat in, int sizeData) {
	const int VERTICAL = 0;
	const int HORIZONTAL = 1;

	// cut the cetner, will afect 5% perices.
	Rect _rect = GetCenterRect(in);
	Mat tmpIn = CutTheRect(in, _rect);
	//Mat tmpIn = in.clone();

	// Low data feature
	Mat lowData;
	resize(tmpIn, lowData, Size(sizeData, sizeData));

	// Histogram features
	Mat vhist = ProjectedHistogram(lowData, VERTICAL);
	Mat hhist = ProjectedHistogram(lowData, HORIZONTAL);

	// Last 10 is the number of moments components
	int numCols = vhist.cols + hhist.cols + lowData.cols * lowData.cols;

	Mat out = Mat::zeros(1, numCols, CV_32F);
	// Asign values to

	int j = 0;
	for (int i = 0; i < vhist.cols; i++) {
		out.at<float>(j) = vhist.at<float>(i);
		j++;
	}
	for (int i = 0; i < hhist.cols; i++) {
		out.at<float>(j) = hhist.at<float>(i);
		j++;
	}
	for (int x = 0; x < lowData.cols; x++) {
		for (int y = 0; y < lowData.rows; y++) {
			out.at<float>(j) += (float)lowData.at <unsigned char>(x, y);
			j++;
		}
	}

	//std::cout << out << std::endl;

	return out;
}

Mat charFeatures2(Mat in, int sizeData) {
	const int VERTICAL = 0;
	const int HORIZONTAL = 1;

	// cut the cetner, will afect 5% perices.
	Rect _rect = GetCenterRect(in);
	Mat tmpIn = CutTheRect(in, _rect);
	//Mat tmpIn = in.clone();

	// Low data feature
	Mat lowData;
	resize(in, lowData, Size(sizeData, sizeData));

	// Histogram features
	Mat vhist = ProjectedHistogram(lowData, VERTICAL);
	Mat hhist = ProjectedHistogram(lowData, HORIZONTAL);

	// Last 10 is the number of moments components
	int numCols = vhist.cols + hhist.cols + lowData.cols * lowData.cols;

	Mat out = Mat::zeros(1, numCols, CV_32F);
	// Asign values to

	// feature,ANNµÄÑù±¾ÌØÕ÷ÎªË®Æ½¡¢´¹Ö±Ö±·½Í¼ºÍµÍ·Ö±æÂÊÍ¼ÏñËù×é³ÉµÄÊ¸Á¿

	int j = 0;
	for (int i = 0; i < vhist.cols; i++) {
		out.at<float>(j) = vhist.at<float>(i);
		j++;
	}
	for (int i = 0; i < hhist.cols; i++) {
		out.at<float>(j) = hhist.at<float>(i);
		j++;
	}
	for (int x = 0; x < lowData.cols; x++) {
		for (int y = 0; y < lowData.rows; y++) {
			out.at<float>(j) += (float)lowData.at <unsigned char>(x, y);
			j++;
		}
	}

	//std::cout << out << std::endl;

	return out;
}