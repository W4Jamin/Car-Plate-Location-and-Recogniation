#include <opencv2/opencv.hpp>
#include <ml.h>

using namespace cv;
using namespace std;

void getLBPFeatures(const Mat& image, Mat& features);
vector<Mat> SVMjudge(vector<Mat> DetectedVec, vector<Mat> JudgedPlate);

vector<Mat> SVMjudge(vector<Mat> DetectedVec, vector<Mat> JudgedPlate)
{
	CvSVM svm;
	svm.load("E:\\svm.xml");


	int num = DetectedVec.size();
	for (int j = 0; j < num; j++)
	{
		Mat temp = DetectedVec[j];

		Mat features;
		getLBPFeatures(temp, features);

		float response = svm.predict(features);

		if (response == 1)
			JudgedPlate.push_back(DetectedVec[j]);
		
	}
	return JudgedPlate;
}

template <typename _Tp> static
void olbp_(InputArray _src, OutputArray _dst) {
	// get matrices
	Mat src = _src.getMat();
	// allocate memory for result
	_dst.create(src.rows - 2, src.cols - 2, CV_8UC1);
	Mat dst = _dst.getMat();
	// zero the result matrix
	dst.setTo(0);
	// calculate patterns
	for (int i = 1; i<src.rows - 1; i++) {
		for (int j = 1; j<src.cols - 1; j++) {
			_Tp center = src.at<_Tp>(i, j);
			unsigned char code = 0;
			code |= (src.at<_Tp>(i - 1, j - 1) >= center) << 7;
			code |= (src.at<_Tp>(i - 1, j) >= center) << 6;
			code |= (src.at<_Tp>(i - 1, j + 1) >= center) << 5;
			code |= (src.at<_Tp>(i, j + 1) >= center) << 4;
			code |= (src.at<_Tp>(i + 1, j + 1) >= center) << 3;
			code |= (src.at<_Tp>(i + 1, j) >= center) << 2;
			code |= (src.at<_Tp>(i + 1, j - 1) >= center) << 1;
			code |= (src.at<_Tp>(i, j - 1) >= center) << 0;
			dst.at<unsigned char>(i - 1, j - 1) = code;
		}
	}
}

void olbp(InputArray src, OutputArray dst) {
	switch (src.getMat().type()) {
	case CV_8SC1:   olbp_<char>(src, dst); break;
	case CV_8UC1:   olbp_<unsigned char>(src, dst); break;
	case CV_16SC1:  olbp_<short>(src, dst); break;
	case CV_16UC1:  olbp_<unsigned short>(src, dst); break;
	case CV_32SC1:  olbp_<int>(src, dst); break;
	case CV_32FC1:  olbp_<float>(src, dst); break;
	case CV_64FC1:  olbp_<double>(src, dst); break;
	default: break;
	}
}

Mat olbp(InputArray src) {
	Mat dst;
	olbp(src, dst);
	return dst;
}

static Mat
histc_(const Mat& src, int minVal = 0, int maxVal = 255, bool normed = false) {
	Mat result;
	// Establish the number of bins.
	int histSize = maxVal - minVal + 1;
	// Set the ranges.
	float range[] = { static_cast<float>(minVal), static_cast<float>(maxVal + 1) };
	const float* histRange = { range };
	// calc histogram
	calcHist(&src, 1, 0, Mat(), result, 1, &histSize, &histRange, true, false);
	// normalize
	if (normed) {
		result /= src.total();
	}
	return result.reshape(1, 1);
}

Mat histc(InputArray _src, int minVal, int maxVal, bool normed) {
	Mat src = _src.getMat();
	switch (src.type()) {
	case CV_8SC1:
		return histc_(Mat_<float>(src), minVal, maxVal, normed);
		break;
	case CV_8UC1:
		return histc_(src, minVal, maxVal, normed);
		break;
	case CV_16SC1:
		return histc_(Mat_<float>(src), minVal, maxVal, normed);
		break;
	case CV_16UC1:
		return histc_(src, minVal, maxVal, normed);
		break;
	case CV_32SC1:
		return histc_(Mat_<float>(src), minVal, maxVal, normed);
		break;
	case CV_32FC1:
		return histc_(src, minVal, maxVal, normed);
		break;
	default:
		CV_Error(CV_StsUnmatchedFormats, "This type is not implemented yet."); break;
	}
	return Mat();
}

Mat spatial_histogram(InputArray _src, int numPatterns, int grid_x, int grid_y) {
	Mat src = _src.getMat();
	// calculate LBP patch size
	int width = src.cols / grid_x;
	int height = src.rows / grid_y;
	// allocate memory for the spatial histogram
	Mat result = Mat::zeros(grid_x * grid_y, numPatterns, CV_32FC1);
	// return matrix with zeros if no data was given
	if (src.empty())
		return result.reshape(1, 1);
	// initial result_row
	int resultRowIdx = 0;
	// iterate through grid
	for (int i = 0; i < grid_y; i++) {
		for (int j = 0; j < grid_x; j++) {
			Mat src_cell = Mat(src, Range(i*height, (i + 1)*height), Range(j*width, (j + 1)*width));
			Mat cell_hist = histc(src_cell, 0, (numPatterns - 1), true);
			// copy to the result matrix
			Mat result_row = result.row(resultRowIdx);
			cell_hist.reshape(1, 1).convertTo(result_row, CV_32FC1);
			// increase row count in result matrix
			resultRowIdx++;
		}
	}
	// return result as reshaped feature vector
	return result.reshape(1, 1);
}

void getLBPFeatures(const Mat& image, Mat& features) {

	Mat grayImage;
	cvtColor(image, grayImage, CV_RGB2GRAY);

	//if (1) {
	//  imshow("grayImage", grayImage);
	//  waitKey(0);
	//  destroyWindow("grayImage");
	//}

	//spatial_ostu(grayImage, 8, 2);

	//if (1) {
	//  imshow("grayImage", grayImage);
	//  waitKey(0);
	//  destroyWindow("grayImage");
	//}

	Mat lbpimage;
	lbpimage = olbp(grayImage);
	Mat lbp_hist = spatial_histogram(lbpimage, 32, 4, 4);

	features = lbp_hist;
}
