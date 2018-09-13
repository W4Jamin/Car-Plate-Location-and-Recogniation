#include <cv.h>
#include <highgui.h>

using namespace cv;
using namespace std;

bool verifyCharSizes(Mat r);
Rect GetChineseRect(const Rect rectSpe);
int GetSpecificRect(const vector<Rect>& vecRect);
int RebuildRect(const vector<Rect>& vecRect, vector<Rect>& outRect, int specIndex);
int plateSegment(Mat src, vector<Mat>& resultVec);



bool verifyCharSizes(Mat r) {
	
	float aspect = 45.0f / 90.0f;
	float charAspect = (float)r.cols / (float)r.rows;
	float error = 0.7f;
	float minHeight = 10.f;
	float maxHeight = 35.f;
	
	float minAspect = 0.05f;
	float maxAspect = aspect + aspect * error;
	
	int area = cv::countNonZero(r);
	
	int bbArea = r.cols * r.rows;
	
	int percPixels = area / bbArea;

	if (percPixels <= 1 && charAspect > minAspect && charAspect < maxAspect &&
		r.rows >= minHeight && r.rows < maxHeight)
		return true;
	else
		return false;
}

int plateSegment(Mat src, vector<Mat>& resultVec)
{
	Mat graysrc;
	cvtColor(src, graysrc, CV_BGR2GRAY);
	Mat thresholdImage, thresh_copy;
	threshold(graysrc, thresholdImage, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	thresholdImage.copyTo(thresh_copy);
	vector<vector<Point>> contours;

	findContours(thresholdImage,
		contours,               // a vector of contours
		CV_RETR_EXTERNAL,
		CV_CHAIN_APPROX_NONE);  // all pixels of each contours

	vector<vector<Point> >::iterator itc = contours.begin();
	vector<Rect> vecRect;

	while (itc != contours.end())
	{
		Rect mr = boundingRect(Mat(*itc));
		Mat auxRoi(thresholdImage, mr);

		if (verifyCharSizes(auxRoi)) vecRect.push_back(mr);
		++itc;
	}
	vector<Rect> sortedRect(vecRect);
	std::sort(sortedRect.begin(), sortedRect.end(),
		[](const Rect& r1, const Rect& r2) { return r1.x < r2.x; }); //ÅÅÐò

	size_t specIndex = 0;

	specIndex = GetSpecificRect(sortedRect);

	Rect chineseRect;
	chineseRect = GetChineseRect(sortedRect[specIndex]);
	rectangle(thresh_copy, chineseRect, Scalar(0, 0, 255));

	vector<Rect> newSortedRect;
	newSortedRect.push_back(chineseRect);
	RebuildRect(sortedRect, newSortedRect, specIndex);

	for (size_t i = 0; i < newSortedRect.size(); i++)
	{
		Rect mr = newSortedRect[i];
		Mat auxRoi(graysrc, mr);
		Mat newRoi;
		threshold(auxRoi, newRoi, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
		Mat Roi = Mat::zeros(newRoi.rows, newRoi.cols+10, CV_8UC1);

		for (int i = 0; i < (Roi.rows); i++)
		{
			for (int j = 5; j < (Roi.cols - 5); j++)
			{
				Roi.at<uchar>(i, j) = newRoi.at<uchar>(i, j - 5);
			/*	float a = Roi.at<float>(i, j);*/
			}
		}
		
		//imwrite("7.jpg",Roi);
		resultVec.push_back(Roi);
	}
	return 0;
}

int GetSpecificRect(const vector<Rect>& vecRect) {
	vector<int> xpositions;
	int maxHeight = 0;
	int maxWidth = 0;

	for (size_t i = 0; i < vecRect.size(); i++) {
		xpositions.push_back(vecRect[i].x);

		if (vecRect[i].height > maxHeight) {
			maxHeight = vecRect[i].height;
		}
		if (vecRect[i].width > maxWidth) {
			maxWidth = vecRect[i].width;
		}
	}

	int specIndex = 0;
	for (size_t i = 0; i < vecRect.size(); i++) {
		Rect mr = vecRect[i];
		int midx = mr.x + mr.width / 2;


		if ((mr.width > maxWidth * 0.8 || mr.height > maxHeight * 0.8) &&
			(midx < int(136 / 7) * 2 &&
			midx > int(136 / 7) * 1)) {
			specIndex = i;
		}
	}

	return specIndex;
}

Rect GetChineseRect(const Rect rectSpe) {
	int height = rectSpe.height;
	float newwidth = rectSpe.width * 1.2f;
	int x = rectSpe.x;
	int y = rectSpe.y;

	int newx = x - int(newwidth * 1.2);
	newx = newx > 0 ? newx : 0;

	Rect a(newx, y, int(newwidth), height);

	return a;
}

int RebuildRect(const vector<Rect>& vecRect, vector<Rect>& outRect, int specIndex)
{
	int count = 6;
	for (size_t i = specIndex; i < vecRect.size() && count; ++i, --count) {
		outRect.push_back(vecRect[i]);
	}

	return 0;
}