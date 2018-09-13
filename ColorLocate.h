#include <cv.h>
#include <highgui.h>
#include "PlateLocate.h"

using namespace cv;
using namespace std;

vector<Mat> ColorLocate(Mat srcImage, vector<Mat> LocatedPlate);

vector<Mat> ColorLocate(Mat srcImage, vector<Mat> LocatedPlate)
{
	vector<RotatedRect> outRects;
	Mat match_grey;
	colorMatch(srcImage, match_grey, BLUE, false);

	Mat thresholdImage;
	threshold(match_grey, thresholdImage, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	Mat element = getStructuringElement(MORPH_RECT, Size(36, 2));
	morphologyEx(thresholdImage, thresholdImage, MORPH_CLOSE, element);
	Mat src_b;
	thresholdImage.copyTo(src_b);
	/*imwrite("colorLocate.jpg", thresholdImage);*/
 	vector<vector<Point>> contours;

	findContours(thresholdImage,
		contours,               // a vector of contours
		CV_RETR_EXTERNAL,
		CV_CHAIN_APPROX_NONE);  // all pixels of each contours

	Mat result;
	srcImage.copyTo(result);
	cv::drawContours(result,
		contours,
		-1,				    // 所有的轮廓都画出
		cv::Scalar(0, 0, 255), // 颜色
		1);					// 线粗

	vector<vector<Point>>::iterator itc = contours.begin();
	while (itc != contours.end()) {
		RotatedRect mr = minAreaRect(Mat(*itc));

		if (!verifySizes(mr))
			itc = contours.erase(itc);
		else {
			++itc;
			outRects.push_back(mr);
		}
	}
	Mat mat_debug;
	srcImage.copyTo(mat_debug);
	for (size_t i = 0; i < outRects.size(); i++)
	{
		RotatedRect roi_rect = outRects[i];

		float r = (float)roi_rect.size.width / (float)roi_rect.size.height;
		float roi_angle = roi_rect.angle;

		Size roi_rect_size = roi_rect.size;
		if (r < 1) {
			roi_angle = 90 + roi_angle;
			swap(roi_rect_size.width, roi_rect_size.height);
		}

		Point2f rect_points[4];
		roi_rect.points(rect_points);
		for (int j = 0; j < 4; j++)
			line(mat_debug, rect_points[j], rect_points[(j + 1) % 4],
			Scalar(0, 255, 255), 1, 8);

		if (roi_angle - 60 < 0 && roi_angle + 60 > 0)
		{
			Rect_<float> safeBoundRect;
			bool isFormRect = calcSafeRect(roi_rect, srcImage, safeBoundRect);
			if (!isFormRect) continue;

			Mat bound_mat = srcImage(safeBoundRect);
			Mat bound_mat_b = src_b(safeBoundRect);

			Point2f roi_ref_center = roi_rect.center - safeBoundRect.tl();

			Mat deskew_mat;
			if ((roi_angle - 5 < 0 && roi_angle + 5 > 0) || 90.0 == roi_angle || -90.0 == roi_angle)
				deskew_mat = bound_mat;
			else
			{
				Mat rotated_mat;
				Mat rotated_mat_b;

				if (!rotation(bound_mat, rotated_mat, roi_rect_size, roi_ref_center,
					roi_angle))
					continue;

				if (!rotation(bound_mat_b, rotated_mat_b, roi_rect_size, roi_ref_center,
					roi_angle))
					continue;

				double roi_slope = 0;
				if (isdeflection(rotated_mat_b, roi_angle, roi_slope))
					affine(rotated_mat, deskew_mat, roi_slope);
				else
					deskew_mat = rotated_mat;
			}

			Mat plate_mat;
			plate_mat.create(36, 136, CV_8UC3);
			if (deskew_mat.cols >= 136 || deskew_mat.rows >= 36)
				resize(deskew_mat, plate_mat, plate_mat.size(), 0, 0, INTER_AREA);
			else
				resize(deskew_mat, plate_mat, plate_mat.size(), 0, 0, INTER_CUBIC);

			/*imwrite("原始车牌.jpg", bound_mat);
			imwrite("校正车牌.jpg", plate_mat);*/
			LocatedPlate.push_back(plate_mat);
			//imwrite("找蓝色.jpg",src_b);
			//imwrite("车牌.jpg", plate_mat);
		}
	}
	return LocatedPlate;
}