#include <stdio.h>
#include <windows.h>
#include <cv.h>
#include <highgui.h>
#include <cvaux.h>
#include <ml.h>

using namespace cv;
using namespace std;


enum Color { BLUE, YELLOW, WHITE, UNKNOWN };
Mat colorMatch(const Mat &src, Mat &match, const Color r, const bool adaptive_minsv);
bool verifySizes(RotatedRect ROI);
int ColorSearch(const Mat &srcImage, const Color r, Mat &out, vector<RotatedRect> &outRects);
bool calcSafeRect(const RotatedRect &roi_rect, const Mat &src, Rect_<float> &safeBoundRect);
bool rotation(Mat &in, Mat &out, const Size rect_size, const Point2f center, const double angle);
void affine(const Mat &in, Mat &out, const double slope);
bool isdeflection(const Mat &in, const double angle, double &slope);


bool verifySizes(RotatedRect mr)
{
	float error = 0.9;
	//Spain car plate size: 52x11 aspect 4,7272
	//China car plate size: 440mm*140mm£¬aspect 3.142857
	float aspect = 3.8;
	//Set a min and max area. All other patchs are discarded
	//int min= 1*aspect*1; // minimum area
	//int max= 2000*aspect*2000; // maximum area
	int min = 44 * 14 * 1; // minimum area
	int max = 44 * 14 * 40; // maximum area
	//Get only patchs that match to a respect ratio.
	float rmin = aspect - aspect*error;
	float rmax = aspect + aspect*error;

	int area = mr.size.height * mr.size.width;
	float r = (float)mr.size.width / (float)mr.size.height;
	if (r < 1)
	{
		r = (float)mr.size.height / (float)mr.size.width;
	}

	if ((area < min || area > max) || (r < rmin || r > rmax))
	{
		return false;
	}
	else
	{
		return true;
	}
}

Mat colorMatch(const Mat &src, Mat &match, const Color r,
	const bool adaptive_minsv) {

	// if use adaptive_minsv
	// min value of s and v is adaptive to h

	const float max_sv = 255;
	const float minref_sv = 64;

	const float minabs_sv = 95;

	// H range of blue 

	const int min_blue = 100;  // 100
	const int max_blue = 140;  // 140


	Mat src_hsv;

	// convert to HSV space
	cvtColor(src, src_hsv, CV_BGR2HSV);

	std::vector<cv::Mat> hsvSplit;
	split(src_hsv, hsvSplit);
	equalizeHist(hsvSplit[2], hsvSplit[2]);
	merge(hsvSplit, src_hsv);

	// match to find the color

	int	min_h = min_blue;
	int	max_h = max_blue;


	float diff_h = float((max_h - min_h) / 2);
	float avg_h = min_h + diff_h;

	int channels = src_hsv.channels();
	int nRows = src_hsv.rows;

	// consider multi channel image
	int nCols = src_hsv.cols * channels;
	if (src_hsv.isContinuous()) {
		nCols *= nRows;
		nRows = 1;
	}

	int i, j;
	uchar* p;
	float s_all = 0;
	float v_all = 0;
	float count = 0;
	for (i = 0; i < nRows; ++i) {
		p = src_hsv.ptr<uchar>(i);
		for (j = 0; j < nCols; j += 3) {
			int H = int(p[j]);      // 0-180
			int S = int(p[j + 1]);  // 0-255
			int V = int(p[j + 2]);  // 0-255

			s_all += S;
			v_all += V;
			count++;

			bool colorMatched = false;

			if (H > min_h && H < max_h) {
				float Hdiff = 0;
				if (H > avg_h)
					Hdiff = H - avg_h;
				else
					Hdiff = avg_h - H;

				float Hdiff_p = float(Hdiff) / diff_h;

				float min_sv = 0;
				if (true == adaptive_minsv)
					min_sv =
					minref_sv -minref_sv / 2 *(1- Hdiff_p);  // inref_sv - minref_sv / 2 * (1 - Hdiff_p)
				else
					min_sv = minabs_sv;  // add

				if ((S > min_sv && S < max_sv) && (V > min_sv && V < max_sv))
					colorMatched = true;
			}

			if (colorMatched == true) {
				p[j] = 0;
				p[j + 1] = 0;
				p[j + 2] = 255;
			}
			else {
				p[j] = 0;
				p[j + 1] = 0;
				p[j + 2] = 0;
			}
		}
	}

	// cout << "avg_s:" << s_all / count << endl;
	// cout << "avg_v:" << v_all / count << endl;

	// get the final binary

	Mat src_grey;

	std::vector<cv::Mat> hsvSplit_done;
	split(src_hsv, hsvSplit_done);
	src_grey = hsvSplit_done[2];

	match = src_grey;

	return src_grey;
}

bool calcSafeRect(const RotatedRect &roi_rect, const Mat &src,
	Rect_<float> &safeBoundRect) {
	Rect_<float> boudRect = roi_rect.boundingRect();

	float tl_x = boudRect.x > 0 ? boudRect.x : 0;
	float tl_y = boudRect.y > 0 ? boudRect.y : 0;

	float br_x = boudRect.x + boudRect.width < src.cols
		? boudRect.x + boudRect.width - 1
		: src.cols - 1;
	float br_y = boudRect.y + boudRect.height < src.rows
		? boudRect.y + boudRect.height - 1
		: src.rows - 1;

	float roi_width = br_x - tl_x;
	float roi_height = br_y - tl_y;

	if (roi_width <= 0 || roi_height <= 0) return false;

	//  a new rect not out the range of mat

	safeBoundRect = Rect_<float>(tl_x, tl_y, roi_width, roi_height);

	return true;
}

bool rotation(Mat &in, Mat &out, const Size rect_size, const Point2f center, const double angle)
{
	Mat in_large;
	in_large.create(int(in.rows * 1.5), int(in.cols * 1.5), in.type());

	float x = in_large.cols / 2 - center.x > 0 ? in_large.cols / 2 - center.x : 0;
	float y = in_large.rows / 2 - center.y > 0 ? in_large.rows / 2 - center.y : 0;

	float width = x + in.cols < in_large.cols ? in.cols : in_large.cols - x;
	float height = y + in.rows < in_large.rows ? in.rows : in_large.rows - y;

	/*assert(width == in.cols);
	assert(height == in.rows);*/

	if (width != in.cols || height != in.rows) return false;

	Mat imageRoi = in_large(Rect_<float>(x, y, width, height));
	addWeighted(imageRoi, 0, in, 1, 0, imageRoi);

	Point2f center_diff(in.cols / 2.f, in.rows / 2.f);
	Point2f new_center(in_large.cols / 2.f, in_large.rows / 2.f);

	Mat rot_mat = getRotationMatrix2D(new_center, angle, 1);


	Mat mat_rotated;
	warpAffine(in_large, mat_rotated, rot_mat, Size(in_large.cols, in_large.rows), CV_INTER_CUBIC);


	Mat img_crop;
	getRectSubPix(mat_rotated, Size(rect_size.width, rect_size.height),
		new_center, img_crop);

	out = img_crop;


	return true;
}

bool isdeflection(const Mat &in, const double angle, double &slope)
{
	int nRows = in.rows;
	int nCols = in.cols;

	assert(in.channels() == 1);

	int comp_index[3];
	int len[3];

	comp_index[0] = nRows / 4;
	comp_index[1] = nRows / 4 * 2;
	comp_index[2] = nRows / 4 * 3;

	const uchar* p;

	for (int i = 0; i < 3; i++) {
		int index = comp_index[i];
		p = in.ptr<uchar>(index);

		int j = 0;
		int value = 0;
		while (0 == value && j < nCols) value = int(p[j++]);

		len[i] = j;
	}

	// cout << "len[0]:" << len[0] << endl;
	// cout << "len[1]:" << len[1] << endl;
	// cout << "len[2]:" << len[2] << endl;

	// len[0]/len[1]/len[2] are used to calc the slope

	double maxlen = max(len[2], len[0]);
	double minlen = min(len[2], len[0]);
	double difflen = abs(len[2] - len[0]);

	double PI = 3.14159265;

	double g = tan(angle * PI / 180.0);

	if (maxlen - len[1] > nCols / 32 || len[1] - minlen > nCols / 32) {

		double slope_can_1 =
			double(len[2] - len[0]) / double(comp_index[1]);
		double slope_can_2 = double(len[1] - len[0]) / double(comp_index[0]);
		double slope_can_3 = double(len[2] - len[1]) / double(comp_index[0]);
		// cout<<"angle:"<<angle<<endl;
		// cout<<"g:"<<g<<endl;
		// cout << "slope_can_1:" << slope_can_1 << endl;
		// cout << "slope_can_2:" << slope_can_2 << endl;
		// cout << "slope_can_3:" << slope_can_3 << endl;
		// if(g>=0)
		slope = abs(slope_can_1 - g) <= abs(slope_can_2 - g) ? slope_can_1
			: slope_can_2;
		// cout << "slope:" << slope << endl;
		return true;
	}
	else {
		slope = 0;
	}

	return false;
}


void affine(const Mat &in, Mat &out, const double slope)
{
	// imshow("in", in);
	// waitKey(0);

	Point2f dstTri[3];
	Point2f plTri[3];

	float height = (float)in.rows;
	float width = (float)in.cols;
	float xiff = (float)abs(slope) * height;

	if (slope > 0) {

		// right, new position is xiff/2

		plTri[0] = Point2f(0, 0);
		plTri[1] = Point2f(width - xiff - 1, 0);
		plTri[2] = Point2f(0 + xiff, height - 1);

		dstTri[0] = Point2f(xiff / 2, 0);
		dstTri[1] = Point2f(width - 1 - xiff / 2, 0);
		dstTri[2] = Point2f(xiff / 2, height - 1);
	}
	else {

		// left, new position is -xiff/2

		plTri[0] = Point2f(0 + xiff, 0);
		plTri[1] = Point2f(width - 1, 0);
		plTri[2] = Point2f(0, height - 1);

		dstTri[0] = Point2f(xiff / 2, 0);
		dstTri[1] = Point2f(width - 1 - xiff + xiff / 2, 0);
		dstTri[2] = Point2f(xiff / 2, height - 1);
	}

	Mat warp_mat = getAffineTransform(plTri, dstTri);

	Mat affine_mat;
	affine_mat.create((int)height, (int)width, CV_8UC3);

	if (in.rows > 36 || in.cols > 136)

		warpAffine(in, affine_mat, warp_mat, affine_mat.size(),
		CV_INTER_AREA);
	else
		warpAffine(in, affine_mat, warp_mat, affine_mat.size(), CV_INTER_CUBIC);

	out = affine_mat;

}
