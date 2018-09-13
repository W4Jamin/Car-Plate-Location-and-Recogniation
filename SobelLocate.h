#include <cv.h>
#include <highgui.h>

using namespace cv;
using namespace std;

bool verifySizess(RotatedRect ROI);
Mat histeq(Mat ima);
vector<Mat> SobelLocate(Mat srcImage, vector<Mat> DetectResultVec);

//int main()
//{
//	vector<Mat> DetectResultVec;
//	
//	vector<Mat> DetectedVec;
//
//	Mat srcImage = imread("E:\\4.jpg"); //E:\\general_test\\��ATH859.jpg
//
//	DetectedVec = SobelLocate(srcImage, DetectResultVec);
//
//	return 0;
//}

vector<Mat> SobelLocate(Mat srcImage, vector<Mat> DetectResultVec)
{
	int m_angle = 60;
	Mat grayImage, blurImage;
	cvtColor(srcImage, grayImage, CV_BGR2GRAY);
	blur(grayImage, blurImage, Size(5, 5));


	Mat sobelImage;
	Sobel(blurImage, sobelImage, CV_8U, 1, 0, 3, 1, 0);
	//imwrite("��Ե����.jpg", sobelImage);

	Mat thresholdImage;
	threshold(sobelImage, thresholdImage, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	Mat erode_dilate;
	Mat XKernal = getStructuringElement(MORPH_RECT, Size(18, 1));
	dilate(thresholdImage, erode_dilate, XKernal, Point(-1, -1), 2);
	erode(erode_dilate, erode_dilate, XKernal, Point(-1, -1), 4);
	dilate(erode_dilate, erode_dilate, XKernal, Point(-1, -1), 2);

	Mat YKernal = getStructuringElement(MORPH_RECT, Size(1, 5));
	dilate(erode_dilate, erode_dilate, YKernal, Point(-1, -1), 2);
	erode(erode_dilate, erode_dilate, YKernal, Point(-1, -1), 4);
	dilate(erode_dilate, erode_dilate, YKernal, Point(-1, -1), 2);

	Mat closeImage;
	Mat structuringElement = getStructuringElement(MORPH_RECT, Size(40, 3));
	morphologyEx(thresholdImage, closeImage, CV_MOP_CLOSE, structuringElement);
	//imwrite("��̬ѧ�㷨��.jpg", erode_dilate);

	vector< vector< Point> > contours;
	findContours(erode_dilate,
		contours, // �����������飬ÿһ��������һ��point���͵�vector��ʾ
		CV_RETR_EXTERNAL, // ��ʾֻ���������
		CV_CHAIN_APPROX_NONE); // �����Ľ��ư취������洢���е�������

	vector<vector<Point> >::iterator itc = contours.begin();
	vector<RotatedRect> rects;

	while (itc != contours.end())
	{
		RotatedRect ROI = minAreaRect(Mat(*itc));
		if (!verifySizess(ROI)){
			itc = contours.erase(itc);
		}
		else{
			++itc;
			rects.push_back(ROI);
		}
	}

	Mat result;
	srcImage.copyTo(result);
	cv::drawContours(result,
		contours,
		-1,				    // ���е�����������
		cv::Scalar(0, 0, 255), // ��ɫ
		1);					// �ߴ�

	//imwrite("��ֵ����.jpg", thresholdImage);

	//imwrite("��ͨ����.jpg", result);
	for (int i = 0; i < rects.size(); i++)
	{
		RotatedRect minRect = rects[i];
		if (verifySizess(minRect))
		{


			Point2f rect_points[4];
			minRect.points(rect_points);
			for (int j = 0; j < 4; j++)
				line(result, rect_points[j], rect_points[(j + 1) % 4], Scalar(0, 255, 255), 1, 8);


			float r = (float)minRect.size.width / (float)minRect.size.height;
			float angle = minRect.angle;
			Size rect_size = minRect.size;
			if (r < 1)
			{
				angle = 90 + angle;

				swap(rect_size.width, rect_size.height);

			}

			//���ץȡ�ķ�����ת����m_angle�Ƕȣ����ǳ��ƣ���������
			if (angle - m_angle < 0 && angle + m_angle > 0)
			{
				Point2f rect_points[4];
				minRect.points(rect_points);
				for (int j = 0; j < 4; j++)
					line(result, rect_points[j], rect_points[(j + 1) % 4], Scalar(0, 255, 0), 2, 8);
				//Create and rotate image
				Mat rotmat = getRotationMatrix2D(minRect.center, angle, 1);
				Mat img_rotated;
				warpAffine(srcImage, img_rotated, rotmat, srcImage.size(), CV_INTER_CUBIC);



				Mat img_crop;
				getRectSubPix(img_rotated, rect_size, minRect.center, img_crop);

				Mat resultResized;
				resultResized.create(36, 136, CV_8UC3);
				resize(img_crop, resultResized, resultResized.size(), 0, 0, INTER_CUBIC);

				Mat blurResult;
				/*cvtColor(resultResized, grayResult, CV_BGR2GRAY);*/
				blur(resultResized, blurResult, Size(3, 3));
				blurResult = histeq(blurResult);

				imwrite("�ų���1.jpg", blurResult);
				DetectResultVec.push_back(blurResult);
			}
		}
	}
	
	return DetectResultVec;
}

bool verifySizess(RotatedRect ROI)
{
	// �������ó���Ĭ�ϲ���������ʶ������������Ƿ�ΪĿ�공��
	float error = 0.6;
	// ���������ƿ�߱�: 520 / 110 = 4.7272   440/140 = 3.1429
	float aspect = 3.6;
	// �趨�����������С/���ߴ磬���ڴ˷�Χ�ڵĲ�����Ϊ����
	int min = 15 * aspect * 15;    // 15������
	int max = 140 * aspect * 140;  // 125������
	float rmin = aspect - aspect*error;
	float rmax = aspect + aspect*error;

	int area = ROI.size.height * ROI.size.width;
	float r = (float)ROI.size.width / (float)ROI.size.height;
	if (r<1)
		r = (float)ROI.size.height / (float)ROI.size.width;

	// �ж��Ƿ�������ϲ���
	if ((area < min || area > max) || (r < rmin || r > rmax))
		return false;
	else
		return true;
}

Mat histeq(Mat ima)
{
	Mat imt(ima.size(), ima.type());
	// ������ͼ��Ϊ��ɫ����Ҫ��HSV�ռ�����ֱ��ͼ���⴦��
	// ��ת����RGB��ʽ
	if (ima.channels() == 3)
	{
		Mat hsv;
		vector<Mat> hsvSplit;
		cvtColor(ima, hsv, CV_BGR2HSV);
		split(hsv, hsvSplit);
		equalizeHist(hsvSplit[2], hsvSplit[2]);
		merge(hsvSplit, hsv);
		cvtColor(hsv, imt, CV_HSV2BGR);
	}
	// ������ͼ��Ϊ�Ҷ�ͼ��ֱ����ֱ��ͼ���⴦��
	else if (ima.channels() == 1){
		equalizeHist(ima, imt);
	}
	return imt;
}
