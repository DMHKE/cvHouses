#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <iostream>

using namespace cv;
using namespace std;

//Canny/Edge Values
#define MAXPOLYCOUNT 12
#define MINPOLYCOUNT 3

#define MAXSIZEVALUE 8
#define MINSIZEVALUE 0.4

#define DIFFAREAFACTOR 0.4

#define SHADOW 90

//Area / Homogeneous Operator Values
//TODO

bool isGreen(int r, int g, int b)
{
	if (g > 50 && g > r && g > b)
	{
		return true;
	}
	return false;
}

bool isBlue(int r, int g, int b)
{
	if (b > 100 && b > (g * 2) && b > (r * 2))
	{
		return true;
	}
	return false;
}

bool isShadow(int r, int g, int b, double luminanceMedian)
{
	double luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
	if (luminance < luminanceMedian / 2)
	{
		return true;
	}
	return false;
}

/** @brief Checks if the size of the structure is in the correct range
*
* @param size input size of the structure
* @param avarage avarage size of all found structures
*
* @return computed boolean if the size is correct or not
*/
bool sizeFilter(double size, double avarage)
{
	return size > (avarage * MINSIZEVALUE) && size < avarage* MAXSIZEVALUE;// Old values: area > 1000 && area < 10000
}

/** @brief Checks if the count of the polygons of the given structure is in the correct range
*
* @param polycount apporx. polycount of the structure
*
* @return computed boolean if the polycount is correct or not
*/
bool polyFilter(size_t polycount)
{
	return !(polycount <= MINPOLYCOUNT || polycount >= MAXPOLYCOUNT);
}

/** @brief Calculates the median area out of all found areas in the picture
*
* @param contours point vectors of all found contours
*
* @return computed double value of the median area
*/
double calculateMedianArea(vector<vector<Point>> contours)
{
	double sum = 0;
	double count = 0;
	vector<vector<Point>> conPoly(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		double peri = arcLength(contours[i], true);
		approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);

		//Filter complex or non complex polygons
		if (conPoly[i].size() <= 3 || conPoly[i].size() >= 12)
		{
			continue;
		}

		sum += area;
		count++;
	}

	return sum / count;
}

double calculateMedianLuminance(Mat img)
{
	double luminanceSum = 0;
	int count = 0;
	for (int i = 0; i < img.cols; i++)
	{
		for (int j = 0; j < img.rows; j++)
		{
			Vec3b color = img.at<Vec3b>(j, i);

			int r = color.val[2];
			int g = color.val[1];
			int b = color.val[0];
			luminanceSum += 0.2126 * r + 0.7152 * g + 0.0722 * b;
			count++;
		}
	}
	return luminanceSum / count;
}

/** @brief Calculates the area difference factor (<= 0.4 big difference) between bounding box and area
*
* @param boundRect bounding box of the area
* @param area area of the found shape
*
* @return computed boolean if the difference is bigger than the threshhold.
*/
bool areaDiffFilter(Rect boundRect, double area)
{
	double boundingRectArea = (double)boundRect.height * (double)boundRect.width;
	double diffArea = area / boundingRectArea;
	if (diffArea <= DIFFAREAFACTOR)
	{
		return false;
	}
	return true;
}

double caluclateCannyThresh(Mat greyImg)
{
	Mat dest;
	double cannyHighThresh = threshold(greyImg, dest, 0, 255, THRESH_BINARY | THRESH_OTSU);
	return cannyHighThresh;
}

bool onSegment(Point p, Point q, Point r)
{
	if (q.x <= max(p.x, r.x) && q.x >= min(p.x, r.x) &&
		q.y <= max(p.y, r.y) && q.y >= min(p.y, r.y))
	{
		return true;
	}
	return false;
}

int orientation(Point p, Point q, Point r)
{
	int val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);

	if (val == 0) return 0; // collinear
	return (val > 0) ? 1 : 2; // clock or counterclock wise
}

// The function that returns true if line segment 'p1q1'
// and 'p2q2' intersect.
bool doIntersect(Point p1, Point q1, Point p2, Point q2)
{
	// Find the four orientations needed for general and
	// special cases
	int o1 = orientation(p1, q1, p2);
	int o2 = orientation(p1, q1, q2);
	int o3 = orientation(p2, q2, p1);
	int o4 = orientation(p2, q2, q1);

	// General case
	if (o1 != o2 && o3 != o4)
		return true;

	// Special Cases
	// p1, q1 and p2 are collinear and p2 lies on segment p1q1
	if (o1 == 0 && onSegment(p1, p2, q1)) return true;

	// p1, q1 and p2 are collinear and q2 lies on segment p1q1
	if (o2 == 0 && onSegment(p1, q2, q1)) return true;

	// p2, q2 and p1 are collinear and p1 lies on segment p2q2
	if (o3 == 0 && onSegment(p2, p1, q2)) return true;

	// p2, q2 and q1 are collinear and q1 lies on segment p2q2
	if (o4 == 0 && onSegment(p2, q1, q2)) return true;

	return false; // Doesn't fall in any of the above cases
}

bool isInsidePolygon(int x, int y, vector<Point> polygon)
{
	if (polygon.size() < 3)
	{
		return false;
	}

	Point extremePoint = { 10000, y };

	int intersections = 0;
	int i = 0;
	Point p = { x, y };
	do
	{
		int nextIndex = (i + 1) % (polygon.size() - 1);
		if (doIntersect(polygon[i], polygon[nextIndex], p, extremePoint))
		{
			// If the point 'p' is collinear with line segment 'i-next',
			// then check if it lies on segment. If it lies, return true,
			// otherwise false
			if (orientation(polygon[i], p, polygon[nextIndex]) == 0)
			{
				return onSegment(polygon[i], p, polygon[nextIndex]);
			}
			intersections++;
		}
		i = nextIndex;
	} while (i != 0);

	// Return true if count is odd, false otherwise
	return intersections & 1; // Same as (count%2 == 1)
}

double* caluclateMedianRGB(Mat img, vector<Point> polygon, Rect boundingRect)
{
	double rgb[3];
	int startX = boundingRect.x;
	int startY = boundingRect.y;
	int maxX = startX + boundingRect.width;
	int maxY = startY + boundingRect.height;
	double r = 0;
	double g = 0;
	double b = 0;
	double count = 0;
	for (int i = startX; i < maxX; i++)
	{
		for (int j = startY; j < maxY; j++)
		{
			if (isInsidePolygon(i, j, polygon))
			{
				Vec3b color = img.at<Vec3b>(j, i);

				int rt = color.val[2];
				int gt = color.val[1];
				int bt = color.val[0];



				b += bt;
				g += gt;
				r += rt;
				count++;
			}
		}
	}
	rgb[0] = r / count;
	rgb[1] = g / count;
	rgb[2] = b / count;
	return rgb;
}

bool valueInRange(int value, int min, int max)
{
	return (value >= min) && (value <= max);
}

bool rectOverlap(Rect A, Rect B)
{
	bool xOverlap = valueInRange(A.x, B.x, B.x + B.width) || valueInRange(B.x, A.x, A.x + A.width);

	bool yOverlap = valueInRange(A.y, B.y, B.y + B.height) || valueInRange(B.y, A.y, A.y + A.height);

	return xOverlap && yOverlap;
}

bool overLapFilter(vector<Rect> rectangleList, Rect rect)
{
	for (int j = 0; j < rectangleList.size(); j++)
	{
		if (!rectangleList.empty())
		{
			if (rectOverlap(rectangleList[j], rect))
			{
				return true;
			}
		}
	}
	return false;
}

/** @breif Preprocessing of the image for edge processing
*
* @param imgInput Input image
*
* @return preprocessed image
*/
Mat preprocessingEdges(Mat imgInput)//Bilateral + Denoising + Hist
{
	Mat imgGray, imgBlur, imgCanny, imgDenoise, imgOutput, imgDilate, imgBilateral, imgHist, imgSharp, imgLaplace;

	//Optimise Edges 5, 75, 75
	bilateralFilter(imgInput, imgBilateral, 9, 160, 160);

	cvtColor(imgBilateral, imgGray, COLOR_BGR2GRAY);
	medianBlur(imgGray, imgBlur, 3);
	GaussianBlur(imgGray, imgBlur, Size(3, 3), 0);
	fastNlMeansDenoising(imgBlur, imgDenoise, 5, 7, 30);
	equalizeHist(imgDenoise, imgHist);//imgDenoise

	float imgData[9] = { -1,-1,-1,
						-1,9,-1,
						-1,-1,-1 };
	Mat sharpeningKernel = Mat(3, 3, CV_32F, imgData);

	filter2D(imgHist, imgSharp, -1, sharpeningKernel);

	imshow("Hist Img", imgSharp);

	//Laplacian(imgSharp, imgLaplace, -1);
	double upperThresh = caluclateCannyThresh(imgSharp);
	double lowerThresh = 0.1 * upperThresh;
	Canny(imgSharp, imgCanny, lowerThresh, upperThresh, 3, true);

	//Closing the lines
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	dilate(imgCanny, imgDilate, kernel);

	Mat erodeKernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	erode(imgDilate, imgOutput, erodeKernel);

	return imgOutput;
}

/** @breif highlight all countours of the found structures and count the structures
*
* @param imgInput preprocessed input image
* @param imgOutput original image that gets the countours and data printed on
*
* @return number of found structures after filtering the wrong contours
*/
int highlightContoursEdge(Mat originalImg, Mat imgInput, Mat imgOutput)
{
	int houseCount = 0;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	//TODO hierarchy benutzen
	findContours(imgInput, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);

	//TODO Join regions / areas that are close together

	vector<vector<Point>> conPoly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Rect> rectangleList(contours.size());

	double medianArea = calculateMedianArea(contours);
	double medianLuminance = calculateMedianLuminance(originalImg);
	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (sizeFilter(area, medianArea))
		{

			double peri = arcLength(contours[i], true);
			approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);
			if (polyFilter(conPoly[i].size()))
			{
				boundRect[i] = boundingRect(conPoly[i]);
				if (areaDiffFilter(boundRect[i], area))
				{
					double* rgb = caluclateMedianRGB(originalImg, conPoly[i], boundRect[i]);
					double r = rgb[0];
					double g = rgb[1];
					double b = rgb[2];
					if (!isShadow(r, g, b, medianLuminance) && !isBlue(r, g, b) && !isGreen(r, g, b))
					{
						if (!overLapFilter(rectangleList, boundRect[i]))
						{
							rectangleList.push_back(boundRect[i]);
							houseCount++;
							drawContours(imgOutput, conPoly, i, Scalar(0, 0, 255), 2);
							rectangle(imgOutput, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 3);
							putText(imgOutput, to_string(houseCount), { boundRect[i].x, boundRect[i].y - 5 }, FONT_HERSHEY_PLAIN, 1, Scalar(255, 0, 0), 2);
						}
					}
				}
			}
		}
	}
	String houseCountOutput = "House Count: " + to_string(houseCount);
	putText(imgOutput, houseCountOutput, { 15, 20 }, FONT_HERSHEY_PLAIN, 2, Scalar(255, 0, 255), 2);
	return houseCount;
}

/** @breif Find the house count with edge detection methods
*
* @param path image path
*/
void findHousesEdge(string path)
{
	//Edges
	Mat img = imread(path);
	Mat imgPreprocessed;

	imgPreprocessed = preprocessingEdges(img);
	imshow("Image Preprocessed", imgPreprocessed);

	Mat original = img.clone();
	int count = highlightContoursEdge(original, imgPreprocessed, img);
	cout << "House Count Edges: " << count << endl;
	imshow("House Count Edges", img);
}


//NEU

void houghTest(string path)
{
	Mat img = imread(path);
	Mat imgResult = img.clone();
	Mat imgBilateral, imgGray, imgBlur, imgSharp, imgHist, imgCanny, imgHough;

	bilateralFilter(img, imgBilateral, -1, 10, 10);

	cvtColor(imgBilateral, imgGray, COLOR_BGR2GRAY);

	GaussianBlur(imgGray, imgBlur, Size(3, 3), 0);
	equalizeHist(imgBlur, imgHist);

	float imgData[9] = { -1,-1,-1,
						-1,9,-1,
						-1,-1,-1 };
	Mat sharpeningKernel = Mat(3, 3, CV_32F, imgData);

	filter2D(imgHist, imgSharp, -1, sharpeningKernel);

	double upperThresh = caluclateCannyThresh(imgGray);
	double lowerThresh = 0.1 * upperThresh;
	Canny(imgSharp, imgCanny, lowerThresh, upperThresh, 3, true);
	vector<Vec4i> lines;
	HoughLinesP(imgCanny, lines, 1, CV_PI / 180, upperThresh, 3, 1);

	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		line(imgResult, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
	}
	imshow("Original", img);
	imshow("Hough", imgResult);
}

void vegetationTest(String path)
{
	Mat img = imread(path, IMREAD_GRAYSCALE);
	Mat blur, thresh;

	GaussianBlur(img, blur, Size(19,19), 0);
	threshold(blur, thresh, 0, 255, THRESH_BINARY + THRESH_OTSU);

	imshow("Vegetation", thresh);
}

void thresholdTests(string path)
{
	Mat img = imread(path, IMREAD_GRAYSCALE);
	Mat imgOriginal = img.clone();
	Mat imgThresh, imgCanny;
	adaptiveThreshold(img, imgThresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
	imshow("Thresh", imgThresh);

	double upperThresh = caluclateCannyThresh(imgThresh);
	double lowerThresh = 0.1 * upperThresh;
	Canny(imgThresh, imgCanny, lowerThresh, upperThresh, 3, true);
	vector<Vec4i> lines;
	HoughLinesP(imgCanny, lines, 1, CV_PI / 180, upperThresh, 3, 1);

	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		line(imgOriginal, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
	}
	imshow("Img Canny", imgCanny);
	imshow("Hough", imgOriginal);
}

void ShadowDetection(Mat image)
{
	Mat imageShadow = image.clone();

	int iW = imageShadow.size().width;
	int iH = imageShadow.size().height;

	Mat imgTmp = imageShadow.clone();

	unsigned char* dataTmp = imgTmp.data;

	unsigned char* data = imageShadow.data;
	int channel = imageShadow.channels();
	for (int i = 5; i < iH - 5; i++) //
	{
		for (int j = 5; j < iW - 5; j++)
		{
			int B = data[channel * (i * iW + j)];
			int G = data[channel * (i * iW + j) + 1];
			int R = data[channel * (i * iW + j) + 2];
			float H;
			float S;
			float V;
			//Convert RGB to HSV
			float var_R = (R / 255.0);                    //RGB from 0 to 255
			float var_G = (G / 255.0);
			float var_B = (B / 255.0);

			float var_Min = MIN(MIN(var_R, var_G), var_B);  //Min. value of RGB
			float   var_Max = MAX(MAX(var_R, var_G), var_B);  //Max. value of RGB
			float   del_Max = var_Max - var_Min;       //Delta RGB value 

			V = var_Max;

			if (del_Max == 0)                     //This is a gray, no chroma...
			{
				H = 0;                              //HSV results from 0 to 1
				S = 0;
			}
			else                                    //Chromatic data...
			{
				S = del_Max / var_Max;

				float del_R = (((var_Max - var_R) / 6) + (del_Max / 2)) / del_Max;
				float del_G = (((var_Max - var_G) / 6) + (del_Max / 2)) / del_Max;
				float del_B = (((var_Max - var_B) / 6) + (del_Max / 2)) / del_Max;

				if (var_R == var_Max) H = del_B - del_G;
				else if (var_G == var_Max) H = (1 / 3) + del_R - del_B;
				else if (var_B == var_Max) H = (2 / 3) + del_G - del_R;

				if (H < 0) H += 1;
				if (H > 1) H -= 1;
			}

			//if(V>0.3 && V<0.85 && H<85 && S<0.15)
			//if(V>0.5 && V<0.95 &&  S<0.2)
			if (V > 0.3 && V < 0.95 && S < 0.2)
			{
				data[channel * (i * iW + j)] = 0;// dataTmp[channel*(i*iW+j)];
				data[channel * (i * iW + j) + 1] = 0;// dataTmp[channel*(i*iW+j)+1];
				data[channel * (i * iW + j) + 2] = 0;// dataTmp[channel*(i*iW+j)+2];
			}
			else
			{


				data[channel * (i * iW + j)] = 255;
				data[channel * (i * iW + j) + 1] = 255;
				data[channel * (i * iW + j) + 2] = 255;
			}
		}
	}


	//Find big area of shadow
	Mat imageGray;
	cvtColor(imageShadow, imageGray, COLOR_RGB2GRAY);

	int dilation_size = 2;
	Mat element = getStructuringElement(MORPH_ELLIPSE,
		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		Point(dilation_size, dilation_size));
	/// Apply the dilation operation to remove small areas
	dilate(imageGray, imageGray, element);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;


	/// Find contours
	findContours(imageGray, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE, Point(0, 0));


	vector<vector<Point> > contoursResult;


	for (int m = 0; m < contours.size(); m++)
	{
		int area = contourArea(contours[m]);
		drawContours(image, contours, m, Scalar(0, 0, 255), 2);
		if (area > 400 && area < iW * iH / 10)
		{
			contoursResult.push_back(contours[m]);
		}
	}
	imshow("Shadow test", image);
}

int main()
{
	string path = "Resources/TestBild2.png";
	//findHousesEdge(path);
	
	//houghTest(path);
	//vegetationTest(path);
	thresholdTests(path);

	//Mat img = imread(path);
	//ShadowDetection(img);

	waitKey(0);

	return 0;
}