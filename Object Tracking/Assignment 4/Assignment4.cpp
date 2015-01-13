#include <sstream>
#include <string>
#include <iostream>
#include <math.h>
#include <time.h>
#include <opencv/highgui.h>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv/cv.h>

using namespace cv;

void initialize(int &xcoordinate, int &ycoordinate, Mat threshold){									
	Mat temp;
	threshold.copyTo(temp);
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );
	if (hierarchy.size() > 0) {
		for (int index = 0; index >= 0; index = hierarchy[index][0]) {
			Moments moment = moments((cv::Mat)contours[index]);
			double area = moment.m00;
			xcoordinate = moment.m10/area;
			ycoordinate = moment.m01/area;
		}
	}
}


string intToString(int number){
	std::stringstream ss;
	ss << number;
	return ss.str();
}

void TotalDistanceCovered(int &xcoordinate, int &ycoordinate, long &totaldistance, Mat threshold){		
	Mat temp;
	int a=0, b=0;
	threshold.copyTo(temp);
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );
	if (hierarchy.size() > 0) {
		for (int index = 0; index >= 0; index = hierarchy[index][0]) {
			Moments moment = moments((cv::Mat)contours[index]);
			double area = moment.m00;
			a = moment.m10/area;
			b = moment.m01/area;
		}
		totaldistance += sqrt((double)(((xcoordinate-a)*(xcoordinate-a))+((ycoordinate-b)*(ycoordinate-b))));
		xcoordinate = a;
		ycoordinate = b;
	}
}

string floatToString(float number){
	std::ostringstream ss;
	ss << number;
	std::string s(ss.str());
	return s.c_str();
}

int main()
{
	Mat inputVideo;											
	Mat frame, frame_gray;									
	Mat HSV, edges, dst;									
	Mat threshold, threshold2;
	Mat descriptors_object, descriptors_scene;
	time_t start,end;										
	int totalTime=0;											
	time(&start);
	int x=0, y=0;
	long dist=0;
	double max_dist = 0; double min_dist = 100;
	bool init=false;
	Mat img_matches;
	int minHessian = 500;
	vector<vector<Point> > contours;						
	vector<Vec4i> hierarchy;								
	char waitKey;											
	VideoCapture capture;									
	namedWindow("Webcam", WINDOW_AUTOSIZE );			
	capture.open(0);										
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);				
	capture.set(CV_CAP_PROP_FRAME_HEIGHT,480);				
	while(1){
		if(capture.read(inputVideo)){
			cvtColor(inputVideo,HSV,COLOR_BGR2HSV);									
			inRange(HSV,Scalar(84,184,58),Scalar(119,256,148),threshold);			                            //Object is Blue Color
			erode(threshold, threshold, getStructuringElement(MORPH_RECT, Size(3,3)));
			dilate(threshold,threshold,getStructuringElement( MORPH_RECT,Size(8,8))); 
			dilate(threshold,threshold,getStructuringElement( MORPH_RECT,Size(8,8))); 
			threshold.copyTo(threshold2);
			findContours( threshold, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );   
			vector<vector<Point> > contours_poly( contours.size() );
			vector<Rect> boundRect( contours.size() );
			for( int i = 0; i < contours.size(); i++ )
			{
				approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );	
				boundRect[i] = boundingRect( Mat(contours_poly[i]) );			
			}
			for( int i = 0; i< contours.size(); i++ )
			{
				drawContours( inputVideo, contours_poly, i, Scalar(255, 0, 0), 2, 8, vector<Vec4i>(), 0, Point() );   	
				rectangle( inputVideo, boundRect[i].tl(), boundRect[i].br(), Scalar(255, 0, 0), 2, 8, 0 );				
			}
			imshow( "Webcam", inputVideo);
			if(!init){
				initialize(x, y, threshold2);					
				init=true;
			}
			else{
				TotalDistanceCovered(x,y, dist,threshold2);			
			}
			cv::initModule_nonfree();
			Mat img_object = imread( "C:/Users/Siddharth/Documents/Visual Studio 2010/Projects/Assignment 4/01.png", CV_LOAD_IMAGE_GRAYSCALE );
			Mat img_scene = inputVideo;
			
			if( !img_object.data || !img_scene.data )
			{ std::cout<< " --(!) Error reading images " << std::endl; return -1; }
		
			//-- Step 1: Detect the keypoints using SURF Detector
			
			SurfFeatureDetector detector( minHessian );

			std::vector<KeyPoint> keypoints_scene;
			std::vector<KeyPoint> keypoints_object;
			detector.detect( img_object, keypoints_object );
			detector.detect( img_scene, keypoints_scene );

			//-- Step 2: Calculate descriptors (feature vectors)
			SurfDescriptorExtractor extractor;

			extractor.compute( img_object, keypoints_object, descriptors_object );
			extractor.compute( img_scene, keypoints_scene, descriptors_scene );

			//-- Step 3: Matching descriptor vectors using FLANN matcher
			FlannBasedMatcher matcher;	
			std::vector<DMatch> matches;
			matcher.match( descriptors_object, descriptors_scene, matches );

			

			//-- Quick calculation of max and min distances between keypoints
			for( int i = 0; i < descriptors_object.rows; i++ )
			{ double dist = matches[i].distance;
			if( dist < min_dist ) min_dist = dist;
			if( dist > max_dist ) max_dist = dist;
			}

			printf("-- Max dist : %f \n", max_dist );
			printf("-- Min dist : %f \n", min_dist );

			//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
			std::vector< DMatch > good_matches;
		
			for( int i = 0; i < descriptors_object.rows; i++ )
			{ if( matches[i].distance < 3*min_dist )
			{ good_matches.push_back( matches[i]); }
			}

		    
			drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
            good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
            vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

			//-- Localize the object
			std::vector<Point2f> obj;
			std::vector<Point2f> scene;

			for( int i = 0; i < good_matches.size(); i++ )
			{
				//-- Get the keypoints from the good matches
				obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
				scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
			}

			Mat H = findHomography( obj, scene, CV_RANSAC );

			//-- Get the corners from the image_1 ( the object to be "detected" )
			std::vector<Point2f> obj_corners(4);
			obj_corners[0] = cvPoint(0,0); 
			obj_corners[1] = cvPoint( img_object.cols, 0 );
			obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); 
			obj_corners[3] = cvPoint( 0, img_object.rows );
			std::vector<Point2f> scene_corners(4);
	
			perspectiveTransform( obj_corners, scene_corners, H);

		    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
			line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
			line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
			line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
			line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );

			//-- Show detected matches
			imshow( "Good Matches & Object detection", img_matches );
			
			waitKey = cvWaitKey(30);							
			if(waitKey == 27)									
				break;
		}													
	}
	destroyWindow("Webcam");
	time(&end);
	totalTime = (int)(difftime(end, start));
	while(1){
		int distance_from_cam = 2;
		float speed = ((float)(dist*distance_from_cam*4))/((float)(640*totalTime*2));   
		//putText(inputVideo, "Speed: "+floatToString(speed)+"ft/sec",Point(50,50),2,1,Scalar(0,0,255),2);
		namedWindow("Output Window", WINDOW_AUTOSIZE);
		imshow( "Output Window", inputVideo);						
		waitKey = cvWaitKey(30);							
		if(waitKey == 27)									
			break;
	}
	return 0;
}
