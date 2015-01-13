#include <iostream>
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <opencv2/nonfree/features2d.hpp>

using namespace cv;

int main()
{
	Mat input = imread("temoc4.jpg");
	Mat input2;
	Mat desc2, desc1;
	SurfFeatureDetector detector(400);
	vector<KeyPoint> keypts1;
	vector<KeyPoint> keypts2;
	detector.detect(input, keypts1);
	SurfDescriptorExtractor extractor;
	extractor.compute( input, keypts1, desc1 );
	Mat output;
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	Mat inputVideo;											// Matrix to store input video
	Mat frame, frame_gray;									// Matrix to store grabbed frame
	Mat HSV, edges, dst;									// Matrix to store HSV converted frame
	Mat tracking, tracking2;								// Matrix to store frame with tracked red object
	vector<vector<Point> > contours;						// Each contour stored as vector of points
	vector<Vec4i> hierarchy;								// Contains information about image topology
	char checkKey;											// Used to exit the capturing process
	VideoCapture capture;									// Declate VideoCapture object
	namedWindow("CameraFeed", WINDOW_AUTOSIZE );			// Window to display the video being captured
	capture.open(0);										// Open Webcam
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);				// Set frame width
	capture.set(CV_CAP_PROP_FRAME_HEIGHT,480);				// Set frame height
	while(1){
		if(capture.read(inputVideo)){
			input2 = inputVideo;
			detector.detect(input2, keypts2);
			extractor.compute( input2, keypts2, desc2 );
			matcher.match( desc1, desc2, matches );
			double max_dist = 0; double min_dist = 100;
			// Calculation of minimum and maximum distances between keypoints
			for( int i = 0; i < desc1.rows; i++ )
			{ double dist = matches[i].distance;
			if( dist < min_dist )
				min_dist = dist;
			if( dist > max_dist )
				max_dist = dist;
			}

			// Getting the good matches
			std::vector< DMatch > good_matches;

			for( int i = 0; i < desc1.rows; i++ )
			{ if( matches[i].distance <= 3*min_dist )
			{ good_matches.push_back( matches[i]); }
			}

			std::vector< Point2f >  obj;
			std::vector< Point2f >  scene;

			for( unsigned int i = 0; i < good_matches.size(); i++ )
			{
				// Getting the keypoints from the good matches
				obj.push_back( keypts1[ good_matches[i].queryIdx ].pt );
				scene.push_back( keypts2[ good_matches[i].trainIdx ].pt );
			}

			Mat H = findHomography( obj, scene, CV_RANSAC );

			// Derive the corners to make bounding box
			std::vector< Point2f > obj_corners(4);
			obj_corners[0] = cvPoint(0,0);
			obj_corners[1] = cvPoint( input.cols, 0 );
			obj_corners[2] = cvPoint( input.cols, input.rows );
			obj_corners[3] = cvPoint( 0, input.rows );
			std::vector< Point2f > scene_corners(4);
			std::vector< Point2f > Object(2);

			perspectiveTransform( obj_corners, scene_corners, H);

			// Draw a box around the object by joining the corners
			/*
			line( input2, scene_corners[0] , scene_corners[1], Scalar(255,0,0), 2 );
			line( input2, scene_corners[1] , scene_corners[2], Scalar(255,0,0), 2 );
			line( input2, scene_corners[2] , scene_corners[3], Scalar(255,0,0), 2 );
			line( input2, scene_corners[3] , scene_corners[0], Scalar(255,0,0), 2 );
			 */
			int width = scene_corners[2].x-scene_corners[0].x;
			int height = scene_corners[2].y-scene_corners[0].y;
			Object[0].x=scene_corners[0].x-width;
			Object[1].x=scene_corners[2].x+width;
			Object[0].y=scene_corners[0].y-height;
			Object[1].y=scene_corners[2].y+2*height;
			rectangle( input2, Object[0], Object[1], cv::Scalar(0,0,255), 2);
			imshow( "CameraFeed", input2);
			checkKey = cvWaitKey(30);							// cvWaitKey called to give highgui time to process the captured image
			if(checkKey == 27)									// Terminate capture if Escape key pressed
				break;
		}														// Program needs to be terminated manually since otherwise there wouldn't be enough time to monitor the played video, the grabbed frame and marked frame in real time.
	}
	return 0;
}