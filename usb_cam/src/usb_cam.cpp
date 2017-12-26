#include<ros/ros.h>
#include<image_transport/image_transport.h>
#include<opencv2/opencv.hpp>
#include<cv_bridge/cv_bridge.h>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "openCamera");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::Publisher pub_rgb = it.advertise("camera/rgb/image_raw", 1);
  
  sensor_msgs::ImagePtr msg_rgb;
  cv::VideoCapture cap(0);
  cv::Mat img_rgb;
  cv::Mat img_bgr;
  while (ros::ok()) {
    cap >> img_bgr;
    if (!img_bgr.empty())
    {
	cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);
       
      
	msg_rgb = cv_bridge::CvImage(std_msgs::Header(), "rgb8", img_rgb).toImageMsg();
        pub_rgb.publish(msg_rgb);
       
    }
  }
  return 0;
}
