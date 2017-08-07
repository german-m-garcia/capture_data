#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <ros/package.h>
#include <visualization_msgs/Marker.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/integral_image_normal.h>
#include <image_geometry/pinhole_camera_model.h>

#include <pcl/surface/concave_hull.h>
#include <pcl/filters/project_inliers.h>


#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <image_transport/image_transport.h>


#include "tf/tf.h"

#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/PointCloud2.h>
#include <thread>
#include <chrono>
#include <string>


using namespace sensor_msgs;
using namespace message_filters;

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointCloud<PointT>::Ptr PointCloudPtr;
typedef pcl::PointCloud<PointT>::ConstPtr PointCloudConstPtr;

ros::Publisher pub,pub_subsampled;
ros::Publisher vis_pub;
message_filters::Subscriber<Image>* img_sub_;
message_filters::Subscriber<Image>* depth_sub_;
message_filters::Subscriber<CameraInfo>* camera_info_sub_;
message_filters::Subscriber<sensor_msgs::PointCloud2>* cloud_sub_;
typedef sync_policies::ApproximateTime<Image, Image> MySyncPolicy;
typedef sync_policies::ApproximateTime<Image, Image,sensor_msgs::PointCloud2> MySyncCloudPolicy;
Synchronizer<MySyncPolicy> *sync_;
Synchronizer<MySyncCloudPolicy> *sync_cloud_;
CameraInfoPtr camera_info_;
image_geometry::PinholeCameraModel cam_model_; // init cam_model
bool not_init_ = true;

std::queue<cv::Mat> images_queue_, depth_queue_, masks_queue_;

std::string output_path;
int images_index = 0;


void save_to_disk(){
	while(true){

		if(!masks_queue_.empty() || !images_queue_.empty()|| !depth_queue_.empty()){
			cv::Mat mask = masks_queue_.front();
			cv::Mat img = images_queue_.front();
			cv::Mat depth = depth_queue_.front();
			masks_queue_.pop();
			images_queue_.pop();
			depth_queue_.pop();

			std::string filename = "/mask_"+std::to_string(images_index)+".png";
			std::string filename_img = "/img_"+std::to_string(images_index)+".png";
			std::string filename_depth = "/depth_"+std::to_string(images_index)+".xml";
			std::cout <<"saving file to "<<output_path+filename<<std::endl;
			cv::imwrite(output_path+filename,mask);
			cv::imwrite(output_path+filename_img,img);
			//cv::imwrite(output_path+filename_depth,depth);
			cv::FileStorage fp(output_path+filename_depth, cv::FileStorage::WRITE);
			fp << "depth" << depth;
			fp.release();
			images_index++;
		}
		else{
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
	}

}

void camera_info_callback(CameraInfoConstPtr cam_info){
	*camera_info_ = *cam_info;
	if(not_init_){
		not_init_=false;
		cam_model_.fromCameraInfo(camera_info_); // fill cam_model with CameraInfo
	}
}

void subsample(PointCloudConstPtr  src_cloud, pcl::PointCloud<PointT>::Ptr& cloud_filtered){
	 // Create the filtering object: downsample the dataset using a leaf size of 1cm
	pcl::VoxelGrid<PointT> vg;

	vg.setInputCloud (src_cloud);
	vg.setLeafSize (0.005f, 0.005f, 0.005f);
	vg.filter (*cloud_filtered);
	std::vector<int> indices;
	pcl::removeNaNFromPointCloud(*cloud_filtered, *cloud_filtered, indices);
	cloud_filtered->points.pop_back();
}

/** \brief Compute normals using integral images.
 * \param cloud_input cloud containing xyz point information
 * \param cloud_normals cloud where normals are stored
 */
void computeNormals(PointCloudConstPtr cloud_input, pcl::PointCloud<pcl::Normal>::Ptr cloud_normals, int support_size = 5)
{

	pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> normal_estimator;
	pcl::search::Search<pcl::PointXYZRGB>::Ptr tree = boost::shared_ptr<
				pcl::search::Search<pcl::PointXYZRGB> >(
				new pcl::search::KdTree<pcl::PointXYZRGB>);
	normal_estimator.setSearchMethod(tree);
	normal_estimator.setInputCloud(cloud_input);
	normal_estimator.setRadiusSearch(0.03);
	//normal_estimator.setKSearch(20);
	normal_estimator.compute(*cloud_normals);

}

void pointToPixel(PointT& p,cv::Point2d& uv){


	cv::Point3d pt_cv(p.x,p.y, p.z);//init Point3d
	uv = cam_model_.project3dToPixel(pt_cv); // project 3d point to 2d point
}

void extract_mask( PointCloudPtr colours_cloud, cv::Mat& mask){

	cv::Point2d uv;
	for(PointT& p : *colours_cloud){

		pointToPixel(p,uv);
		mask.at<uint8_t>(uv) = 255;
	}
	cv::floodFill(mask, uv, cv::Scalar(255));
}

void get_closest_cluster(PointCloudPtr src_cloud,std::vector<pcl::PointIndices>& cluster_indices,Eigen::Vector4f& centroid, PointCloudPtr colours_cloud){
	double dist= 0., min_dist = 99999.;
	int selected_index = 0, i = 0;
	pcl::PointXYZ p1(centroid[0],centroid[1],centroid[2]);
	//look for the cluster closer to the center of the plane
	for(pcl::PointIndices& indices : cluster_indices){
		Eigen::Vector4f cluster_centroid;
		pcl::compute3DCentroid(*src_cloud, indices, cluster_centroid);
		pcl::PointXYZ p2(cluster_centroid[0],cluster_centroid[1],cluster_centroid[2]);
		dist = pcl::euclideanDistance(p1,p2);
		if (dist < min_dist){
			selected_index = i;
			min_dist = dist;
		}
		i++;

	}
	//colour it
	{
		int points = 0;
		int r = 255;
		int g = 0;
		int b = 0;

		for (std::vector<int>::const_iterator pit = cluster_indices[selected_index].indices.begin (); pit != cluster_indices[selected_index].indices.end (); ++pit){
			colours_cloud->points.push_back (src_cloud->points[*pit]); //*
			colours_cloud->points[points].r = r;
			colours_cloud->points[points].g = g;
			colours_cloud->points[points].b = b;
			points++;
		}
	}

}

void split_clusters(PointCloudConstPtr src_cloud, std::vector<pcl::PointIndices>& cluster_indices,PointCloudPtr dst_cloud){



	dst_cloud->header = src_cloud->header;
	int points = 0;
	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
	{
		int r = rand() % 255;
		int g = rand() % 255;
		int b = rand() % 255;
		for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit){
			dst_cloud->points.push_back (src_cloud->points[*pit]); //*
			dst_cloud->points[points].r = r;
			dst_cloud->points[points].g = g;
			dst_cloud->points[points].b = b;
			points++;
		}


	std::cout << "PointCloud representing the Cluster: " << dst_cloud->points.size () << " data points." << std::endl;

	}
	dst_cloud->width = points;//cloud_cluster->points.size ();
	dst_cloud->height = 1;
	dst_cloud->is_dense = true;
}

void euclidean_clustering(PointCloudPtr src_cloud,Eigen::Vector4f& centroid, PointCloudPtr colours_cloud){
	pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
	PointCloudPtr tmp(new PointCloud);

	for(PointT& p : *src_cloud){
		if( !std::isnan(p.x))
			tmp->points.push_back(p);
	}
	tree->setInputCloud (tmp);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<PointT> ec;
	ec.setClusterTolerance (0.02); // 2cm
	ec.setMinClusterSize (100);
	ec.setMaxClusterSize (25000);
	ec.setSearchMethod (tree);
	ec.setInputCloud (tmp);
	ec.extract (cluster_indices);
	if(cluster_indices.size()==0)
		return;
	get_closest_cluster(tmp,cluster_indices,centroid,colours_cloud);
	//split_clusters(tmp,cluster_indices,colours_cloud);
}


void hull(PointCloudPtr cloud,pcl::PointIndices::Ptr inliers,pcl::ModelCoefficients::Ptr coefficients){
	PointCloudPtr cloud_projected (new pcl::PointCloud<PointT>);
	// Project the model inliers
	pcl::ProjectInliers<PointT> proj;
	proj.setModelType (pcl::SACMODEL_PLANE);
	proj.setIndices (inliers);
	proj.setInputCloud (cloud);
	proj.setModelCoefficients (coefficients);
	proj.filter (*cloud_projected);
	std::cerr << "PointCloud after projection has: "
			<< cloud_projected->points.size () << " data points." << std::endl;

	// Create a Concave Hull representation of the projected inliers
	pcl::PointCloud<PointT>::Ptr cloud_hull (new pcl::PointCloud<PointT>);
	pcl::ConcaveHull<PointT> chull;
	chull.setInputCloud (cloud_projected);
	chull.setAlpha (0.1);
	chull.reconstruct (*cloud_hull);
	pub_subsampled.publish(*cloud_hull);
}

void publish_plane_normal(Eigen::Vector4f& centroid,pcl::ModelCoefficients::Ptr coefficients,std::string& frame_id){
	visualization_msgs::Marker marker;
	marker.header.frame_id = frame_id;
	marker.header.stamp = ros::Time();

	marker.id = 0;
	marker.type = visualization_msgs::Marker::ARROW;
	marker.action = visualization_msgs::Marker::ADD;
	marker.pose.position.x =centroid[0];
	marker.pose.position.y = centroid[1];
	marker.pose.position.z = centroid[2];
	marker.pose.orientation.x = coefficients->values[0];//0.0;
	marker.pose.orientation.y = coefficients->values[1];//0.0;
	marker.pose.orientation.z = coefficients->values[2];//0.0;
	marker.pose.orientation.w = 1.0;


//	tf::Quaternion qori(marker.pose.orientation.x,marker.pose.orientation.y,marker.pose.orientation.z,marker.pose.orientation.w);
//	tf::Quaternion qtf = tf::createQuaternionFromRPY (0,3.141592/4.,0);//3.141592/4.
//	qori =qtf*qori;
//	qori = qori.normalize();
//	marker.pose.orientation.x = qori.x();
//	marker.pose.orientation.y = qori.y();
//	marker.pose.orientation.z = qori.z();
//	marker.pose.orientation.w = qori.w();

	marker.scale.x = 0.1;
	marker.scale.y = 0.02;
	marker.scale.z = 0.02;
	marker.color.a = 1.0; // Don't forget to set the alpha!
	marker.color.r = 1.0;
	marker.color.g = 0.0;
	marker.color.b = 0.0;


	vis_pub.publish( marker );
}

void sac_plane(PointCloudPtr cloud){
	pcl::PointCloud<PointT>::Ptr cloud_support_plane_ptr_ (new pcl::PointCloud<PointT>),cloud_objects_ptr_ (new pcl::PointCloud<PointT>),
			colours_cloud (new pcl::PointCloud<PointT>);
	pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	// Create the segmentation object
	pcl::SACSegmentation<PointT> seg;
	// Optional
	seg.setOptimizeCoefficients (true);
	// Mandatory
	seg.setModelType (pcl::SACMODEL_PLANE);
	seg.setMethodType (pcl::SAC_RANSAC);
	seg.setDistanceThreshold (0.02);
	seg.setInputCloud (cloud);
	seg.segment (*inliers, *coefficients);





	/// create clouds of support plane and objects /////////////////////////
	pcl::ExtractIndices<PointT> extract_indices;
	extract_indices.setInputCloud(cloud);
	extract_indices.setIndices(inliers);
	extract_indices.setNegative(false);
	extract_indices.filter(*cloud_support_plane_ptr_);
	extract_indices.setNegative(true);
	extract_indices.filter(*cloud_objects_ptr_);
	//compute the centroid
	Eigen::Vector4f centroid;
	pcl::compute3DCentroid(*cloud_support_plane_ptr_, centroid);

	publish_plane_normal(centroid,coefficients,cloud->header.frame_id);


	//hull(cloud,inliers,coefficients);
	euclidean_clustering(cloud_objects_ptr_, centroid,colours_cloud);
	if(colours_cloud->points.size()==0)
		return;
	pub.publish(*cloud_support_plane_ptr_);
	colours_cloud->header = cloud->header;
	colours_cloud->height = cloud->height;
	colours_cloud->width = cloud->width;
	//pub_subsampled.publish(*colours_cloud);
	cv::Mat mask(cloud->height,cloud->width, CV_8UC1, cv::Scalar(0) );
	extract_mask(colours_cloud, mask);
	cv::imshow("mask", mask);
	cv::waitKey(1);
	masks_queue_.push(mask);

}

void planes (PointCloudConstPtr cloud) {

	pcl::PointCloud<PointT>::Ptr cloud_support_plane_ptr_ (new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr cloud_objects_ptr_ (new pcl::PointCloud<PointT>);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
	/// fit plane through horizontal points ////////////////////////////////
	pcl::SACSegmentationFromNormals<PointT, pcl::Normal> plane_seg;
	plane_seg.setOptimizeCoefficients(true);
	plane_seg.setModelType( pcl::SACMODEL_NORMAL_PLANE);
	plane_seg.setMethodType(pcl::SAC_RANSAC);
	plane_seg.setMaxIterations(250);
	plane_seg.setDistanceThreshold(0.05);
	plane_seg.setAxis(Eigen::Vector3f(1.0f, 1.0f, 0.0f));
	plane_seg.setEpsAngle(3.14);

	plane_seg.setInputCloud(cloud);
	computeNormals(cloud, cloud_normals,5);
	plane_seg.setInputNormals(cloud_normals);

	pcl::PointIndices::Ptr inliers_cluster_ptr(new pcl::PointIndices());
	pcl::ModelCoefficients::Ptr model_coefficients_ptr(new pcl::ModelCoefficients());
	plane_seg.segment(*inliers_cluster_ptr, *model_coefficients_ptr);

	/// create clouds of support plane and objects /////////////////////////
	pcl::ExtractIndices<PointT> extract_indices;
	extract_indices.setInputCloud(cloud);
	extract_indices.setIndices(inliers_cluster_ptr);
	extract_indices.setNegative(false);
	extract_indices.filter(*cloud_support_plane_ptr_);
	extract_indices.setNegative(true);
	extract_indices.filter(*cloud_objects_ptr_);
    
	pub.publish(*cloud_support_plane_ptr_);
	pub_subsampled.publish(*cloud_objects_ptr_);


  
 
  return;
}



void table_segment(PointCloudConstPtr cloud){
  
  // Create the filtering object: downsample the dataset using a leaf size of 1cm
  pcl::VoxelGrid<PointT> vg;
  pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>);
  pcl::PointCloud<pcl::Normal> cloud_normals;
  vg.setInputCloud (cloud);
  vg.setLeafSize (0.005f, 0.005f, 0.005f);
  vg.filter (*cloud_filtered);
  cloud_filtered->header = cloud->header;
  std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl; //*


  PointCloudPtr cloud_f (new pcl::PointCloud<PointT>);
  PointCloudPtr colours_cloud (new pcl::PointCloud<PointT>);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);




  // Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentation<PointT> seg;

  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointCloud<PointT>::Ptr cloud_plane (new pcl::PointCloud<PointT> ());
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (100);
  seg.setDistanceThreshold (0.05);


  int i=0, nr_points = (int) cloud_filtered->points.size ();
  while (cloud_filtered->points.size () > 0.9 * nr_points)
  {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (cloud_filtered);
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0)
    {
      std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
      break;
    }

    // Extract the planar inliers from the input cloud
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud (cloud_filtered);
    extract.setIndices (inliers);
    extract.setNegative (false);

    // Get the points associated with the planar surface
    extract.filter (*cloud_plane);
    pub.publish(*cloud_plane);
    std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

    // Remove the planar inliers, extract the rest
    extract.setNegative (true);
    extract.filter (*cloud_f);
    *cloud_filtered = *cloud_f;
  }

  PointCloudPtr tmp_cloud(new pcl::PointCloud<PointT>);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cloud_filtered, *cloud_filtered, indices);
  std::cout <<"cloud_filtered size()="<<cloud_filtered->points.size()<<std::endl;
  if(cloud_filtered->points.size() < 10)
	  return;
  cloud_filtered->points.pop_back();

//  for(PointT& po : *cloud_filtered){
//  	std::cout<<" p.x, p.y, p.z="<<po.x<<" "<<po.y<<" "<< po.z<<std::endl;
//  }
  
  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
  tree->setInputCloud (cloud_filtered);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<PointT> ec;
  ec.setClusterTolerance (0.02); // 2cm
  ec.setMinClusterSize (10);
  ec.setMaxClusterSize (25000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_filtered);
  ec.extract (cluster_indices);
  
  split_clusters(cloud_filtered,cluster_indices,colours_cloud);

  //colours_cloud->header = cloud_filtered->header;

  return;
 

	
}

void sync_short_callback(const ImageConstPtr& colour_image,
				const ImageConstPtr& depth_image) {
	cv_bridge::CvImagePtr cv_ptr;
	cv_bridge::CvImagePtr cv_depth_ptr;




	//convert the synchronized images messages to OpenCV Mats
	try {
		cv_ptr = cv_bridge::toCvCopy(colour_image,
				sensor_msgs::image_encodings::BGR8);

		cv_depth_ptr = cv_bridge::toCvCopy(depth_image,
				sensor_msgs::image_encodings::TYPE_32FC1);

	} catch (cv_bridge::Exception& e) {
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}
	cv::Mat outMat;
	cv::Mat src_ = cv_ptr->image;
	cv::Mat depth_ = cv_depth_ptr->image;
	cv::medianBlur(depth_, depth_, 3);
	cv::medianBlur(depth_, depth_, 5);
	cv::imshow("short callback", src_);
	cv::waitKey(1);

}

void sync_cloud_callback(const ImageConstPtr& colour_image,
				const ImageConstPtr& depth_image, const sensor_msgs::PointCloud2ConstPtr& cloud_in) {
	cv_bridge::CvImagePtr cv_ptr;
	cv_bridge::CvImagePtr cv_depth_ptr;
	PointCloudPtr input_cloud(new PointCloud);



	//convert the synchronized images messages to OpenCV Mats
	try {
		cv_ptr = cv_bridge::toCvCopy(colour_image,
				sensor_msgs::image_encodings::BGR8);

		cv_depth_ptr = cv_bridge::toCvCopy(depth_image,
				sensor_msgs::image_encodings::TYPE_32FC1);

	} catch (cv_bridge::Exception& e) {
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}
	cv::Mat outMat;
	cv::Mat src_ = cv_ptr->image;
	cv::Mat depth_ = cv_depth_ptr->image;
	cv::imshow("short callback", src_);
	cv::waitKey(1);
	//convert the point cloud to the pcl format
	pcl::fromROSMsg(*cloud_in, *input_cloud);
	sac_plane(input_cloud);
	images_queue_.push(src_);
	depth_queue_.push(depth_);

}

void cloud_callback (const /*sensor_msgs::PointCloud2ConstPtr&*/PointCloudConstPtr cloud_in){
     ROS_INFO_STREAM("inside callback header frame_id="<< cloud_in->header.frame_id);
     PointCloudPtr input_cloud, tmp(new pcl::PointCloud<PointT>);


     input_cloud.reset(new PointCloud(*cloud_in));
     sac_plane(input_cloud);

     //std::vector<int> indices;
	 //pcl::removeNaNFromPointCloud(*input_cloud, *tmp, indices);

//     subsample(input_cloud,tmp);
//     input_cloud=tmp;
//
//     input_cloud->header = cloud_in->header;
     //pub_subsampled.publish(*tmp);
     //planes(input_cloud);

}


int main (int argc, char** argv) {
     ros::init (argc, argv, "cloud_sub");
     ros::NodeHandle nh;
     ros::Rate loop_rate(30);
     ros::Subscriber sub;
     std::string camera_source_, depth_source_, cloud_source_, record_path_;
     camera_info_ = CameraInfoPtr(new CameraInfo);
     std::thread save_disk_thread (save_to_disk);

     sub = nh.subscribe ("/camera/depth_registered/camera_info", 10, camera_info_callback);
     //get camera source
     nh.param<std::string>("rgb_camera_source", camera_source_,
			"/camera/rgb/image_raw");

	//get depth map source

     nh.param<std::string>("depth_source", depth_source_,
			"/camera/depth_registered/image_raw");

     nh.param<std::string>("cloud_source", cloud_source_,
    			"/camera/depth_registered/points");

     nh.param<std::string>("output_path", output_path,
         			"/home/martin/Datasets/testrecorder");


     //subscribers
     img_sub_ = new message_filters::Subscriber<Image>(nh, camera_source_,
     						1);
     depth_sub_ = new message_filters::Subscriber<Image>(nh, depth_source_,
     				1);

     cloud_sub_ = new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, cloud_source_,
				1);
     sync_cloud_ = new Synchronizer<MySyncCloudPolicy>(MySyncCloudPolicy(500), *img_sub_,
     						*depth_sub_, *cloud_sub_);
     sync_cloud_->registerCallback(
     				boost::bind(sync_cloud_callback,  _1,
     						_2, _3));

     pub = nh.advertise<PointCloud> ("planes", 10);
     vis_pub = nh.advertise<visualization_msgs::Marker>( "normal_plane", 0 );
     pub_subsampled = nh.advertise<PointCloud> ("subsampled", 10);

     ros::spin();
 }
