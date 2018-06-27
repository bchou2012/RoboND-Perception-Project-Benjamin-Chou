#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

import time #delay to take screen shots

# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:
    
    #display raw input
    pcl_raw.publish(pcl_msg)
    
    # TODO: Convert ROS msg to PCL data
    pcl_data = ros_to_pcl(pcl_msg)
    
    # TODO: Statistical Outlier Filtering
    outlier_filter = pcl_data.make_statistical_outlier_filter()
    
    #trial and error values
    mean_k = 25
    thresh_scale = 0.001
    
    outlier_filter.set_mean_k(mean_k)
    outlier_filter.set_std_dev_mul_thresh(thresh_scale)
    
    pcl_o_f = outlier_filter.filter()
    # TODO: Voxel Grid Downsampling
   
    vox = pcl_o_f.make_voxel_grid_filter()
    LEAF_SIZE = 0.005 #value from testing in class lab
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    
    pcl_vox = vox.filter()
   
    # TODO: PassThrough Filter
    #X axis passthrough
    passthrough_x = pcl_vox.make_passthrough_filter()
    
    #all max_min values determined trial and error, should have used measure function in Rviz/Gazebo
    x_min = 0.33
    x_max = 0.88
 
    passthrough_x.set_filter_field_name('x')
    passthrough_x.set_filter_limits(x_min, x_max)
    
    passthrough = passthrough_x.filter()
    
    #Y axis passthrough
    passthrough_y = passthrough.make_passthrough_filter()
    
    y_min = -0.46
    y_max = 0.46
    
    passthrough_y.set_filter_field_name('y')
    passthrough_y.set_filter_limits(y_min, y_max)
    passthrough = passthrough_y.filter()

    #Z axis passthrough
    passthrough_z = passthrough.make_passthrough_filter()
    
    z_min = 0.605
    z_max = 0.9
    
    passthrough_z.set_filter_field_name('z')
    passthrough_z.set_filter_limits(z_min, z_max)
    passthrough = passthrough_z.filter()
    
    # TODO: RANSAC Plane Segmentation
    seg = passthrough.make_segmenter()
    
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    
    #value determined through trial and error
    max_distance_table = 0.008
    seg.set_distance_threshold(max_distance_table)
    inliers, outliers = seg.segment()

    # TODO: Extract inliers and outliers
    
    cloud_objects = passthrough.extract(inliers, negative=True)
    cloud_table = passthrough.extract(inliers, negative=False)

    # TODO: Euclidean Clustering
    #color information is not required
    euclidean_objects = XYZRGB_to_XYZ(cloud_objects)
    
    #initialize KDtree
    tree = euclidean_objects.make_kdtree()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    ec = euclidean_objects.make_EuclideanClusterExtraction()
    #clustering values determined through trial and error
    ec.set_ClusterTolerance(0.01)
    ec.set_MinClusterSize(25)
    ec.set_MaxClusterSize(5000)
    
    #set euclidian cluster search method
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()
    
    #variable for coloring the clusters
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list =[]
    
    #assign color to each cluster
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([euclidean_objects[indice][0],
                euclidean_objects[indice][1],
                euclidean_objects[indice][2],
                rgb_to_float(cluster_color[j])])
     
    #Convert cluster back to
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    
    # TODO: Convert PCL data to ROS messages
    #use pcl_to_ros() function
    
    ros_filtered = pcl_to_ros(pcl_o_f)
    ros_vox = pcl_to_ros(pcl_vox)
    ros_passthrough = pcl_to_ros(passthrough)
    ros_objects = pcl_to_ros(cloud_objects)
    ros_table = pcl_to_ros(cloud_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # TODO: Publish ROS messages

    stat_filter.publish(ros_filtered)
    pub_vox.publish(ros_vox)
    pub_pass.publish(ros_passthrough)
    pub_objects.publish(ros_objects)
    pub_table.publish(ros_table)
    pub_cloud.publish(ros_cluster_cloud)

# Exercise-3 TODOs:

    #intialize detected objects variables. 
    detected_objects_labels = []
    detected_objects = []
    
    #Looping through the cluster list:
    for index, pts_list in enumerate(cluster_indices):
        pcl_cluster = cloud_objects.extract(pts_list)
        pcl_to_ros_cluster = pcl_to_ros(pcl_cluster)
        
        #Construct the color histogram uses HSV instead of RGB 
        chists = compute_color_histograms(pcl_to_ros_cluster, using_hsv= True)
        #collect the surface normals
        normals = get_normals(pcl_to_ros_cluster)
        nhists = compute_normal_histograms(normals)
        #concatenate the HSV and surface normals
        feature = np.concatenate((chists, nhists))
        
        #make the prediction based on the combined HSV and surface normal information
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        #match the label
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)
        
        label_pos = list(euclidean_objects[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label, label_pos, index))
        
        #fill detected objects and label variables
        det_ob = DetectedObject()
        det_ob.label = label
        det_ob.cloud = pcl_to_ros_cluster
        detected_objects.append(det_ob)
        
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    #publish detected objects
    detected_objects_pub.publish(detected_objects)

    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables
    #initialize YAML variables
    test_scene_num = Int32()
    test_scene_num.data = 3 #1,2,or 3
    object_name = String()
    arm_name = String()
    pick_pose = Pose()
    place_pose = Pose()
    output_name = 'output_3.yaml'#output_[scene].yaml
    yaml_output = []
    
    #Intialize parameter lists
    labels = []
    centroids = []
    
    #Retrieve object and dropbox parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')
    
    #Turns dropbox parameters into dictionary callable by group
    dropbox_by_group = {}
    for i in dropbox_param:
        dropbox_by_group[i['group']] = {'position': i['position'], 'name': i['name']}
        
    #Create centroids for labels
    for object in object_list:
        labels.append(object.label)
        points_arr = ros_to_pcl(object.cloud).to_array()
        centroids.append(np.mean(points_arr, axis=0)[:3])
    
    #Look through list
    for i in object_list_param:
        
        #define object name and group
        object_name.data = i['name']
        object_group = i['group']
        
        #loop object through the labels
        for j, k in enumerate(labels):
            if object_name.data == k:
                    
                    #define object centroid
                    pick_pose.position.x = np.asscalar(centroids[j][0])
                    pick_pose.position.y = np.asscalar(centroids[j][1])
                    pick_pose.position.z = np.asscalar(centroids[j][2])
                    
                    #define matching container location
                    place_pose.position.x = dropbox_by_group[object_group]['position'][0]
                    place_pose.position.y = dropbox_by_group[object_group]['position'][1]
                    place_pose.position.z = dropbox_by_group[object_group]['position'][2]
                    
                    #definte matching container by color
                    arm_name.data = dropbox_by_group[object_group]['name']
                    
                    #construct YAML data
                    yaml_entry = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
                    #append YAML data
                    yaml_output.append(yaml_entry)
        
        #pick_place commented out due to not attempting challenge            
        '''            
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
        '''
    # TODO: Output your request parameters into output yaml file
    
    send_to_yaml(output_name, yaml_output)
    print("YAML written")
    time.sleep(60) #delay to stop the loop and write another YAML
    return
    
if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    pcl_raw = rospy.Publisher("/pcl_raw", PointCloud2, queue_size=1) #raw input
    pcl_pub = rospy.Publisher("/pcl_pub", PointCloud2, queue_size=1)
    stat_filter = rospy.Publisher("/stat_filter", PointCloud2, queue_size=1) #noise filter
    pub_vox = rospy.Publisher("/pub_vox", PointCloud2, queue_size=1) #voxel downsampling
    pub_pass = rospy.Publisher("/pub_pass", PointCloud2, queue_size=1) #passthrough filter
    pub_objects = rospy.Publisher("/pub_objects", PointCloud2, queue_size=1) #objects only
    pub_table = rospy.Publisher("/pub_table", PointCloud2, queue_size=1) #table
    pub_cloud = rospy.Publisher("/pub_cloud", PointCloud2, queue_size=1) #cluster mask grouping
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    arm_mover_pub = rospy.Publisher("pr2/world_joint_controller/command", Float64, queue_size=10) #challenge not undertaken
    
    # Initialize color_list
    get_color_list.color_list = []
    # TODO: Load Model From disk
    model = pickle.load(open('model_3.sav', 'rb')) #file format model_[no].sav
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']
    
    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
