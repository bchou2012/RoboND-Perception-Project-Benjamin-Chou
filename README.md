Udacity Perception Project Writeup
=

This project goal is to implement a perception pipeline, from extracting and training models for recognition, implementing filters and image processing, to identifying simulated RGB objects and outputting their identities and locations.

## Feature Extraction and SVM Model Training

First we extract features from a data set and train our project to recognize distinct models. 

**Feature Extraction**

Using the supplied sensor stick code from the class exercises, we extract features using `capture_features.py`.  

<img src="https://github.com/bchou2012/RoboND-Perception-Project-Benjamin-Chou/blob/master/images/capture_training.png?raw=true" alt="Feature Extraction" width="640px">

Initially we used 100 captures per model, but during later testing with the perception pipeline with world_3 the identification of objects was inconsistent. This resulted in rerunning the feature extraction with 500 captures, which provided more accurate object identification but much longer extraction run time. See `capture_features_project_1.py,
capture_features_project_2.py, capture_features_project_3.py` in the `/sensor_stick_scripts/` folder.  

**SVM Training**

Each training set for each world was then trained with `train_svm.py`, included in the `/sensor_stick_scripts/` folder with the following identification matrix results:

*World_1* 

<img src="https://github.com/bchou2012/RoboND-Perception-Project-Benjamin-Chou/blob/master/images/Confusion_1.png?raw=true" alt="World_1" width="480px">
<img src="https://github.com/bchou2012/RoboND-Perception-Project-Benjamin-Chou/blob/master/images/Normalized_confusion_1.png?raw=true" alt="World_1 Normalized" width="480px">

*World_2* 

<img src="https://github.com/bchou2012/RoboND-Perception-Project-Benjamin-Chou/blob/master/images/Confusion_2.png?raw=true" alt="World_2" width="480px">
<img src="https://github.com/bchou2012/RoboND-Perception-Project-Benjamin-Chou/blob/master/images/Normalized_confusion_2.png?raw=true" alt="World_2 Normalized" width="480px">

*World_3* 

<img src="https://github.com/bchou2012/RoboND-Perception-Project-Benjamin-Chou/blob/master/images/Confusion_3.png?raw=true" alt="World_3" width="480px">
<img src="https://github.com/bchou2012/RoboND-Perception-Project-Benjamin-Chou/blob/master/images/Normalized_confusion_3.png?raw=true" alt="World_3 Normalized" width="480px">

Through trial and error, for the SVC kernel in `/sensor_stick_scripts/`, line 66 was set to 
`clf = svm.SVC(kernel='linear')` The other kernel options `poly, rbf, sigmoid` did not produce better results for this project

For the purposes of this project, having over 90% success rate with all objects between the 3 scenarios is acceptable. With our training information saved, we move onto the perception pipeline. 

The `compute_color_histograms()` function in the capture\_features files was set to `using_hsv=True` for reasons detailed in the object recognition writeup. 

## Perception Pipeline

Our raw data appears as below: 

<img src="https://github.com/bchou2012/RoboND-Perception-Project-Benjamin-Chou/blob/master/images/raw%20image.png?raw=true" alt="Raw image with noise" width="640px">


**Outlier Removal Filter**

The noise must be filtered out first using the `make_statistical_outlier_filter()`function. The filter looks at a number of collected points and removes any points whose distance from the other points exceeds a threshold distance. 

<img src="https://github.com/bchou2012/RoboND-Perception-Project-Benjamin-Chou/blob/master/images/statisical_filter.png?raw=true" alt="Statistical Filter Applied" width="640px">

Trial and error testing showed a point quantity of 25 and a threshold of 0.001 proved to remove a significant amount of noise.

**Voxel Grid Downsampling**

Next we downsample the data to increase processing speed without compromising object resolution using the `make_voxel_grid_filter()` function. The filter reduces the resolution of the cloud to a set leaf size. Too large a leaf-size and you lose a significant amount of object detail, too small and the data contains unnecessary resolution. This leaf size is dependent on the scale of the application. 

<img src="https://github.com/bchou2012/RoboND-Perception-Project-Benjamin-Chou/blob/master/images/voxel_downsampling.png?raw=true" alt="Voxel Downsampling" width="640px">

The `LEAF_SIZE` of 0.005 from the class exercises appear to be sufficient for the purposes of the project. 

**Pass Through Filtering**

For the purposes of object identification we do not need any data from outside a certain set of dimensions. As an example, we are not concerned with the foot of the table or the edges of the boxes to either side of the robot. For that `make_passthrough_filter()` limits the data to coordinates within a maximum and minimum distance along an axis. Through trial and error the values used are:

    x_min = 0.33
    x_max = 0.88
    y_min = -0.46
    y_max = 0.46
    z_min = 0.605
    z_max = 0.9

<img src="https://github.com/bchou2012/RoboND-Perception-Project-Benjamin-Chou/blob/master/images/passthrough.png?raw=true" alt="Passthrough Filter" width="640px">

**RANSAC Plane Segmentation**

We need to separate the table surface from the objects next. For this we will be using RANSAC Plane Segmentation with the `make_segmenter()` function. The function attempts to define a plane by sampling points and then collecting all the points within a defined distance from the plane as the inlier. Ideally the objects will be collected as the outlier. Through trial and error a sufficient defined distance was `max_distance_table = 0.008`.

Afterwards we extract the outliers with the `extract()` with the `negative=True` argument. 

<img src="https://github.com/bchou2012/RoboND-Perception-Project-Benjamin-Chou/blob/master/images/table.png?raw=true" alt="Extracted Table" width="640px">
<img src="https://github.com/bchou2012/RoboND-Perception-Project-Benjamin-Chou/blob/master/images/objects.png?raw=true" alt="Extracted Objects" width="640px">

**Euclidian Clustering**

Now that we have our isolated objects, we will group them into distinct groups. The lessons discussed the characteristics of K-means vs DBSCAN clustering, and we went with DBSCAN clustering. DBSCAN allows us to define any number of clusters instead of a predetermined number of clusters, and exclude outliers in the data. 

We remove RGB from the cloud data since clustering doesn't use color. Next we initialize the KD tree with `make_kdtree()` for defining volumes of grouping. 

Now we can classify the clusters with `make_EuclideanClusterExtraction()`. We will need to determine the cluster tolerance distance, the minimum number of points to define a cluster, and the maximum number of points to the cluster. 

A tolerance that is too large will include far-away object points, and too small will break down an object with gaps in the cloud due to blocking into multiple clusters. 

Too small a minimum will include stray noise, and too large will exclude small objects.

Too small a maximum will divide objects into multiple clusters. Too large, combined with tolerance, may group multiple objects into a single cluster.

For this project, trial and error gave us a tolerance of 0.01, minimum of 25, and maximum of 5000. Other applications will have different values depending on targeted object properties. 

After the clusters have been defined, each one is assigned a color for grouping identification. 

<img src="https://github.com/bchou2012/RoboND-Perception-Project-Benjamin-Chou/blob/master/images/cluster%20cloud.png?raw=true" alt="Euclidean Clustering" width="640px">

## Object Recognition

Now that we have our clusters, now we need to match them to the training information from the objects. We will be using color histograms and surface normals to match the clusters to the objects. 

**Color Histograms**

For the color histogram, we use HSV instead of RGB. HSV allows for greater complexity of color definition, accounting for intensities of the color and how illuminated or shaded the color is.  

As a result when using the `compute_color_histograms()` function it was set to `using_hsv=True`. We use this same parameter for the capture\_features files in SVM training above.

**Surface Normals**

We next calculate the surface normals with `get_normals()` then construct the histogram with `compute_normal_histograms()`. 

After concatenating the color and surface histograms, we make the predictions with `clf.predict()` and publish. 

**World 1 Predictions**

<img src="https://github.com/bchou2012/RoboND-Perception-Project-Benjamin-Chou/blob/master/images/output_labels_1.png?raw=true" alt="World 1 Predictions" width="640px">

**World 2 Predictions**

<img src="https://github.com/bchou2012/RoboND-Perception-Project-Benjamin-Chou/blob/master/images/output_labels_2.png?raw=true" alt="World 2 Predictions" width="640px">

**World 3 Predictions**

<img src="https://github.com/bchou2012/RoboND-Perception-Project-Benjamin-Chou/blob/master/images/output_labels_3.png?raw=true" alt="World 3 Predictions" width="640px">

##Publish YAML Files

Now that we have our predictions, we need to publish the output as YAML files. 

After loading the object list with the objects and their dropbox assignment, and the dropbox list, we create centroids for each predicted object, then loop through the predicted object list against the object and dropbox list and create a YAML dictionary for each predicted object, then write the output. 

The output YAML files are stored in the `\pr2_robot\scripts\` directory for review. 

## Challenge

Due to time constraints the challenge was not attempted.

## Code Implementation

For a complete breakdown see comments in `project_template.py` in `/pr2_robot/scripts/`.

For the capture\_features files in `/sensor_stick_scripts/`, the `compute_color_histograms()` function in the files was set to `using_hsv=True`, and the number of training samples was set to 500. 

For the SVC kernel in `train_svm.py`in `/sensor_stick_scripts/`, it was set to linear in line 66:

`clf = svm.SVC(kernel='linear')`

## Results:

Checking the output.yaml files against the pick\_list and dropbox YAML files in `/pr2_robot/config`, we had the following results:

World 1: 3/3

World 2: 5/5

World 3: 7/8

For World 3, the glue was consistently misidentified throughout code testing. 

## Future Improvements

There are multiple areas for improvement with this project that could be implemented. The noise filtering variables could be finer-tuned for better noise removal. A larger number of captures for the feature extraction could allow the glue to be correctly identified. The challenge would allow for learning of removing objects from the cloud field to prevent collision during the pick and place operation. 
