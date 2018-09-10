## Writeup / README

### This is a brief description about my project.  
My project code is contained in the jupyter notebook located in "./project-find lane.ipynb".  
My result images are located in "./output_images". My result video is located in "./output_videos".

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/test_undistorted.jpg "Undistorted"
[image2]: ./output_images/test_undistorted_lane.jpg "Undistorted Road"
[image3]: ./output_images/test_threshold_lane.jpg "Binary"
[image4]: ./output_images/test_warp_lane.jpg "Warp"
[image5]: ./output_images/test_fitted_lane.jpg "Fit Visual"
[image6]: ./output_images/test_result_lane.jpg "Output"
[video1]: ./output_videos/result_project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

Camera Calibration can be divided into two small steps: 

#### 1. Computed the camera matrix and distortion coefficients by given a set of chessboard images.

First, I prepare `object points` and `image points`, which are the chessboard corners in the world and images.  
Then I used the `cv2.calibrateCamera()` function to compute the camera calibration and distortion coefficients.  

#### 2. Apply distortion correction to original images.

I use the `cv2.undistort()` function to distortion correction the original images. 
This is an example of a distortion corrected calibration image.

![alt text][image1]

### Pipeline (single images)

#### 1. Camera Calibration

This is an example of how I apply the distortion correction to test images:

![alt text][image2]

#### 2. Create a binary image with clear lanes

I used a combination of color and gradient thresholds to generate a binary image.  
Here's an example of my output for this step.

![alt text][image3]

#### 3. Perspective transform

I define the source and destination points for perspective transform.  
I verified my perspective transform by inspecting if the lines are parallel in the warped image.

![alt text][image4]

#### 4. Find lane pixel and fit polynomial

I locate the lane's start points by region mask and histogram. Then I use slide window to find lane pixels.  
So I can use these lane pixels to fit 2 order polynomial coefficient for plotting lines.

![alt text][image5]

#### 5. Compute curvature and vehicle position in meter

I did these by unit conversion and math formula from pixels to meters.

#### 6. Warp back and draw fitted lane

Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Define a `Line` class to track the line's characters
#### 2. Find enough lane pixels 
First, we can search around last frame's fit.  If it can't work well, reset to slide window.  
If slide window can't work well, we can use last frame's pixels to get a approximate result.  
#### 3. Fit polynomial and sanity  check 
I compare this fit and last frame's fit, they should have similar curvature.  
If this fit isn't sanity, we can use the average fit to get a approximate result.  

Here's a [link to my video result](./output_videos/result_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  

There are two main problems I faced in this project:  
1.adjust the color and gradient threshold to get a binary image with clear lane and eliminate other interference as much as possible.  
2.build a pipeline to find enough pixels for fitting lines

#### 2. Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline may fail at two steps: create threshold image, find lane pixels.  
I can adjust the color and gradient threshold to get a binary image with clear lane and eliminate other interference.

