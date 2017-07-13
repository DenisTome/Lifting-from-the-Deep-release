#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Dec 20 17:39 2016

@author: Denis Tome'
"""
import cv2
from graph_functions import PoseEstimator
from utils import draw_limbs
from utils import plot_pose
import matplotlib.pyplot as plt
from os.path import dirname, realpath

def main():
    image_file_name = 'images/test_image.png'
    image = cv2.imread(image_file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # conversion to rgb

    # create pose estimator
    image_size = image.shape
    session_dir = dirname(realpath(__file__))
    session_path = session_dir + '/saved_sessions/init_session/init'
    
    pose_estimator = PoseEstimator(image_size, session_path)

    # load model and run evaluation on image
    pose_estimator.initialise()

    # estimation
    pose_2d, visibility, pose_3d = pose_estimator.estimate(image)

    # close model
    pose_estimator.close()

    # Show 2D and 3D poses
    display_results(image, pose_2d, visibility, pose_3d)


def display_results(in_image, data_2d, joint_visibility, data_3d):
    """Plot 2D and 3D poses for each of the people in the image."""
    plt.figure()
    draw_limbs(in_image, data_2d, joint_visibility)
    plt.imshow(in_image)
    plt.axis('off')

    # Show 3D poses
    for single_3D in data_3d:
        # or plot_pose(Prob3dPose.centre_all(single_3D))
        plot_pose(single_3D)

    plt.show()

if __name__ == '__main__':
    import sys
    sys.exit(main())
