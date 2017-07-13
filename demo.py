# -*- coding: utf-8 -*-
"""
Created on Dec 20 17:39 2016

@author: Denis Tome'
"""
import cv2
import os
from graph_functions import PoseEstimator
from utils import draw_limbs
from utils import plot_pose
import matplotlib.pyplot as plt


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


# test image
f_name = 'images/test_image.png'
image = cv2.cvtColor(cv2.imread(f_name), cv2.COLOR_BGR2RGB)

# create pose estimator
pose_estimator = PoseEstimator(image.shape)

# load model and run evaluation on image
sess_dir = os.path.dirname(__file__)
sess = pose_estimator.load_model(sess_dir + '/saved_sessions/init_session/init')
pose_2d, visibility, pose_3d = pose_estimator.estimate(image, sess)

# close model
sess.close()

# Show 2D and 3D poses
display_results(image, pose_2d, visibility, pose_3d)





