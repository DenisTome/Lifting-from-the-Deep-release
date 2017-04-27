# -*- coding: utf-8 -*-
"""
Created on Dec 20 17:39 2016

@author: Denis Tome'
"""
import numpy as np
import cv2
import utils.config as config
import tensorflow as tf
import cpm
from utils.draw import *
from prob_model import Prob3dPose
import utils.process as ut

fname = 'images/test_image.png'

image = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
scale = config.INPUT_SIZE/(image.shape[0] * 1.0)
image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
b_image = np.array(image[np.newaxis] / 255.0 - 0.5, dtype=np.float32)

tf.reset_default_graph()

with tf.variable_scope('CPM'):
    # placeholders for person network
    image_in = tf.placeholder(tf.float32, [1, config.INPUT_SIZE, image.shape[1], 3])
    heatmap_person = cpm.inference_person(image_in)
    heatmap_person_large = tf.image.resize_images(heatmap_person, [config.INPUT_SIZE, image.shape[1]])

    # placeholders for pose network
    N = 16
    pose_image_in = tf.placeholder(tf.float32, [N, config.INPUT_SIZE, config.INPUT_SIZE, 3])
    pose_centermap_in = tf.placeholder(tf.float32, [N, config.INPUT_SIZE, config.INPUT_SIZE, 1])
    heatmap_pose = cpm.inference_pose(pose_image_in, pose_centermap_in)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, 'saved_sessions/person_MPI/init')
    hmap_person = sess.run(heatmap_person_large, {image_in: b_image})

hmap_person = np.squeeze(hmap_person)
centers = ut.detect_objects_heatmap(hmap_person)
b_pose_image, b_pose_cmap = ut.prepare_input_posenet(b_image[0], centers, [config.INPUT_SIZE, image.shape[1]],
                                                     [config.INPUT_SIZE, config.INPUT_SIZE])

# sess = tf.InteractiveSession()
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, 'saved_sessions/pose_MPI/init')

    feed_dict = {
        pose_image_in: b_pose_image,
        pose_centermap_in: b_pose_cmap
    }

    _hmap_pose = sess.run(heatmap_pose, feed_dict)

# Estimate 2D poses
parts, visible = ut.detect_parts_heatmaps(_hmap_pose, centers, [config.INPUT_SIZE, config.INPUT_SIZE])

# Estimate 3D poses
poseLifting = Prob3dPose()
pose2D, weights = Prob3dPose.transform_joints(parts, visible)
pose3D = poseLifting.compute_3d(pose2D, weights)

# Show 2D poses
plt.figure()
draw_limbs(image, parts, visible)
plt.imshow(image)
plt.axis('off')

# Show 3D poses
for single_3D in pose3D:
    plot_pose(single_3D)

plt.show()





