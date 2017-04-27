# -*- coding: utf-8 -*-
"""
Created on Mar 23 11:34 2017

@author: Denis Tome'
"""
import cPickle as pickle
import tensorflow as tf

from utils import cpm
from utils.config import config


def tf_init_weights(root_scope, params_dict):
    names_to_values = {}
    for scope, weights in params_dict.iteritems():
        variables = tf.get_collection(tf.GraphKeys.VARIABLES,
                                      '%s/%s' % (root_scope, scope))
        assert len(weights) == len(variables)
        for v, w in zip(variables, weights):
            assert (v.get_shape() == w.shape)
            names_to_values[v.name] = w
    return tf.contrib.framework.assign_from_values(names_to_values)


person_pkl_file = '../caffe_models/exported_caffemodels/trained_person_MPI.pkl'
pose_pkl_file = '../caffe_models/exported_caffemodels/trained_MPI.pkl'

# loading weights from pickles
person_params = pickle.load(open(person_pkl_file))
pose_params = pickle.load(open(pose_pkl_file))
person_net_conf, pose_net_conf = config()

tf.reset_default_graph()
with tf.variable_scope('CPM'):
    # input dims for the person network
    PH, PW = 368, 654
    image_in = tf.placeholder(tf.float32, [person_net_conf['BS'], person_net_conf['H'], person_net_conf['H'], 3])
    heatmap_person = cpm.inference_person(image_in)
    init_person_op, init_person_feed = tf_init_weights('CPM/PersonNet', person_params)

    # input dims for the pose network
    N, H, W = 16, 376, 376
    pose_image_in = tf.placeholder(tf.float32, [pose_net_conf['BS'], pose_net_conf['H'], pose_net_conf['H'], 3])
    pose_centermap_in = tf.placeholder(tf.float32, [pose_net_conf['BS'], pose_net_conf['H'], pose_net_conf['H'], 1])
    heatmap_pose = cpm.inference_pose(pose_image_in, pose_centermap_in)
    init_pose_op, init_pose_feed = tf_init_weights('CPM/PoseNet', pose_params)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    sess.run(init_person_op, init_person_feed)
    saver = tf.train.Saver()
    saver.save(sess, '../saved_sessions/person_MPI/init')

with tf.Session() as sess:
    sess.run(init)
    sess.run(init_pose_op, init_pose_feed)
    saver = tf.train.Saver()
    saver.save(sess, '../saved_sessions/pose_MPI/init')
