# -*- coding: utf-8 -*-
"""
Created on Dec 15 14:46 2016

@author: Denis Tome'
"""
import caffe
import cPickle as pickle
import argparse
import os


def convert_weights(w):
    if len(w.shape) == 4:
        return w.transpose((2, 3, 1, 0))
    elif len(w.shape) == 1:
        return w
    else:
        raise ValueError('Unsupported weights')


def dump_parameters(net, path):
    params = {name: [convert_weights(blob.data) for blob in blobs]
              for name, blobs in net.params.iteritems()}
    pickle.dump(params, open(path, 'w+'))


def check_file_exists(file_path):
    """Check if file exists"""
    try:
        return os.path.isfile(file_path)
    except:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('deploy_file', metavar='deploy_file', type=str, help='Path deploy.prototxt file')
    parser.add_argument('model_file', metavar='model_file', type=str, help='Path model.caffemodel file')
    parser.add_argument('output_file', metavar='output_file', type=str, help='Path saved pkl model')

    args = parser.parse_args()
    if not check_file_exists(args.deploy_file):
        raise Exception('File not found at path: %r' % args.deploy_file)
    if not check_file_exists(args.model_file):
        raise Exception('File not found at path: %r' % args.model_file)

    pkl_file = args.output_file + '.pkl'

    caffe_net = caffe.Net(args.deploy_file, args.model_file, caffe.TEST)
    dump_parameters(caffe_net, pkl_file)

if __name__ == '__main__':
    main()
