"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
import sys
import math
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate

def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            # Read the file containing the pairs used for testing
            pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
            # print("pairs.txt is: {}".format(pairs))
            # Get the paths for the corresponding images
            # print("os.path.expanduser(args.lfw_dir) is: {}".format(os.path.expanduser(args.lfw_dir)))
            # print("pairs[0:10,] is: {}".format(pairs[0:10,]))
            # print("args.lfw_file_ext is: {}".format(args.lfw_file_ext))
            paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)
            # each element of paths is a 2-tuple of the image paths of the 2 images in the pair
            # each element of actual_issame is whether the two images in the pair are actually the same, 
            # it can be seen as the ground truth
            # print("paths is: {}".format(paths))
            # print("len(actual_issame) is: {}".format(len(actual_issame)))
            # print("actual_issame is: {}".format(actual_issame))

            # Load the model
            print('Model directory: %s' % args.model_dir)
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))
            
            
#            print([op.name for op in tf.get_default_graph().get_operations()])
            # print('Metagraph file: %s' % meta_file)
            # print('Checkpoint file: %s' % ckpt_file)
            facenet.load_model(args.model_dir, meta_file, ckpt_file)
            
            # Get input and output tensors
            # print("Getting the input and output tensors for facenet")
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            print([op.name for op in tf.get_default_graph().get_operations()])
            
            image_size = images_placeholder.get_shape()[1]
            embedding_size = embeddings.get_shape()[1]
            # print("image_size is: {}".format(image_size))
            # print("embedding_size is: {}".format(embedding_size))
        
            # Run forward pass to calculate embeddings
            # print('Runnning forward pass on LFW images')
            batch_size = args.lfw_batch_size
            nrof_images = len(paths)
            nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))

            # print("batch_size is: {}".format(batch_size))
            # print("nrof_images is: {}".format(nrof_images))
            # print("nrof_batches is: {}".format(nrof_batches))

            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches):
                # print("Batch: {0}/{1}".format(i+1, len(nrof_batches)))
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                # Load the images
                images = facenet.load_data(paths_batch, False, False, image_size)
                # Feed it into the network
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                # Update the emb_array
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            
            # print("emb_array is: {}".format(emb_array))
            # tpr is the true positive rate (how many samples that were classified as true were actually true)
            # fpr is the false positive rate (how many samples that were classified as true were actually false)
            # accuracy is (tp + tn)/number of samples
            # val is the percentage of samples that were classified to be correct
            # val_std is the std of val
            # far is the percentage of samples that were classified to be wrong
            tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(emb_array, 
                actual_issame, nrof_folds=args.lfw_nrof_folds)

            # print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
            # print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

            auc = metrics.auc(fpr, tpr)
            # print('Area Under Curve (AUC): %1.3f' % auc)
            eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
            # print('Equal Error Rate (EER): %1.3f' % eer)
            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('lfw_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('model_dir', type=str, 
        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters')
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='../data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
