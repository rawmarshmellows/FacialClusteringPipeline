"""Helper for evaluation on the Labeled Faces in the Wild dataset 
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

import os
import numpy as np
import facenet

def evaluate(embeddings, actual_issame, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    # Note there that x[startAt:endBefore:skip] is the notation
    # so embeddings1 and embeddings2 are simply two lists with
    # embeddings1[i] and embeddings2[i] being the pairs that we need to validate    
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    print("thresholds is: {}".format(thresholds))
    print("embeddings1 is: {}".format(embeddings1))
    print("embeddings2 is: {}".format(embeddings2))
    tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds)
    thresholds = np.arange(0, 4, 0.001)
    # val is the percentage of samples that were classified to be correct
    # val_std is the std of val
    # far is the percentage of samples that were classified to be wrong
    val, val_std, far = facenet.calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far

def get_paths(lfw_dir, pairs, file_ext):
    # Note here that we need to understand what exactly the pairs.txt file is to understand 
    # what this code means

    # The pairs.txt file 10 sets of pairs, with each set containing 300 matching pairs 
    # and 300 mismatching pairs

    # The matching pairs are formatted as:
    # <person name> <person image number> <person image number>
    # For example:
    # George_W_Bush   10   24
    # This would mean that the pair consists of images George_W_Bush_0010.jpg and
    # George_W_Bush_0024.jpg.
    # Hence we know that facenet should classify these two images as the same

    # The mismatching pairs are formatted as:
    # <person name> <person image number> <another person> <another person's image number>
    # for example:
    # George_W_Bush   12   John_Kerry   8
    # This would mean that the pair consists of images George_W_Bush_0012.jpg and
    # John_Kery_0008.jpg.
    # Hence we know that facenet should classify these two image as different


    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    print("There are {} pairs".format(len(pairs)))
    for pair in pairs:

        # These are just verifying that the images do actually exist
        if len(pair) == 3:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
            path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])+'.'+file_ext)
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
            path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])+'.'+file_ext)
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist 

            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            # print("path0 is: {}".format(path0))
            # print("path1 is: {}".format(path1))
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    
    return path_list, issame_list

def read_pairs(pairs_filename):
    pairs = []
    print("pairs_filename is: {}".format(pairs_filename))
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)



