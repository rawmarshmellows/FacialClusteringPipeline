"""Performs face alignment and stores face thumbnails in the output directory."""
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

from scipy import misc
import sys
sys.path.insert(0, "../src")
import os
import argparse
import tensorflow as tf
import numpy as np
import detect_face
import facenet
import random
from time import sleep

def main(args):
    sleep(random.random())
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    # facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = facenet.get_dataset(args.input_dir)
    # # print('Dataset is:')
    # # print(dataset)
    # print('Creating networks and loading parameters')
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor used for the pyramid

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
    
    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        if args.random_order:
            random.shuffle(dataset)
        for cls in dataset: # note here that cls images of the same label
            output_class_dir = os.path.join(output_dir, cls.name)
            # # print(output_class_dir)
            if not os.path.exists(output_class_dir):
                # # print("New directory created")
                os.makedirs(output_class_dir)
                if args.random_order:
                    random.shuffle(cls.image_paths)
            # # print("Entering directory: {}".format(output_class_dir))
            for image_path in cls.image_paths: # getting the path of a certain label of images
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename+'.png')
                # # print("os.path.split(image_path) is: {}".format(os.path.split(image_path)))
                # # print("os.path.splitext(os.path.split(image_path)[1]) is: {}".format(os.path.splitext(os.path.split(image_path)[1])))
                # print("filename is: {}".format(filename))
                # print("image_path is: {}".format(image_path))
                # # print("output_filename is: {}".format(output_filename))
                if not os.path.exists(output_filename): # if the file doesn't exist
                    try: # try to read it
                        img = misc.imread(image_path)
                        # print("Reading image")
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        # print(errorMessage)
                    else:
                        if img.ndim<2:
                            # # print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                        img = img[:,:,0:3]  # getting only the rgb channels but not the alpha channel
                        # # print("Detecting faces")
                        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

                        # Get the number of boxes that were found
                        nrof_faces = bounding_boxes.shape[0]
                        if nrof_faces>0:
                            det = bounding_boxes[:,0:4] # these are the coordinates of the x1, y1, x2, y2 for the corners of the bbox
                            img_size = np.asarray(img.shape)[0:2]

                            # IF WE WANT TO GET ALL THE FACES THEN WE NEED TO CHANGE THIS PART OF THE CODE
                            if nrof_faces>1 and not args.all_faces:
                                # this is the area for all of the bboxes that were found
                                bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                                img_center = img_size / 2

                                # find the offset from the center of the image for each of the boxes
                                offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])

                                # find the squared offset distance from the center for each of the boxes
                                offset_dist_squared = np.sum(np.power(offsets,2.0),0)

                                # 
                                # # print("bounding_box_size-offset_dist_squared*2.0 is: {}".format(bounding_box_size-offset_dist_squared*2.0))

                                # note here that it is picking the bounding box that is the biggest and closest to the center
                                index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                                det = det[index,:]

                            elif nrof_faces >1 and args.all_faces:
                                for i, bb in enumerate(det):
                                    # # print("bb before: {}".format(bb))
                                    tmp_img = img.copy()
                                    bb[0] = np.maximum(bb[0]-args.margin/2, 0).astype(np.int32)
                                    bb[1] = np.maximum(bb[1]-args.margin/2, 0).astype(np.int32)
                                    bb[2] = np.minimum(bb[2]+args.margin/2, img_size[1]).astype(np.int32)
                                    bb[3] = np.minimum(bb[3]+args.margin/2, img_size[0]).astype(np.int32)
                                    # # print("bb after: {}".format(bb))
                                    cropped = tmp_img[bb[1]:bb[3], bb[0]:bb[2], :]
                                    scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear') 
                                    image_path = os.path.join(output_class_dir, "face_{}.png".format(i))
                                    # # print("image_path is: {}".format(image_path))
                                    misc.imsave(image_path, scaled)    
                                # print("Found {} face(s) in this image".format(i))
                                return

                            # Draw the bbox that is nearest to the center of the image if there is only one face
                            det = np.squeeze(det)
                            bb = np.zeros(4, dtype=np.int32)

                            # find the bbox for the face
                            bb[0] = np.maximum(det[0]-args.margin/2, 0)
                            bb[1] = np.maximum(det[1]-args.margin/2, 0)
                            bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
                            bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])

                            # crop the image so that only the face remains
                            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]

                            # resize the face so that it is a face
                            scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                            nrof_successfully_aligned += 1

                            # save the resized image
                            # print("saving at location: {}".format(output_filename))
                            misc.imsave(output_filename, scaled)
                            text_file.write('%s %d %d %d %d\n' % (output_filename, bb[0], bb[1], bb[2], bb[3]))
                        else:
                            # # print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
                            
    # print('Total number of images: %d' % nrof_images_total)
    # print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
            

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order', 
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--all_faces', type=bool, 
        help='Save all the faces from the image.', default = False)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
