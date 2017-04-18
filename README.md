# An attempt at an end to end pipeline for facial clustering

## Introduction
This is an end to end pipeline for facial clustering of photos (similar to what Apple and Android has). Please look at `pipeline/Facial Clustering.ipynb` to see the current progress. 

## Method
There are three main steps in creating an end to end pipeline for the facial clustering of images:
1. Detecting and creating the bounding boxes for the faces
    * For detecting and creating the bounding boxes for the faces I'm using the Multi-Task Cascaded Convolutional Networks ( [MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/)) which has 3 CNNs with the output of one network feeding into the next. Each step refines the bounding boxes for the faces, with the first network being looser on the conditions for the bounding boxes and the last network being the most strict.

2. Creating the embeddings for the faces
    * For creating the embeddings I'm using a modified version [FaceNet](http://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf) which is a Resnet-Inception model that creates a 128 dimensional embedding for a face.

3. Clustering the faces
    * For clustering the faces I'm using the [Rank-Order](https://pdfs.semanticscholar.org/efd6/4b7641bea8ca536f4e179be6e2dd25d519d6.pdf) clustering algorithm. This is a kind of agglomerative clustering technique, which merges the embeddings based on the rank-order distance, and a cluster-level normalized distance. 

As I was unable to train the models myself David Sandberg has graciously trained MTCNN and Resnet-Inception models and I was able to modify his code to my use case. Here is a [link](https://github.com/davidsandberg/facenet) to the Github page

## To setup the environment and install required packages
Assuming that you are using the [Anaconda](https://conda.io/docs/download.html) package manager.
1. Download/Clone the repo and `cd` to it
2. Create a new virtual environment: `conda create -n facialClustering python=3`
3. Install Tensorflow: `pip install tensorflow`
4. Download the weights of Resnet-Inception from [here](https://drive.google.com/file/d/0B5MzpY9kBtDVTGZjcWkzT3pldDA/view) and put it in the `pipeline` directory. I have already included two folders of faces which are subsets from the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) dataset.
5. Go into the `pipeline` folder and open Jupyter Notebook: `jupyter notebook`
6. Open `Facial Clustering.ipynb`
7. Hopefully you will be able to open it, if it doesn't work you can download the `Facial+Clustering.html` and read the notebook.

## TODO:
*  Use alignment technique described in [Face Search at Scale : 80 Million Gallery](https://arxiv.org/pdf/1507.07242.pdf) to speed up the clustering process
*  Create an efficient data structure for obtaining the distances between clusters, instead of having to recalculate them after each clustering iteration.
*  Use factory pattern to make usage of different methods for face detection/alignment cleaner


## Done:
*  ~~Use Delaunay Triangulation to align faces~~ not useful
*  ~~Understand and edit code from [facenet](https://github.com/davidsandberg/facenet) and repurpose the MTCNN code to retrieve multiple faces instead of one~~
*  ~~Use facenet to find the deep features~~
*  ~~Use K-means to cluster~~ not useful
*  ~~Use Affinity Propagation to cluster~~ not useful
*  ~~Read and implement part of the rank order clustering from A Rank-Order Distance based Clustering Algorithm for Face Tagging)[https://pdfs.semanticscholar.org/efd6/4b7641bea8ca536f4e179be6e2dd25d519d6.pdf]~~
*  ~~Use t-SNE using Tensorboard to visualize the embeddings to see if they actually cluster~~
