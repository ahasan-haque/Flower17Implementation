from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from lib.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import os
import random
import argparse
import progressbar


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-o", "--output", required=True, help="path to output HDF5 file")
ap.add_argument("-b", "--batch-size",  type=int, default=32,
                help="batch-size of images to be passed through the network")
ap.add_argument("-s", "--buffer-size", type=int, default=1000,
                help="buffer size of extracted features")

args = vars(ap.parse_args())

batch_size = args["batch_size"]

# Firstly, loading image paths from dataset path

imagePaths = list(paths.list_images(args['dataset']))
random.shuffle(imagePaths)

labels = [p.split(os.path.sep)[-2] for p in imagePaths]
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Load the VGG16 network

model = VGG16(weights='imagenet', include_top=False)

# Initialize HDF5 dataset writer

dataset = HDF5DatasetWriter((len(imagePaths), 512 * 7 * 7), args['output'], dataKey='features',
                            bufSize=args['buffer_size'])
dataset.storeClassLabels(label_encoder.classes_)

# Initialize progressbar

widgets = ["Extracting Features", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
progress_bar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

# loop over images in batches

for i in np.arange(0, len(imagePaths), batch_size):
    batchPaths = imagePaths[i: i + batch_size]
    batchLabels = labels[i: i + batch_size]
    batchImages = []

    # loop over current batch
    for (j, imagePath) in enumerate(batchPaths):
        # load a single image from path in a fixed size and make a keras compitable image array
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)

        # preprocess image to pass through model

        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        # adding image to batch that will be passed through model at a time
        batchImages.append(image)

    # stacking images vertically as numpy array
    batchImages = np.vstack(batchImages)

    # pass though the model to extract features and express features in proper dimension
    features = model.predict(batchImages, batch_size=batch_size)

    features = features.reshape((features.shape[0], 512 * 7 * 7))

    # add retrieved feature to HDF5 dataset
    dataset.add(features, batchLabels)
    # Update the progress bar
    progress_bar.update(i)

dataset.close()
progress_bar.finish()
