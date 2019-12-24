from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lib.preprocessing import AspectAwarePreprocessor, ImageToArrayPreprocessor, SimplePreprocessor
from lib.nn.conv import VGGNet
from lib.datasets import SimpleDatasetLoader
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# dataset is in flowers17/{category}/{image} format. So we can split the path and then take the second last item
# to fetch category
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

aspect_aware_processor = AspectAwarePreprocessor(64, 64)
image_to_array_processor = ImageToArrayPreprocessor()

# Preprocess data by first crop them maintaining aspect ration, and then convert to image array.

dataset_loader = SimpleDatasetLoader(preprocessors=[aspect_aware_processor, image_to_array_processor])
data, labels = dataset_loader.load(imagePaths, verbose=500)

# take datapoint value between 0 and 1

data = data.astype("float") / 255.0

# split train-test data in 75-25 ratio

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

opt = SGD(lr=0.05)

model = VGGNet.build(width=64, height=64, depth=3, classes=len(classNames))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=1)

predictions = model.predict(testX, batch_size=32)

# Generate classification report

print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('minivgg.png')
plt.show()
