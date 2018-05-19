import cv2
import os
import numpy as np
from PIL import Image

# create folder if isn't exist
folder = 'training_data/'
if os.path.isdir(os.path.abspath(os.curdir)+folder):
    os.mkdir(folder)

# prebuilt frontal face training model, for face detection
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default2.xml");
# identifier that has labels of faces in dataset
identifier = cv2.face.LBPHFaceRecognizer_create()
# method to get the images and label data
def getImagesAndLabels(path):
    # get all file path
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
    print(imagePaths)
    # initialize empty face sample
    faceSamples=[]
    # initialize empty id
    ids = []
    # main loop through dataset
    for imagePath in imagePaths:
        # get the image and convert it to grayscale
        PIL_img = Image.open(imagePath).convert('L')
        # PIL image to numpy array
        img_numpy = np.array(PIL_img,'uint8')
        # get the image id
        id = int(os.path.split(imagePath)[1].split(".")[1])
        # get the face from the training images
        faces = face_detector.detectMultiScale(img_numpy)
        # loop through each face, append to their respective ID
        for (x,y,w,h) in faces:
            # add the image to face samples
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            # add the ID to IDs
            ids.append(id)

    return faceSamples, ids

# Get the faces and IDs
faces, ids = getImagesAndLabels('dataset')
# Train the model using the faces and IDs
identifier.train(faces, np.array(ids))
# Save the model into trainer.yml
identifier.write(folder + 'trainer.yml')
