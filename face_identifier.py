import cv2
import numpy as np
from imageio import get_reader
import random



class Identifier(object):
    """Final class that catches frames from webcam/video
        and tries to identify face on it"""

    def __init__(self, source):
        super(Identifier, self).__init__()
        # initialize webcam/video to catch frames
        self.vid_cam = get_reader(source,  'ffmpeg')

    # prebuilt frontal face training model, for face detection
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default2.xml')
    font = cv2.FONT_HERSHEY_SIMPLEX

    def image_loop(self):
        # main application method

        # identifier that has labels of faces in dataset
        identifier = cv2.face.LBPHFaceRecognizer_create()
        identifier.read('training_data/trainer.yml')

        # creates list of frame colors for every face
        colors = [self.color for i in range(0,6)]

        for i, im in enumerate(self.vid_cam):
            # main loop of application

            # catch frame and prepare it for use
            image = self.capture(i)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # face detecting variable
            faces = self.face_detector.detectMultiScale(gray, 1.2, 5)

            for (x,y,w,h) in faces:
                # loop to find faces on frame and identify them

                # list of face Ids
                Id = identifier.predict(gray[y:y+h,x:x+w])

                # branch to stick to face its id and color
                if(Id[0] == 1):
                    color = colors[0]
                    Id = "Oleg"
                elif (Id[0] == 2):
                    color = colors[1]
                    Id = "Vadim"
                elif (Id[0] == 3):
                    color = colors[2]
                    Id = "Kostik"
                elif (Id[0] == 4):
                    color = colors[3]
                    Id = "Yana"
                elif (Id[0] == 5):
                    color = colors[4]
                    Id = "Vika"
                elif (Id[0] == 6):
                    color = colors[5]
                    Id = "Sasha"
                else:
                    Id = "Unknown"

                # create frame, text for every face on frame and show it
                cv2.rectangle(image, (x-20,y-20), (x+w+20,y+h+20), color, 4)
                cv2.rectangle(image, (x-22,y-90), (x+w+22, y-22), color, -1)
                cv2.putText(image, str(Id), (x,y-40), self.font, 1, (255,255,255), 3)
                cv2.imshow('frame',image) 

                # end loop on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    def capture(self, i):
        # main method to prepare image for work

        image = self.vid_cam.get_data(i)
        image = cv2.flip(image, 0)
        red = image[:,:,2].copy()
        blue = image[:,:,0].copy()
        image[:,:,0] = red
        image[:,:,2] = blue
        return image

    @property
    def color(self):
        # property to generate random color in tuple
        return (random.randint(0,255),random.randint(0,255),random.randint(0,255))

    def end(self):
        # end of application
        self.vid_cam.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # path to webcam/video
    source = '<video0>'
    col = Identifier(source)
    col.image_loop()
    col.end()