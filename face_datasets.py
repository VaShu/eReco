import cv2
from imageio import get_reader
import os


class Collector(object):
    """Class to collect faces from webcam/video
        into dataset for further needs"""

    def __init__(self, id, source, limit, folder):
        super(Collector, self).__init__()
        # folder to save images of faces
        self.folder = folder
        # id of a current user
        self.id = id
        # amount of faces to find
        self.limit = limit
        # initialize webcam/video to catch frames
        self.vid_cam = get_reader(source,  'ffmpeg')

        # create folder if isn't exist
        if os.path.isdir(os.path.abspath(os.curdir)+folder):
            os.mkdir(folder)

    # prebuilt frontal face training model, for face detection
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default2.xml')

    def image_loop(self):
        # main application method

        # amount of found faces
        count = 0

        for i, im in enumerate(self.vid_cam):
            # main loop of application
            image = self.capture(i)

            # catch frame and prepare it for use
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # face detecting variable
            faces = self.face_detector.detectMultiScale(gray, 1.2, 5)

            for (x,y,w,h) in faces:
                # loop to find faces on frame and save them

                # create frame for every found face show it
                cv2.rectangle(image, (x,y), (x+w,y+h), (200,15,15), 2)
                # increase when face is found
                count += 1
                cv2.imwrite(self.folder + "User." + str(self.id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
                cv2.imshow('frame', image)
                # amount of images we found in percents
                print('Processing [{0}%]'.format(int((count/self.limit)*100)))

                # end loop on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

                # if all faces found end loop
            elif count>=self.limit:
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

    def end(self):
        # end of application
        self.vid_cam.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # path to webcam/video
    source = '<video0>'
    # id of a current user
    id = 2
    # amount of faces to find
    limit = 100
    # folder to save images of faces
    folder = 'dataset/'
    col = Collector(id, source, limit, folder)
    col.image_loop()
    col.end()
