#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Abel Babu
#
# Created:     22/11/2012
# Copyright:   (c) Abel Babu 2012
# Licence:     <your licence>
#---------------------------------------------------------------------------------

import cv2,cv,sys,time,numpy as np,profile

class face_Detect():

    def __init__(self):

        self.smile=cv2.imread('smile.JPG')
        self.poker=cv2.imread('poker.PNG')
        print self.smile.shape
        cv2.namedWindow('Image')
        self.vid = cv2.VideoCapture(0)
        self.vid.set(cv.CV_CAP_PROP_FPS,33)
        im =self.vid.read()[1]
        self.shape=im.shape
        print self.shape
        cv2.waitKey(20)
        print self.vid.get(cv.CV_CAP_PROP_FPS)
        self.haar1=cv2.CascadeClassifier()
        self.haar2=cv2.CascadeClassifier()
        print self.haar1.load("lbpcascade_frontalface.xml")
        print self.haar2.load("smiled_01.xml")
        self.kernel = np.ones((5,5),'uint8')
        self.new=np.array([])
        self.S=0
        self.C=0


    def filters(self,im):
        im1=cv2.erode(im,self.kernel)
        im=cv2.dilate(im1,self.kernel)
        im1 = cv2.morphologyEx(im,cv2.MORPH_OPEN,self.kernel)
        im = cv2.morphologyEx(im1,cv2.MORPH_CLOSE,self.kernel)
        return im

    def face_decide(self,im,faces,smiles):
        self.S=0
        self.C=0
        if np.array(faces).any():
            for face in faces:
                if face[0]==face[0]+face[2] or face[1]==face[1]+face[3]:
                    return im
                self.C=self.C+1
                im=self.smile_decide(im,face,smiles)
            return im
        else:
            return im

    def smile_decide(self,im,face,smiles):
        if np.array(smiles).any():
            for smile in smiles:
                print smile
                #print 'face:',face,'smile:',smile
                if face[0]<(smile[2]+2*smile[0])/2<face[0]+face[2] and face[1]<(smile[3]+2*smile[1])/2<face[1]+face[3]:
                    cv2.rectangle(im,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(255,0,0))
                    small = cv2.resize(self.smile, (face[2],face[3]), interpolation=cv2.INTER_AREA)
                    #print 'HAPPY and smiling'
                    im[face[1]:(face[1]+face[3]),face[0]:(face[0]+face[2])]=small[:,:]
                    return im
                else:
                    cv2.rectangle(im,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(255,0,0))
                    small = cv2.resize(self.poker,(face[2],face[3]), interpolation=cv2.INTER_AREA)
                    #print 'SAD with smile detected'
                    im[face[1]:(face[1]+face[3]),face[0]:(face[0]+face[2])]=small[:,:]
                    return im
        else:
            cv2.rectangle(im,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(255,0,0))
            small = cv2.resize(self.poker,(face[2],face[3]), interpolation=cv2.INTER_AREA)
            #print 'SAD with no smiles dectected '
            im[face[1]:(face[1]+face[3]),face[0]:(face[0]+face[2])]=small[:,:]
            self.S=self.S+1
            return im

    def mood_meter(self,im):
        count=0
        im[20:self.shape[0]-20,10:40]=255
        if self.C == 0:
            return im
        inc=(self.shape[0]-40)/(self.C)
        count=(self.C-self.S)*inc
        print 'meter',self.shape[0]-count
        im[self.shape[0]-count-20:self.shape[0]-20,10:40]=[0,255,0]
        return im



    def core(self):
        t1=time.time()
        im =self.vid.read()[1]
        im1=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        img=self.filters(im1)
        faces=self.haar1.detectMultiScale(img,1.2,2,cv.CV_HAAR_DO_CANNY_PRUNING,(100,100))
        smiles=self.haar2.detectMultiScale(img,1.2,2,cv.CV_HAAR_DO_CANNY_PRUNING,(100,100))
        im=self.face_decide(im,faces,smiles)
        im=self.mood_meter(im)
        cv2.imshow("Image",im)
        cv2.waitKey(10)
        t2=time.time()
        print 'Time:',round(t2-t1,3),'sec'

def main():
    work=face_Detect()
    while True:
        try:
            work.core()
        except KeyboardInterrupt:
            work.vid.release()
            cv2.destroyAllWindows()
            sys.exit()
    work.vid.release()
    cv2.destroyAllWindows()
    sys.exit()

if __name__ == '__main__':
    main()






