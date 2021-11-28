from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import *
from PyQt5 import uic
import sys
from numpy.core.fromnumeric import shape
from numpy.matrixlib.defmatrix import matrix
from tensorflow.python.keras.engine.training import Model
from ui import Ui_Form
from ui2 import Ui_Form2
import cv2
import os
import numpy as np
from PyQt5.QtCore import QLibraryInfo
import math
import tensorflow as tf
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
import matplotlib.image as iimg
"""from tensorflow.keras import models
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

input_shape = (224, 224, 3)

model = Sequential([
    Conv2D(64, (3, 3), input_shape=input_shape, padding='same',
           activation='relu'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dense(4096, activation='relu'),
    Dense(1000, activation='softmax')
])"""

'''
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)

'''
def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
 
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated
def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted

def mag(x,y):
    n,m = x.shape
    img_new = []
    for i in range(n):
        line = []
        for j in range(m):
            line.append(math.ceil(x[i,j]**2+y[i,j])**(1/2))
        img_new.append(line)
    img_new = np.array(img_new)
    img_new = cv2.convertScaleAbs(img_new)
    return img_new

def convolution(k, data):
    n,m = data.shape
    img_new = []
    for i in range(n-3):
        line = []
        for j in range(m-3):
            a = data[i:i+3,j:j+3]
            line.append(np.sum(np.multiply(k, a)))
        img_new.append(line)
    img_new = np.array(img_new)
    img_new = cv2.convertScaleAbs(img_new)
    return img_new

def update(x):
    img = cv2.imread(filename='Q1_Image/Dog_Strong.jpg')
    img2 = cv2.imread(filename='Q1_Image/Dog_Weak.jpg')
    blenda = cv2.getTrackbarPos('Blend',"Strong Dog")
    blendb = cv2.getTrackbarPos('Blend',"Weak Dog")
    mix = cv2.addWeighted(img, blenda/(blenda+blendb), img2, blendb/(blenda + blendb), 0)
    cv2.imshow("Mix",mix)

def rgb2gray(rgb,weight):
    return np.dot(rgb[...,:3], weight)

class Window:

    def __init__(self):
        self.Form2 = QtWidgets.QWidget()
        self.ui2 = Ui_Form2()
        self.ui2.setupUi(self.Form2)
        self.flag = 0
        self.Form = QtWidgets.QWidget()
        self.ui = Ui_Form()
        self.ui.setupUi(self.Form)
        self.ui.loadBut.clicked.connect(self.load)
        self.ui.colorSepBut.clicked.connect(self.colorSep)
        self.ui.colorTransBut.clicked.connect(self.colorTrans)
        self.ui.blendBut.clicked.connect(self.blend)
        self.ui.blurBut.clicked.connect(self.blur1)
        self.ui.bilateralBut.clicked.connect(self.bilateral)
        self.ui.MedianBut.clicked.connect(self.median)
        self.ui.blurBut_2.clicked.connect(self.blur2)
        self.ui.sobelXBut.clicked.connect(self.sobelx)
        self.ui.sobelYBut.clicked.connect(self.sobely)
        self.ui.MagBut.clicked.connect(self.magnitude)
        self.ui.resizeBut.clicked.connect(self.resizeUI)
        self.ui2.pushButton_2.clicked.connect(self.but2)
        self.ui.accBut.clicked.connect(self.acc)
        self.ui.modelBut.clicked.connect(self.modelstr)
        self.ui.testBut.clicked.connect(self.test)
        self.ui.hyperBut.clicked.connect(self.hyper)
        self.ui.trainBut.clicked.connect(self.train)
        self.ui.transBut.clicked.connect(self.transUI)
        self.ui.shearBut.clicked.connect(self.shearUI)
        self.ui.rotBut.clicked.connect(self.rotateUI)
        #layout = QVBoxLayout()


    def load(self):
        filename, _ = QFileDialog.getOpenFileName(self.Form,"Open file", "./")
        self.img = cv2.imread(filename=filename)
        cv2.imshow('load',self.img)

    def colorSep(self):
        self.img = cv2.imread(filename='Q1_Image/Sun.jpg')
        (B,G,R) = cv2.split(self.img)
        zeros = np.zeros(self.img.shape[:2],dtype="uint8")
        cv2.imshow('B channel',cv2.merge([B,zeros,zeros]))
        cv2.imshow('G channel',cv2.merge([zeros,G,zeros]))
        cv2.imshow('R channel',cv2.merge([zeros,zeros,R]))

    def colorTrans(self):
        self.img = cv2.imread('Q1_Image/Sun.jpg')
        cv2.imshow('I1',rgb2gray(self.img,[0.07,0.72,0.21])/255)
        cv2.imshow('I2',rgb2gray(self.img,[1/3,1/3,1/3])/255)

    def blend(self):
        self.img = cv2.imread(filename='Q1_Image/Dog_Strong.jpg')
        self.img2 = cv2.imread(filename='Q1_Image/Dog_Weak.jpg')
        cv2.imshow("Strong Dog",self.img)
        cv2.imshow("Weak Dog",self.img2)
        cv2.createTrackbar("Blend","Strong Dog",0,255,update)
        cv2.createTrackbar("Blend","Weak Dog",0,255,update)
    
    def blur1(self):
        self.img = cv2.imread('Q2_Image/Lenna_whiteNoise.jpg')
        blur = cv2.GaussianBlur(self.img, (5, 5), 0)
        cv2.imshow("Lenna origin",self.img)
        cv2.imshow("Lenna blur",blur)
        
    def bilateral(self):
        self.img = cv2.imread('Q2_Image/Lenna_whiteNoise.jpg')
        bilateral = cv2.bilateralFilter(self.img, 9, 90, 90)
        cv2.imshow("Lenna origin",self.img)
        cv2.imshow("Lenna blur",bilateral)

    def median(self):
        self.img = cv2.imread('Q2_Image/Lenna_pepperSalt.jpg')
        median3 = cv2.medianBlur(self.img,3)
        median5 = cv2.medianBlur(self.img,5)
        cv2.imshow("Lenna origin",self.img)
        cv2.imshow("Lenna blur 3*3",median3)
        cv2.imshow("Lenna blur 5*5",median5)
    
    def blur2(self):
        self.img = cv2.imread('Q3_Image/House.jpg',cv2.IMREAD_GRAYSCALE)
        kernal = [[1/16,2/16,1/16],[2/16,4/16,2/16],[1/16,2/16,1/16]]
        kernal = np.array(kernal)
        blur = convolution(kernal,self.img)
        cv2.imshow("origin",self.img)
        cv2.imshow("blur",blur)
    
    def sobelx(self):
        self.img = cv2.imread('Q3_Image/House.jpg',cv2.IMREAD_GRAYSCALE)
        kernal = [[1,2,1],[0,0,0],[-1,-2,-1]]
        kernal = np.array(kernal)
        sobelx = convolution(kernal,self.img)
        cv2.imshow("origin",self.img)
        cv2.imshow("sobelx",sobelx)
    
    def sobely(self):
        self.img = cv2.imread('Q3_Image/House.jpg',cv2.IMREAD_GRAYSCALE)
        kernal = [[-1,0,1],[-2,0,2],[-1,0,1]]
        kernal = np.array(kernal)
        sobely = convolution(kernal,self.img)
        cv2.imshow("origin",self.img)
        cv2.imshow("sobely",sobely)

    def magnitude(self):
        self.img = cv2.imread('Q3_Image/House.jpg',cv2.IMREAD_GRAYSCALE)
        kernal = [[1,2,1],[0,0,0],[-1,-2,-1]]
        kernal = np.array(kernal)
        sobelx = convolution(kernal,self.img)
        kernal = [[-1,0,1],[-2,0,2],[-1,0,1]]
        kernal = np.array(kernal)
        sobely = convolution(kernal,self.img)
        magnitude = mag(sobelx,sobely)
        print(sobelx)
        print(magnitude)
        cv2.imshow("origin",self.img)
        cv2.imshow("magnitude",magnitude)

    def resizeUI(self):
        self.flag = 0
        self.ui2.lineEdit_2.setEnabled(True)
        self.ui2.lineEdit_3.setEnabled(False)
        self.ui2.lineEdit_4.setEnabled(False)
        self.ui2.label.setText("X")
        self.ui2.label_2.setText("Y")
        self.ui2.label_3.setText("")
        self.ui2.label_4.setText("")
        self.Form2.show()

    def but2(self):
        if(self.flag == 0):
            self.img = cv2.imread("Q4_Image/SQUARE-01.png")
            self.img = cv2.resize(self.img, (int(self.ui2.lineEdit.text()), int(self.ui2.lineEdit_2.text())), interpolation=cv2.INTER_AREA)
            cv2.imshow("img",self.img)
            self.Form2.hide()
        if(self.flag == 1):
            self.img = translate(self.img, int(self.ui2.lineEdit.text()), int(self.ui2.lineEdit_2.text()))
            cv2.imshow("img",self.img)
            self.Form2.hide()
        if(self.flag == 2):
            self.img = rotate(self.img, int(self.ui2.lineEdit.text()))
            print(float(self.ui2.lineEdit_2.text()))
            matrix = np.array([[float(self.ui2.lineEdit_2.text()),0,0],[0,float(self.ui2.lineEdit_3.text()),0]])
            self.img = cv2.warpAffine(self.img,matrix,(self.img.shape[0],self.img.shape[1]))
            cv2.imshow("img",self.img)
            self.Form2.hide()
        if(self.flag == 3):
            matrix = np.array([[1,float(self.ui2.lineEdit_2.text()),0],[float(self.ui2.lineEdit_3.text()),1,0]])
            self.img = cv2.warpAffine(self.img,matrix,(self.img.shape[0],self.img.shape[1]))
            cv2.imshow("img",self.img)
            self.Form2.hide()

    def acc(self):
        self.img = cv2.imread("epoch_accuracy.png")
        cv2.imshow("acc",self.img)
        self.img2 = cv2.imread("epoch_loss.png")
        cv2.imshow("loss",self.img2)

    def modelstr(self):
        self.model = tf.keras.models.load_model("model.h5")
        self.model.summary()

    def test(self):
        self.model = tf.keras.models.load_model("model.h5")
        (x,y), (x_test, y_test) = datasets.cifar10.load_data()
        num = self.ui.lineEdit.text()
        #print(self.model.predict( x[int(num):int(num)+1] ))
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        langs = ['airplane', 'automobile', 'bird', 'cat','deer', 'dog', 'frog','hourse','ship','truck']
        print(len(self.model.predict( x[int(num):int(num)+1] )[0]))
        ax.bar(langs,self.model.predict( x[int(num):int(num)+1] )[0])
        plt.show()

    def hyper(self):
        print("hyperparameter")
        print("batch: 128")
        print("learning rate: 0.001")
        print("optimizer: Adam")

    def train(self):
        langs = ['airplane.png', 'automobile.png', 'bird.png', 'cat.png','deer.png', 'dog.png', 'frog.png','hourse.png','ship.png']
        img = []
        for i in langs:
            img.append(iimg.imread(i)) 
        f, ax = plt.subplots(3,3)
        ax[0,0].imshow(img[0])
        ax[0,1].imshow(img[1])
        ax[0,2].imshow(img[2])
        ax[1,0].imshow(img[3])
        ax[1,1].imshow(img[4])
        ax[1,2].imshow(img[5])
        ax[2,0].imshow(img[6])
        ax[2,1].imshow(img[7])
        ax[2,2].imshow(img[8])
        ax[0,0].set_title(langs[0])
        ax[0,1].set_title(langs[1])
        ax[0,2].set_title(langs[2])
        ax[1,0].set_title(langs[3])
        ax[1,1].set_title(langs[4])
        ax[1,2].set_title(langs[5])
        ax[2,0].set_title(langs[6])
        ax[2,1].set_title(langs[7])
        ax[2,2].set_title(langs[8])
        plt.show()

    def transUI(self):
        self.flag = 1
        self.ui2.lineEdit_2.setEnabled(True)
        self.ui2.lineEdit_3.setEnabled(False)
        self.ui2.lineEdit_4.setEnabled(False)
        self.ui2.label.setText("Xnew")
        self.ui2.label_2.setText("Ynew")
        self.ui2.label_3.setText("")
        self.ui2.label_4.setText("")
        self.Form2.show()
    
    def rotateUI(self):
        self.flag = 2
        self.ui2.lineEdit_2.setEnabled(True)
        self.ui2.lineEdit_3.setEnabled(True)
        self.ui2.lineEdit_4.setEnabled(False)
        self.ui2.label.setText("Angle")
        self.ui2.label_2.setText("X")
        self.ui2.label_3.setText("Y")
        self.ui2.label_4.setText("")
        self.Form2.show()

    def shearUI(self):
        self.flag = 3
        self.ui2.lineEdit_2.setEnabled(True)
        self.ui2.lineEdit_3.setEnabled(False)
        self.ui2.lineEdit_4.setEnabled(False)
        self.ui2.label.setText("shrx")
        self.ui2.label_2.setText("shry")
        self.ui2.label_3.setText("")
        self.ui2.label_4.setText("")
        self.Form2.show()




app = QtWidgets.QApplication(sys.argv)
ui = Window()
ui.Form.show()
sys.exit(app.exec_())
