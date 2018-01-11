# -*- coding: utf-8 -*-

import u
import os
from scipy import misc
from scipy import ndimage
import numpy as np
import math


oriDir = 'originalData/'
tgtDir = 'processedData/'
imgLength = 512
compressRatio = 0.2
compressLen = int(imgLength * compressRatio)


def flipImageMatrix(img):
    flipped_img = np.ndarray((img.shape), dtype='uint8')
    flipped_img[:, :, 0] = np.fliplr(img[:, :, 0])
    flipped_img[:, :, 1] = np.fliplr(img[:, :, 1])
    flipped_img[:, :, 2] = np.fliplr(img[:, :, 2])
    return flipped_img


def filterOnePic(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j, 0] not in range(128, 228) or x[i, j, 1] not in range(84, 184) or x[i, j, 2] not in range(100,
                                                                                                                200):
                x[i, j, 0] = 0
                x[i, j, 1] = 0
                x[i, j, 2] = 0


def preprocessImgMatrix(x):
    rimg = misc.imresize(x, [imgLength, imgLength], interp='nearest')
    rimg = rimg[compressLen:imgLength - compressLen, compressLen:imgLength - compressLen]
    return rimg


labelTransDict = {'常': 'chang', '黄': 'huang', '红': 'hong', '有': 'you', '无': 'wu', '紫': 'zi', '白': 'bai', }


def transferLabel(label):
    labels = label.split('-')
    for i in range(len(labels)):
        labels[i] = labelTransDict[labels[i]]
    return '-'.join(labels)

def augment(x, imgName):
    i = imgName.index('-')
    rimg = preprocessImgMatrix(x)
    label = imgName[i - 1:i + 4]
    label = transferLabel(label)
    # filterOnePic(rimg)  # 过滤
    for rotatIndex in range(4):
        misc.imsave(tgtDir + label + '-' + imgName[:i - 1] + '-' + str(rotatIndex) + '.jpg', rimg)
        misc.imsave(tgtDir + label + '-' + imgName[:i - 1] + '-r' + str(rotatIndex) + '.jpg',
                    flipImageMatrix(rimg))  # 镜像
        rimg = ndimage.rotate(rimg, 90)  # 翻转


def processData():
    u.updateDir(tgtDir)
    for fn in os.listdir(oriDir):
        for imgName in os.listdir(oriDir + fn):
            imgPath = oriDir + fn + '/' + imgName
            print('processing' + imgPath)
            try:
                x = u.getImageMatrix(imgPath)
            except Exception as e:
                print('----ERROR----')
                continue
            augment(x, imgName)
            augment(ndimage.rotate(x, 45, reshape=False), '45_' + imgName)


def loadData(labelIndex):  # 返回, x_train, y_train, x_test, y_test
    dataset = []
    labelset = []
    for fn in os.listdir(tgtDir):
        label = fn.split('-')[labelIndex]
        x = u.getImageMatrix(tgtDir + fn)
        dataset.append((x, label))
        if label not in labelset:
            labelset.append(label)

    print('index label')
    labelDict = {}
    for i in range(len(labelset)):
        print(i, labelset[i])
        labelDict[labelset[i]] = i

    print('shuffling')
    sampleSize = len(dataset)
    height, width, channels = dataset[0][0].shape
    X = np.zeros((sampleSize, height, width, channels), dtype='uint8')
    y = np.zeros(sampleSize, dtype='uint8')
    seq = np.random.permutation(sampleSize)
    c = 0
    for i in seq:
        X[c] = dataset[i][0]
        y[c] = labelDict[dataset[i][1]]
        c += 1
    print('dataset loading finish!')
    return X, y


def getFilterRange():
    c1, c2, c3 = [], [], []
    for i in range(256):
        c1.append(0)
        c2.append(0)
        c3.append(0)
    dirPath = 'preparedDataset/'
    for fn in os.listdir(dirPath):
        x = u.getImageMatrix(dirPath + fn)
        for r in x:
            for pix in r:
                c1[pix[0]] += 1
                c2[pix[1]] += 1
                c3[pix[2]] += 1
    print ('c1')
    print (c1)
    print ('c2')
    print (c2)
    print ('c3')
    print (c3)


def filterPic():
    tgtDir = 'filteredDataset/'
    u.updateDir(tgtDir)

    for fn in os.listdir(oriDir):
        for imgName in os.listdir(oriDir + fn):
            x = u.getImageMatrix(oriDir + fn + '/' + imgName)
            filterOnePic(x)
            print('saved to: ', tgtDir + fn)
            misc.imsave(tgtDir + fn, x)

            # getFilterRange()
            # 压缩和旋转图片()
            # loadData(0)
            # filterPic()



def lookData(dataType):
    tgtDirPath = 'look/'
    u.updateDir(tgtDirPath)
    import shutil

    c = 0
    for fn in os.listdir(oriDir):
        for imgName in os.listdir(oriDir + fn):
            i = imgName.index('-')
            imgPath = oriDir + fn + '/' + imgName
            label = imgName[i - 1:i + 4].split('-')[dataType]
            shutil.copyfile(imgPath, tgtDirPath + label + str(c) + '.jpg')
            print(c)
            c += 1


# processData()