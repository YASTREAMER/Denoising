import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import torch
from torch import  nn
from torchvision import transforms ,datasets
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler

from tqdm.notebook import tqdm

import cv2

from DataLoader import dataloader

class Main:

    def __init__(self) -> None:

        self.dataloader = dataloader()

        #Setting the size of the training dataset
        self.datasize=64

        #Checking if cuda is available or not 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        #List for storing the training dataset
        self.imagestrain=[]
        self.labeltrain=[]

        #Init the list to store pixel value of each image
        self.pixelvalue=[]

    def MainTrain(self) -> None:

        #Loading the training dataset
        self.imagestrain=self.dataloader.GetImages(dir="Train/low/")
        self.labeltrain=self.dataloader.GetImages(dir="Train/high/")

        for i in range(self.datasize):
            self.pixelvalue.append(self.dataloader.GetPixelValues(self.imagestrain[i]))
            



if __name__ == "__main__":

    main = Main()
    main.MainTrain()

