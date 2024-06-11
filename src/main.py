import numpy as np
import torch
# import time

from train import Train

class Main:

    def __init__(self,batchsize,epochs,lr) -> None:

        #Checking if cuda is available or not 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.Train = Train(batchsize,epochs,lr,self.device)

        #Setting the image size
        self.imagesize = (400, 600, 3)

        #List for storing the training dataset
        self.imagestrain=[]
        self.labeltrain=[]

    def MainTrain(self) -> None:

        # portion=0.4       

        # while portion<0.5:
            #Start the training
        self.Train.model_train()        

            # portion+=0.01


if __name__ == "__main__":

    main = Main(batchsize=256,epochs=200,lr=3e-4)
    main.MainTrain()
    print("exit")

    

