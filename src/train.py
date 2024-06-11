import torch
from tqdm import tqdm
import random
from torchmetrics.image import PeakSignalNoiseRatio

from dataloader import DataLoader
from model import Model

class Train:

    def __init__(self,batchsize,epochs,lr,device) -> None:

        #Setting the device that is available for training
        self.device = device

        #Initialize the data loader
        self.DataLoader = DataLoader()
        
        #Setting the number of epochs
        self.epochs=epochs

        #Setting the batch size
        self.batchsize=batchsize

        #Setting the learning rate
        self.lr=lr

        #Loading the training dataset directory
        self.imagestrain,self.imgnum=self.DataLoader.GetImages(dir="Train/low/")
        self.labeltrain,self.labelnum=self.DataLoader.GetImages(dir="Train/high/")

        

    def model_train(self):

        portion =0.45
        #Initialize the model
        self.model = Model().to(self.device)

        #Optimizer, loss function
        self.opt = torch.optim.Adam(self.model.parameters()
                                    ,lr= self.lr)
        
        self.mse = torch.nn.BCELoss(reduction="sum").to(self.device)
        self.PSNR = PeakSignalNoiseRatio().to(self.device)

        print(f"The amount of loss used is :-{portion}")

        for epoch in tqdm(range(self.epochs)):
            #Getting the pixel value of the image in tensor
            for i in range(self.batchsize):

                randomimage= random.randint(0,self.imgnum-1)

                image = self.DataLoader.GetPixelValues(self.imagestrain[randomimage])
                label = self.DataLoader.GetPixelValues(self.labeltrain[randomimage]) 
                image = image.cuda()
                label = label.cuda()
                x_recontructed,mu,sigma=self.model(image)
                recontructed_loss=self.mse(x_recontructed,label)
                kl_div=-torch.sum(1+torch.log(sigma.pow(2))-mu.pow(2)-sigma.pow(2))

                loss_VAE =recontructed_loss+kl_div

                total_loss = self.PSNR(x_recontructed,label) + (loss_VAE*portion)
                self.opt.zero_grad()
                total_loss.backward()
                self.opt.step()

            if (epoch+1) % 10==0: 

                print(f'Epoch:{epoch+1}, Loss:{total_loss.item():.4f}')
                print(f"The PSNR Loss is: - {self.PSNR(x_recontructed,label)} +  and the MSE Loss is :- {(loss_VAE*portion).item()}")
                
        
                # deout=deout.detach()
                # deout=self.DataLoader.ImageConversion(deout)
                # # print(deout)
                # plt.imshow(deout)
                # plt.show()
            



            
