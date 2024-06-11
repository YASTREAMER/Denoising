import torch
from torch import  nn
from torchvision import transforms ,datasets
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler

class Model(nn.Module):

    def __init__(self) -> None:

        super().__init__()

        #Var need for the model to work
        self.IndicesMax1 = None
        self.IndicesMax2 = None
        self.IndicesMax3 = None

        self.temp1=None
        self.temp2=None   
        self.temp3=None

        #Activation layer
        self.LeakyRelu = nn.LeakyReLU()
        self.Sigmoid = nn.Sigmoid()


        #Encoder class
        self.Conv1 = nn.Conv2d(3,16,3)
        self.MaxPool1=nn.MaxPool2d(4,stride=4,return_indices=True)
        self.Conv2 = nn.Conv2d(16,32,10)
        self.Conv3 = nn.Conv2d(32,64,15)
        self.Conv4 = nn.Conv2d(64,256,6)
        self.MaxPool2=nn.MaxPool2d(2,stride=2,return_indices=True)
        self.Conv5 = nn.Conv2d(256,64,10)
        self.Conv6 = nn.Conv2d(64,4,8)
        self.MaxPool3=nn.MaxPool2d([2,2],stride=1,return_indices=True)
        self.Flatten = nn.Flatten()
        self.Lin1 = nn.Linear(in_features=324,out_features=64)
        self.Mu = nn.Linear(in_features=64,out_features=8)
        self.Sigma = nn.Linear(in_features=64,out_features=8)

        #Decoder 
        self.UnLin1 = nn.Linear(in_features=8,out_features=64)
        self.UnLin2 = nn.Linear(in_features=64,out_features=324)
        self.UnFlatten = nn.Unflatten(1,(18,18)) 
        self.UnMaxpool1 = nn.MaxUnpool2d([2,2],stride=1)
        self.UnConv1 = nn.ConvTranspose2d(4,64,8)
        self.UnConv2 = nn.ConvTranspose2d(64,256,10)
        self.UnMaxpool2 = nn.MaxUnpool2d(2,stride=2)
        self.UnConv3 = nn.ConvTranspose2d(256,64,6)
        self.UnConv4 = nn.ConvTranspose2d(64,32,15)
        self.UnConv5 = nn.ConvTranspose2d(32,16,10)
        self.UnMaxpool3 = nn.MaxUnpool2d(4,stride=4)
        self.UnConv6 = nn.ConvTranspose2d(16,3,3)

    def Encoder(self,x):

        #Encoder
        x = self.Conv1(x)
        self.temp3 = x.shape
        x = self.LeakyRelu(x)
        x, self.IndicesMax1 = self.MaxPool1(x)
        x = self.Conv2(x)
        x = self.LeakyRelu(x)
        x = self.Conv3(x)
        x = self.LeakyRelu(x)
        x = self.Conv4(x)
        x = self.LeakyRelu(x)
        self.temp2 = x.shape
        x, self.IndicesMax2 = self.MaxPool2(x)
        x = self.Conv5(x)
        x = self.LeakyRelu(x)
        x = self.Conv6(x)
        x = self.LeakyRelu(x)
        self.temp1 = x.shape
        x,self.IndicesMax3 = self.MaxPool3(x) 
        x = self.Flatten(x)
        x = self.Lin1(x)
        x = self.LeakyRelu(x)
        mu, sigma = self.Mu(x),self.Sigma(x)

        return mu ,sigma

    def Decoder(self,x):

        #Decoder
        x = self.UnLin1(x)
        x = self.LeakyRelu(x)
        x = self.UnLin2(x)
        x = self.LeakyRelu(x)
        x = self.UnFlatten(x)
        x = self.UnMaxpool1(x,self.IndicesMax3,output_size=self.temp1[1:3])
        x = self.LeakyRelu(x)
        x = self.UnConv1(x)
        x = self.LeakyRelu(x)
        x = self.UnConv2(x)
        x = self.UnMaxpool2(x,self.IndicesMax2,output_size=self.temp2[1:3])
        x = self.LeakyRelu(x)
        x = self.UnConv3(x)
        x = self.LeakyRelu(x)
        x = self.UnConv4(x)
        x = self.LeakyRelu(x)
        x = self.UnConv5(x)
        x = self.UnMaxpool3(x,self.IndicesMax1,output_size=self.temp3[1:3])
        x = self.UnConv6(x)
        reconstructed = self.Sigmoid(x)
        return reconstructed

    def forward(self,x):

        mu,sigma=self.Encoder(x)
        epsilon=torch.rand_like(sigma)
        z_parameterized=mu+sigma*epsilon
        x_recontructed=self.Decoder(z_parameterized)
        
        return x_recontructed,mu,sigma

        

