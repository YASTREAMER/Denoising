import os 
import PIL.Image
import matplotlib.pyplot as plt
from torchvision import transforms 
import torch

class DataLoader:

    def __init__(self) -> None:
        pass

    def GetImages(self,dir)->list:
        images=[]
        imgnum=0
        for image in os.listdir(dir):
            if (image.endswith(".png")):
                images.append(dir+image)
                imgnum+=1

        return images, imgnum
    
    def GetPixelValues(self,dir) -> torch.Tensor:
        
        pixelvalue=plt.imread(dir)
        transformation = transforms.Compose([ 
            transforms.ToTensor(),
            transforms.Resize(size=(400,400))
            ]) 
        pixelvalue=transformation(pixelvalue)
        return pixelvalue
    
    def ImageConversion(self,img) -> PIL.Image:

        tranformation = transforms.Compose([ 
            transforms.ToPILImage()
        ])

        imgtrans=tranformation(img)

        return(imgtrans)

