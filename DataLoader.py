import os 
import matplotlib.pyplot as plt

class dataloader:

    def __init__(self) -> None:
        pass

    def GetImages(self,dir)->list:
        images=[]
        for image in os.listdir(dir):
            if (image.endswith(".png")):
                images.append(dir+image)

        return images
    
    def GetPixelValues(self,dir) -> list:
        
        pixelvalue=plt.imread(dir)
        print(pixelvalue.shape)
        return pixelvalue

