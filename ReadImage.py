import os
from PIL import Image
import numpy as np
import torch 

def read_image(inputFile):
	img = np.asarray(Image.open(inputFile))
	# Scaling the image so that the values are in the range of 0 to 1
	img = img / 255.0
	return img

def GetDataBase(folder_path):
    image_vector = []
    #one hot vector of files (0, 0, 0, 0 , 1) for age in range (60 > 80)
    label_vector = []

    if not os.path.isdir(folder_path):
        print(f"{folder_path} is not a valid directory.")
        return
    
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            name = file_name[:-4]
            age = name[-2:]
            image = read_image(folder_path + file_name)

            if int(age) < 15:
                label = [1, 0, 0, 0, 0]
            elif int(age) >= 15 and int(age) < 30:
                label = [0, 1, 0, 0, 0]
            elif int(age) >= 39 and int(age) <= 40:
                label = [0, 0, 1, 0, 0]
            elif int(age) >= 41 and int(age) < 60:
                label = [0, 0, 0, 1, 0]
            else:
                label = [0, 0, 0, 0, 1]
            
            image_vector.append(torch.Tensor(image).permute(2,0,1))
            label_vector.append(label)

    return image_vector, label_vector  
