import torch
from PIL import Image
import torchvision.transforms as transforms

def img_to_tensor(img_path):
    img = Image.open(img_path).convert('RGB') #this is now a numpy array
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((250,250))
    ])
    
    return transform(img)

    
    