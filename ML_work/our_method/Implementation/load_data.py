import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler,Dataset, Subset
from sklearn.utils import compute_class_weight
import numpy as np


#data loading to generate embeddings
def dataloader_lfw_embeddings():    
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((250,250))
    ])

    dataset = ImageFolder(root='/face_attendance/modified_lfw_datasets/embedding_lfw', transform=transform)
    #class weights computing (since classes are imbalanced in the modified lfw dataset)
    class_weights = compute_class_weight('balanced', classes = np.unique(dataset.targets), y=dataset.targets)
    sample_weights = np.zeros(len(dataset.targets))
    for idx, label in enumerate(dataset.targets):
        sample_weights[idx] = class_weights[label]
        
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    dataloader = DataLoader(dataset, batch_size=64,sampler= sampler)
    
    num_classes = len(dataset.classes)

    return dataloader,num_classes, class_weights