import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utilities import EmbeddingLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class bin_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(in_features=512, out_features=256)
        self.lin2 = nn.Linear(in_features=256, out_features=32)
        self.lin3 = nn.Linear(in_features=32, out_features=2)
        
    
    def forward(self,x): #x is an embedding here
        # torch.flatten(x,1) #flatten the embedding
        x = self.lin1(x)
        x = self.lin2(x)
        logits = self.lin3(x)
        probs = F.softmax(logits)
        return logits,probs
    
    def loss(self,probabilities,labels):
        labels = labels.to(probabilities.device)
        loss = nn.BCELoss()
        loss_val = loss(probabilities, labels)
        return loss_val

def train_classifier(pos_set, neg_set, model2, optimizer):
    
    #prepare dataloader (loading embeddings)
    #now pass to dataloader to load embeddings
    
    dataset = EmbeddingLoader(pos_set,neg_set)
    dataloader = DataLoader(dataset,batch_size=32,shuffle=True)
    #embedding classifier
    num_epochs = 5
    for epoch in range(num_epochs):
        for images,labels in dataloader:
            images = images.to(device)
            optimizer.zero_grad()
            _,probs = model2(images)
            loss = model2.loss(probs,labels)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Total_Loss: {loss.item()}")
    
    return model2
    # os.makedirs(f'/data/facebiometrics/updated/ML_work/our_method/Implementation/one-person-model/models_lfw/{name}', exist_ok=True)
    # torch.save(model2.state_dict(), f'/data/facebiometrics/updated/ML_work/our_method/Implementation/one-person-model/models_lfw/{name}/model_{name}.pth')
    







