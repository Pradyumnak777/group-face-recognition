import torch
from benchmarking.utils import EmbeddingLoader
from torch.utils.data import DataLoader
import os
import shutil

embedding_model = None

# from lfw_bench import dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_classifier(pos_set, neg_set,model2,optimizer):
    #clear temp_model
    if os.path.exists('/face_attendance/api-calls/training-api/temp_model'):
        shutil.rmtree('/face_attendance/api-calls/training-api/temp_model')
    
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
    
    os.makedirs(f'/face_attendance/api-calls/training-api/temp_model', exist_ok=True)
    torch.save(model2.state_dict(), f'/face_attendance/api-calls/training-api/temp_model/model.pth')
    


