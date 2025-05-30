import models as models
import torch
from load_data import dataloader_lfw_embeddings
from utils import EmbeddingLoader
from torch.utils.data import DataLoader
import os

embedding_model = None

# from lfw_bench import dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_resnet():
    global embedding_model
    dataloader,num_classes, class_weights = dataloader_lfw_embeddings()
    
    #change the last layer in resnet to output 'num_classes'
    embedding_model = models.resnet_embedder(num_classes)
    optimizer = torch.optim.Adam(embedding_model.parameters(), lr=0.0001)
    num_epochs = 10
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    for epoch in range(num_epochs):
        for images,labels in dataloader:
            optimizer.zero_grad()
            _,logits = embedding_model(images)
            loss = embedding_model.loss_function(logits,labels,class_weights)
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1}, Total_Loss: {loss.item()}")
    torch.save(embedding_model.state_dict(), '/face_attendance/benchmarking/embed_model.pth')

    # return embedding_model, '/face_attendance/benchmarking/embed_model.pth'

#can define other training models also here



def train_classifier(pos_set, neg_set,name,model2,optimizer):
    
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
    
    os.makedirs(f'/data/facebiometrics/updated/ML_work/our_method/Implementation/one-person-model/models_lfw/{name}', exist_ok=True)
    torch.save(model2.state_dict(), f'/data/facebiometrics/updated/ML_work/our_method/Implementation/one-person-model/models_lfw/{name}/model_{name}.pth')
    


