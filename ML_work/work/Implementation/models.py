import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1


class updated_ResNetEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18
        resnet = resnet18(pretrained=False)
        self.resnet_embedding = nn.Sequential(*list(resnet.children())[:-1])  # Remove last layers

    def forward(self, x):
        embedding = self.resnet_embedding(x)
        embedding = torch.flatten(embedding, 1)  # Flatten the output to a vector
        return embedding

    def triplet_loss(self, anchor, positive, negative, margin=1.0):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0).mean()
        return loss

    


# class resnet_embedder(nn.Module):
#     def __init__(self,num_classes):
#         super().__init__() 
#         from torchvision.models import resnet18
#         resnet = resnet18(pretrained=True)
#         resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
#         self.resnet_embedding = nn.Sequential(*list(resnet.children())[:-2])
#         self.resnet_remaining = nn.Sequential(*list(resnet.children())[-2:])
    
#     def forward(self,x):
#         x = self.resnet_embedding(x)
#         embedding = x #to be returned
#         x = self.resnet_remaining[0](x) #the avg pooling layer in resnet
#         x = torch.flatten(x,1)
#         logits = self.resnet_remaining[1](x)
        
#         return embedding,logits
    
#     def loss_function(self,logits,labels,class_wts):
#         loss = nn.CrossEntropyLoss(weight=class_wts)
#         loss_val = loss(logits, labels)
#         return loss_val

class resnet_embedder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.facenet = InceptionResnetV1(pretrained='vggface2')
        
        self.fc = nn.Linear(512, num_classes)  # InceptionResnetV1 outputs 512-dim embeddings

    def forward(self, x):
        embedding = self.facenet(x)
        logits = self.fc(embedding)
        
        return embedding, logits
    
    def loss_function(self, logits, labels, class_wts):
        # Define the loss function with class weights
        loss = nn.CrossEntropyLoss(weight=class_wts)
        loss_val = loss(logits, labels)
        return loss_val

    

        

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
        