import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FaceEmbedder(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        
        # os.environ['TORCH_HOME'] = '/scratch/facialbiometrics'       
        resnet = resnet18(pretrained=True)
        self.resnet_conv = nn.Sequential(*list(resnet.children())[:-2]) #removes the pooling and fully connected layers, to form embedding
        
        # self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1) 
        
        self.conv4 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=1)
        self.conv5 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=3, stride=2)
        self.conv6 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1)
        
        # for classification
        self.lin1 = nn.Linear(in_features=512*3*3, out_features=class_num) 

    def encode(self, x, w, h):
        x = self.resnet_conv(x)
        x = F.relu(x)
        # x = self.pool1(x)  # h, w here?
        embedding = F.interpolate(x, size=(3, 3), mode='bilinear', align_corners=False)
        x = F.relu(x)
        return x, embedding

    def decode(self, x, w, h):
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        
        x = F.interpolate(x, size=(w, h), mode='bilinear', align_corners=False)
        return x  # returns reconstructed image

    def generate_logits(self, embedding):
        embedding_flat = embedding.view(embedding.size(0), -1)
        predicted_logits = self.lin1(embedding_flat)
        return predicted_logits

    def classification_loss(self, logits, labels, class_wts):
        loss = nn.CrossEntropyLoss(weight=class_wts)
        loss_val = loss(logits, labels)
        return loss_val

    def forward(self, x, w, h):  # to remember input height and width
        x, face_embedding = self.encode(x, w, h)
        x = self.decode(x, w, h)
        face_embedding = F.normalize(face_embedding)
        # classification after flattening embeddings
        face_embeddings = face_embedding.flatten(start_dim=1)
        classifier_logits = self.generate_logits(face_embeddings)
        
        return x, face_embeddings, classifier_logits

    def reconstruction_loss(self, x, reconstruction):
        loss_val = F.mse_loss(reconstruction, x)
        return loss_val


def get_embeddings(img_tensor, num_classes):
    #pass it through the model
    model = FaceEmbedder(num_classes)
    model.load_state_dict(torch.load('/interns/iittcseitr24_10/face_attendance/face_embedding/uiui.pth', map_location=device))
    model.to(device)
    model.eval()
    with torch.no_grad():
        _, embedding, _ = model(img_tensor, img_tensor.size(2), img_tensor.size(3))
    
    return embedding
    
    