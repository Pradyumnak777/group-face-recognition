import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from api_calls_and_functions.shared_material.packages.facenet_pytorch import InceptionResnetV1
import cv2
import random
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([  #image neeeds to be in PIL format for this to work
    transforms.Resize((160, 160)),
    transforms.ToTensor(),  # converts to [0, 1]
    transforms.Normalize([0.5]*3, [0.5]*3) #converts to [-1,1]
])

embedding_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

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
        

class TensorDataset2(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        return transform(self.image_list[idx])
    


class BoostingDataset(Dataset):
    def __init__(self,name,embedding_dict):
        self.name = name
        self.embedding_dict = embedding_dict
        self.keys = list(embedding_dict.keys()) #[name1, name2, ...]
    def __len__(self):
        return len(self.embedding_dict)
    def __getitem__(self,idx):
        key = self.keys[idx]
        embedding = self.embedding_dict[key]  #{name: [.....]} format
        if key == self.name: #it is positive
            new_label = torch.tensor([1, 0], dtype=torch.float).to(device)
            return embedding.squeeze(0), new_label
        else:
            new_label = torch.tensor([0, 1], dtype=torch.float).to(device)
            return embedding.squeeze(0), new_label

class EmbeddingLoader(Dataset): #input numpy arrays
    def __init__(self, pos_array, neg_array):#pos_array is a tensor list, neg_array is a numpy (1000,512)
        super().__init__()
        self.data = np.concatenate((pos_array, neg_array), axis=0)
        self.labels = np.concatenate((np.ones(pos_array.shape[0]), np.zeros(neg_array.shape[0])))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index], 1-self.labels[index]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.float)

class custom_head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64)
        )

    def forward(self, x):
        x = self.net(x)
        # **critical** – unit-length vectors ⇨ cosine distance = Euclidean distance
        x = F.normalize(x, p=2, dim=1)
        return x

    @staticmethod         
    def triplet_loss(a, p, n, margin=0.3):
        # inputs already unit-norm → Euclidean² = 2 – 2·cos
        d_pos = F.pairwise_distance(a, p)
        d_neg = F.pairwise_distance(a, n)
        return torch.mean(F.relu(d_pos - d_neg + margin))


# class TripletDataset(Dataset):
#     def __init__(self, pos_embeds, neg_embeds):
#         self.pos_embeds = pos_embeds
#         self.neg_embeds = neg_embeds
#     def __len__(self):
#         return len(self.pos_embeds) * 20 
#     def __getitem__(self, idx):
#         pos_indices = random.sample(range(len(self.pos_embeds)), 2) #so, anchor != pos
#         anchor = self.pos_embeds[pos_indices[0]]
#         positive = self.pos_embeds[pos_indices[1]]
#         neg_idx = random.randint(0,len(self.neg_embeds) - 1)
#         negative = self.neg_embeds[neg_idx]
#         return (
#             torch.tensor(anchor, dtype=torch.float32),
#             torch.tensor(positive, dtype=torch.float32),
#             torch.tensor(negative, dtype=torch.float32),
#         )

class TripletDataset(Dataset):
    def __init__(self, student_dict):
        self.student_dict = student_dict
        self.labels = list(student_dict.keys())
        
        self.label_embeds_dict = {    # {label:embeds1, label2:e2, ...}
            label: list(range(len(embeds)))
            for label, embeds in student_dict.items()
        }
    
    def __len__(self):
        return len(self.labels) * 20
    
    def __getitem__(self, index):
        #select an anchor at random
        anchor_label = random.choice(self.labels)
        pos_embeds = self.student_dict[anchor_label]
        # print(f"pos_embeds length from dict: {len(pos_embeds)}")
        anchor, positive = random.sample(list(pos_embeds), 2)
        
        neg_label = random.choice([l for l in self.labels if l != anchor_label])
        negative = random.choice(self.student_dict[neg_label])

        return (
            torch.tensor(anchor, dtype=torch.float32),
            torch.tensor(positive, dtype=torch.float32),
            torch.tensor(negative, dtype=torch.float32)
        )
        

def train_head(student_embeddings_dict, model2, optimizer):
    device = next(model2.parameters()).device
    triplet_dataset = TripletDataset(student_embeddings_dict)     
    triplet_loader = DataLoader(triplet_dataset, batch_size=32, shuffle=True)
    num_epochs = 15
    model2.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for anchor,positive,negative in triplet_loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            optimizer.zero_grad()

            anchor_out = model2(anchor)
            positive_out = model2(positive)
            negative_out = model2(negative)
            
            loss = model2.triplet_loss(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(triplet_loader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
            
    return model2

def train_classifier(pos_set, neg_set, model2, optimizer):
    
    #prepare dataloader (loading embeddings)
    #now pass to dataloader to load embeddings
    
    dataset = EmbeddingLoader(pos_set,neg_set)
    dataloader = DataLoader(dataset,batch_size=32,shuffle=True)
    #embedding classifier
    num_epochs = 20
    for epoch in range(num_epochs):
        for images,labels in dataloader:
            images = images.to(device)
            optimizer.zero_grad()
            _,probs = model2(images)
            loss = model2.loss(probs,labels)
            loss.backward()
            optimizer.step()
            total += loss.item()
            
        print(f'Epoch {epoch+1:02d} | avg loss {total/len(dataloader):.4f}')
    
    return model2
    # os.makedirs(f'/data/facebiometrics/updated/ML_work/our_method/Implementation/one-person-model/models_lfw/{name}', exist_ok=True)
    # torch.save(model2.state_dict(), f'/data/facebiometrics/updated/ML_work/our_method/Implementation/one-person-model/models_lfw/{name}/model_{name}.pth')
    
def augment_positive_set(pil_image, num = 15): #image needs to be converted to an rgb_numpy_image
    augmented_faces = [] #will be a list of numpy-aray images
    augmented_faces.append(pil_image)
    image = np.array(pil_image)
    for i in range(num):
        face = image.copy()
        face = cv2.convertScaleAbs(face, alpha=np.random.uniform(0.9, 1.1), beta=np.random.randint(-25, 25))
        angle = np.random.uniform(-10, 10)
        h, w = face.shape[:2]
        matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        face = cv2.warpAffine(face, matrix, (w, h))
        if np.random.rand() > 0.5:
            face = cv2.flip(face, 1)
            
        if np.random.rand() > 0.7:
            face = cv2.GaussianBlur(face, (3, 3), 0) #kernel size is 3x3 and std deviation is set to default=0
        
        face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        augmented_faces.append(face)
    
    return augmented_faces #array of PIL face(so torch.transform can be applied)

def get_embeddings_from_image_list(image_list):
    # embedding_model = models.resnet_embedder(num_classes)    
    # embedding_model.load_state_dict(torch.load(f'/face_attendance/benchmarking/embed_model.pth', map_location=device))
    # embedding_model.to(device)
    
    embeddings_list = []
    dataset = TensorDataset2(image_list)
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)  
            batch = batch.squeeze(1)  
            embeddings = embedding_model(batch)
            embeddings = torch.flatten(embeddings, start_dim=1)
            embeddings = F.normalize(embeddings)
            #we have 32 embeddings (batch size)
            for embedding in embeddings:
                embeddings_list.append(embedding)
    
    return embeddings_list

def get_single_image_embeddings(pil_image):
    # embedding_model.to(device)
    #convert image to tensor
    tensor_image = transform(pil_image)  #dims -> 3,160,160
    tensor_image = tensor_image.unsqueeze(0) #dims -> 1,3,160,160
    tensor_image = tensor_image.to(device)
    with torch.no_grad():
        embedding = embedding_model(tensor_image)
        
    return embedding.detach().cpu() #dims -> 1,512


    






