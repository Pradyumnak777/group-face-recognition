import sys
print("Python interpreter:", sys.executable)


from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import os,cv2
import numpy as np
import models
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
import random
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TensorDataset(Dataset):
    def __init__(self, tensor_list):
        self.tensor_list = tensor_list
    
    def __len__(self):
        return len(self.tensor_list)
    
    def __getitem__(self, idx):
        return self.tensor_list[idx] 
    
class LFWDataset_img(Dataset):
    def __init__(self,dict):
        self.dict = dict
    
    def __len__(self):
        return len(self.dict)
    
    def __getitem__(self,idx): #return the tensor(img) and label, to create a dictionary of {name:embedding}
        label, tensor = self.dict[idx]
        return tensor, label
    
class LFWDataset_embedding(Dataset):
    def __init__(self,dict,name):
        self.dict = dict
        self.name = name
    
    def __len__(self):
        return len(self.dict)
    
    def __getitem__(self,idx): 
        label, embedding = self.dict[idx]
        if label == self.name:
            new_label = torch.tensor([1, 0], dtype=torch.float).to(device)
            return embedding, new_label
        else:
            new_label = torch.tensor([0, 1], dtype=torch.float).to(device)
            return embedding, new_label
        
class LFWDataset_embedding_test(Dataset):
    def __init__(self,dict):
        self.dict = dict
            
    def __len__(self):
        return len(self.dict)
    
    def __getitem__(self,idx): 
        name, embedding = self.dict[idx]
        return embedding, name

class EmbeddingLoader(Dataset): #input numpy arrays
    def __init__(self, pos_array, neg_array):
        super().__init__()
        self.data = np.concatenate((pos_array, neg_array), axis=0)
        self.labels = np.concatenate((np.ones(pos_array.shape[0]), np.zeros(neg_array.shape[0])))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index], 1-self.labels[index]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.float)
    
#this will take data from face_dataset_lfw_train
class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data, self.labels = self._load_data()
        self.class_to_indices = self._create_class_index()
        

    def _load_data(self):
        # Assuming each subfolder in root_dir is named by the class label (e.g., /root_dir/class1, /root_dir/class2)
        data = []
        labels = []
        for label in os.listdir(self.root_dir):
            label_dir = os.path.join(self.root_dir, label)
            if os.path.isdir(label_dir):
                for img_name in os.listdir(label_dir):
                    if img_name.lower().endswith('.jpg'):
                        img_path = os.path.join(label_dir, img_name)
                        data.append(img_path)
                        labels.append(label)
        return data, labels

    def _create_class_index(self):
        class_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)
        return class_to_indices

    def __getitem__(self, index):
        # Load anchor image
        anchor_path = self.data[index]
        anchor_label = self.labels[index]
        anchor = Image.open(anchor_path)

        # Load positive image
        positive_index = random.choice(self.class_to_indices[anchor_label])
        while positive_index == index:
            positive_index = random.choice(self.class_to_indices[anchor_label])
        positive_path = self.data[positive_index]
        positive = Image.open(positive_path)

        # Load negative image
        negative_label = random.choice(list(set(self.labels) - {anchor_label}))
        negative_index = random.choice(self.class_to_indices[negative_label])
        negative_path = self.data[negative_index]
        negative = Image.open(negative_path)

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

    def __len__(self):
        return len(self.data)
        
        
        
    
    
        
def get_lfw_embeddings(path, num_classes = 316):
    from training import embedding_model
    # embedding_model = models.resnet_embedder(num_classes)
    # embedding_model.load_state_dict(torch.load(f'/face_attendance/benchmarking/embed_model.pth', map_location=device))
    embedding_model = InceptionResnetV1(pretrained='vggface2').eval()

    embedding_model.to(device)
    embedding_model.eval()
        
    # img_tensor_list = []
    # embeddings_list = []
    embedding_dict = {}
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((160, 160)),
        ])
    dict = {}
    i = 0
    for sample in os.listdir(path):
        for file in os.listdir(os.path.join(path,sample)):
            img_path = os.path.join(path,sample,file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = transform(img)
            dict[i] = (sample,img_tensor)  # eg - { 1:(pradyumna, tensor[...]) }
            i += 1
        
    dataset = LFWDataset_img(dict)
    batch_size = 32
    dataloader = DataLoader(dataset,batch_size=batch_size, shuffle=True)
    with torch.no_grad():
        j = 0
        for images,labels in dataloader:
            images = images.to(device)
            embeddings = embedding_model(images)
            embeddings = torch.flatten(embeddings, start_dim=1)
            embeddings = F.normalize(embeddings)
            #we have 32 embeddings (batch size)
            for embedding,label in zip(embeddings,labels):
                embedding_dict[j] = (label, embedding)
                j += 1
        
    return embedding_dict
    
def get_embeddings(path, num_classes = 316):
    embedding_model = models.resnet_embedder(num_classes)
    
    # embedding_model.load_state_dict(torch.load(f'/face_attendance/benchmarking/embed_model.pth', map_location=device))
    embedding_model = InceptionResnetV1(pretrained='vggface2').eval()
    embedding_model.to(device)
    img_tensor_list = []
    embeddings_list = []
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((160, 160)),
        ])
    for sample in os.listdir(path):
        img_path = os.path.join(path,sample)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img)
        img_tensor_list.append(img_tensor)
    
    dataset = TensorDataset(img_tensor_list)
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)  
            embeddings = embedding_model(batch)
            embeddings = torch.flatten(embeddings, start_dim=1)
            embeddings = F.normalize(embeddings)
            #we have 32 embeddings (batch size)
            for embedding in embeddings:
                embeddings_list.append(embedding)
    
    return embeddings_list

def get_embeddings_from_list(tensor_list, num_classes=316):
    # embedding_model = models.resnet_embedder(num_classes)    
    # embedding_model.load_state_dict(torch.load(f'/face_attendance/benchmarking/embed_model.pth', map_location=device))
    embedding_model = InceptionResnetV1(pretrained='vggface2').eval()
    embedding_model.to(device)
    
    embeddings_list = []
    dataset = TensorDataset(tensor_list)
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

# def get_embeddings_facenet():
#     new_model = torch.load('/face_attendance/models_npys/20180402-114759-vggface2.pt').eval()
    
    