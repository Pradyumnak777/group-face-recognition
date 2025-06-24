import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from required.facenet_pytorch import InceptionResnetV1
import numpy as np
from torchvision import transforms
import cv2
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([  #image neeeds to be in PIL format for this to work
    transforms.Resize((160, 160)),
    transforms.ToTensor(),  # converts to [0, 1]
    transforms.Normalize([0.5]*3, [0.5]*3) #converts to [-1,1]
])


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
    def __len__(self):
        return len(self.embedding_dict)
    def __getitem__(self,idx):
        label, embedding = self.embedding_dict[idx]  #{name: [.....]} format
        if label == self.name: #it is positive
            new_label = torch.tensor([1, 0], dtype=torch.float).to(device)
            return embedding, new_label
        else:
            new_label = torch.tensor([0, 1], dtype=torch.float).to(device)
            return embedding, new_label

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
    embedding_model = InceptionResnetV1(pretrained='vggface2').eval()
    embedding_model.to(device)
    
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
    embedding_model = InceptionResnetV1(pretrained='vggface2').eval()
    embedding_model.to(device)
    #convert image to tensor
    tensor_image = transform(pil_image)  #dims -> 3,160,160
    tensor_image = tensor_image.unsqueeze(0) #dims -> 1,3,160,160
    tensor_image = tensor_image.to(device)
    with torch.no_grad():
        embedding = embedding_model(tensor_image)
        
    return embedding #dims -> 1,512