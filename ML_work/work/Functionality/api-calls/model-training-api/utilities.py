import sys
sys.path.append('/required/facenet_pytorch')
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from facenet_pytorch import InceptionResnetV1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TensorDataset2(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        return torch.tensor(self.image_list[idx], dtype=torch.float32)

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