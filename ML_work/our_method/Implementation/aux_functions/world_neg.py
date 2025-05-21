#generate embeddings from embedding_lfw
import cv2
import os
from torchvision import transforms
from sklearn.cluster import KMeans
import numpy as np
import sys
sys.path.insert(1,'/data/facebiometrics/updated/ML_work/our_method/Implementation')
from utils import get_embeddings_from_list

def world_neg_cluster(num=1000):
    initial_embeddings = []
    tensor_list = []
    dir = '/data/facebiometrics/updated/ML_work/Datasets/lfw-deepfunneled'
    for name in os.listdir(dir):
        if 'inverted' not in name: #excluding inverted images from clustering process (if embedding_lfw dataset is chosen)
            for sample in os.listdir(os.path.join(dir,name)):
                img_path = os.path.join(dir,name,sample)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((160, 160)),
                ])
                img_tensor = transform(img).unsqueeze(0)  
                tensor_list.append(img_tensor)
                #now pass it to the model to generate embeddings
                # embedding = img_tensor.cpu().numpy()
                # initial_embeddings.append( get_embedding(img_tensor).cpu().numpy() )
               
    
    # initial_embeddings = np.array(get_embeddings_from_list(tensor_list).cpu().numpy(), dtype='float32')
    
    embeddings = get_embeddings_from_list(tensor_list)  
    for embedding in embeddings:
        initial_embeddings.append(embedding.cpu().numpy()) #every tensor is now made into a numpy array
    
    initial_embeddings = np.array(initial_embeddings, dtype='float32')
    #now the 'initial_embeddings' has flattened embeddings
    
    kmeans = KMeans(n_clusters=num, random_state=42)
    kmeans.fit(initial_embeddings)
    
    centroids = kmeans.cluster_centers_

    # world_neg = centroids.reshape(num, 512, 8, 8)
    print(centroids.shape)
    
    return centroids 
    
if __name__ == "__main__":
    world_neg_array = world_neg_cluster()
    np.save('/data/facebiometrics/updated/ML_work/preprocessed/world_neg_array.npy', world_neg_array)