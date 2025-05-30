import pickle
import os
import sys
sys.path.insert(1,'/data/facebiometrics/updated/ML_work/our_method/Implementation')
from utils import get_lfw_embeddings

def train_pkl():
    lfw_train_dict = get_lfw_embeddings('/data/facebiometrics/updated/ML_work/Datasets/modified_lfw_datasets/face_dataset_lfw_train')
    with open('/data/facebiometrics/updated/ML_work/preprocessed/lfw_train_dict.pkl', 'wb') as file:
        pickle.dump(lfw_train_dict, file)

if __name__ == "__main__":
    train_pkl()