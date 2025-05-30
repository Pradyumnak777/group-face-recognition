import pickle
import os
import sys
sys.path.insert(1,'/data/facebiometrics/updated/ML_work/our_method/Implementation')
from utils import get_lfw_embeddings

def test_pkl():
    lfw_test_dict = get_lfw_embeddings('/data/facebiometrics/updated/ML_work/Datasets/modified_lfw_datasets/face_dataset_lfw_test')
    with open('/data/facebiometrics/updated/ML_work/our_method/Implementation/preprocessed/lfw_test_dict.pkl', 'wb') as file:
        pickle.dump(lfw_test_dict, file)

if __name__ == "__main__":
    test_pkl()