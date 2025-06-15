from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import boto3
import os
import numpy as np
import cv2
from utilities import get_embeddings_from_image_list
storage_env = os.getenv("LOCAL_STORAGE_PATH") 
local_storage_path = storage_env if storage_env else os.getcwd()

#take in images from amazon s3-
# s3_client = boto3.client(
#     's3',
#     aws_access_key_id='',
#     aws_secret_access_key=''
# )

# response = s3_client.get_object(bucket, object_key)
# image_data = response['Body'].read()
# image_array = np.frombuffer(image_data,np.uint8)
# image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)  
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#now, this person will have a single model 'for himself'
os.makedirs(f'{local_storage_path}/opm-train_data',exist_ok = True)

image = cv2.imread(f'temp_image.jpg') 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imwrite(f'{local_storage_path}/opm-train_data/original_img.jpg', image)

#-> create positive set. Image taken + various transformatioons
os.makedirs(f'{local_storage_path}/opm-train_data/pos_set',exist_ok = True)

augmented_faces = [] #will be a list of numpy-aray images
for i in range(20):
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
    
    augmented_faces.append(face)

#ok here, we need to check if this is the first entry or not 

    
#now get embeddings for these positive images
pos_embeddings = get_embeddings_from_image_list(augmented_faces)
neg_embeddings = np.load('work/Functionality/api-calls/model-training-api/required/world_neg_array.npy')







 





world_neg_array = np.load('work/Implementation/preprocessed/world_neg_array.npy')
