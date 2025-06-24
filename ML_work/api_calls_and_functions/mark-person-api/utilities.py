from ultralytics import YOLO
import os
import cv2
from PIL import Image
import io
import shutil
import torch
import traceback
import numpy as np
import head_segmentation.segmentation_pipeline as seg_pipeline
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([  #image neeeds to be in PIL format for this to work
    transforms.Resize((160, 160)),
    transforms.ToTensor(),  # converts to [0, 1]
    transforms.Normalize([0.5]*3, [0.5]*3) #converts to [-1,1]
])

#reading the env variable-
local_path = os.getcwd()

def group2head(image_data: bytes) -> bytes:
    try:
        if os.path.exists(f'{local_path}/yolo_results'):
            shutil.rmtree(f'{local_path}/yolo_results')

        grp_image = Image.open(io.BytesIO(image_data))
        # grp_image.save(f'{local_path}/recieved_image.jpg')

        print("Loading YOLO model...")
        model = YOLO(f'preprocessed/best.pt')

        print("Detecting faces...")
        face_detection_result = model.predict(
            source=grp_image,
            conf=0.5,
        )
        img_np = np.array(grp_image)
        boxes = face_detection_result[0].boxes.xyxy.cpu().numpy()
        crops = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            face_crop = img_np[y1:y2, x1:x2]
            crops.append(face_crop)  

        print("Loading head segmentation model...")
        ckpt_path = os.path.join("preprocessed", "head_segmentation.ckpt")
        segmentation_pipeline = seg_pipeline.HumanHeadSegmentationPipeline(
            device=device,
            model_path=ckpt_path
        )

        segmented_heads = []
        for cur_img in crops:  # 'crops' is a list of face crops from img_np
            segmentation_map = segmentation_pipeline.predict(cur_img)

            segmented_region = cur_img * cv2.cvtColor(segmentation_map, cv2.COLOR_GRAY2RGB)
            segmented_region_rgb = cv2.cvtColor(segmented_region, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(segmented_region_rgb)
            segmented_heads.append(pil_image)

        return segmented_heads, boxes  #boxes will help us maintain correspndence.

    except Exception as e:
        print("Exception in group2head():", e)
        traceback.print_exc()
        raise e

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

def get_registered_embedding_from_s3(student_name, s3_client, s3_bucket):
    s3_key = f'registration/{student_name}/{student_name}_embedding/{student_name}.npy'
    buffer = io.BytesIO()
    s3_client.download_fileobj(s3_bucket, s3_key, buffer)
    buffer.seek(0)
    embedding_np = np.load(buffer)
    return torch.tensor(embedding_np, dtype=torch.float32)

def get_only_student_embedding(s3_client, s3_bucket):
    # Step 1: List all objects under 'registration/'
    response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix='registration/')
    objects = response.get("Contents", [])

    # Step 2: Find the one embedding file path
    embedding_file = None
    for obj in objects:
        key = obj["Key"]
        if "_embedding/" in key and key.endswith(".npy"):
            embedding_file = key
            break

    if embedding_file is None:
        raise Exception("No embedding file found in S3.")

    # Step 3: Load embedding as tensor
    buffer = io.BytesIO()
    s3_client.download_fileobj(s3_bucket, embedding_file, buffer)
    buffer.seek(0)
    embedding_np = np.load(buffer)
    return torch.tensor(embedding_np, dtype=torch.float32), embedding_file

def get_all_student_models(s3_client, s3_bucket):
    response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix='registration/')
    objects = response.get("Contents", [])
    
    models = []  # List of tuples: (student_name, model)
    
    for obj in objects:
        key = obj["Key"]
        if "_model/" in key and key.endswith(".pth"):
            student_name = key.split('/')[1]  # assuming format registration/{student_name}/{...}
            
            # Download model file into buffer
            buffer = io.BytesIO()
            s3_client.download_fileobj(s3_bucket, key, buffer)
            buffer.seek(0)
            
            # Load model
            model = InceptionResnetV1(pretrained='vggface2').eval()  # Adjust args if needed
            state_dict = torch.load(buffer, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            model.load_state_dict(state_dict)
            model.eval()
            
            models.append((student_name, model))
    
    return models
    
    


    
    