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
from api_calls_and_functions.shared_material.packages.facenet_pytorch import InceptionResnetV1
from api_calls_and_functions.shared_material.util_functions.models_n_training import bin_classifier, custom_head
from pathlib import Path
import time
#global loads-
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = YOLO("api_calls_and_functions/shared_material/packages/best.pt")
model.to(device)
model.fuse()   # tiny speed boost
model.eval()

segmentation_pipeline = seg_pipeline.HumanHeadSegmentationPipeline(
    device=device,
    model_path="api_calls_and_functions/shared_material/packages/head_segmentation.ckpt",
)

transform = transforms.Compose([  #image neeeds to be in PIL format for this to work
    transforms.Resize((160, 160)),
    transforms.ToTensor(),  # converts to [0, 1]
    transforms.Normalize([0.5]*3, [0.5]*3) #converts to [-1,1]
])

#reading the env variable-
default_local_path = Path(__file__).resolve().parent
local_path = default_local_path

embedding_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
head_model = custom_head().eval().to(device)

def group2head(image_data: bytes) -> bytes:
    try:
        if os.path.exists(f'{local_path}/yolo_results'):
            shutil.rmtree(f'{local_path}/yolo_results')

        grp_image = Image.open(io.BytesIO(image_data))
        grp_image.save(f'{local_path}/recieved_image.jpg')
        # start = time.time()
        # # print(device)
        # print("Loading YOLO model...")
        # model = YOLO(f'api_calls_and_functions/register_face_api/preprocessed/best.pt')
        # print("YOLO load time:", time.time() - start)
        
        # staart_2 = time.time()
        # model.to(device)
        # print("Using device:", device)
        # print("YOLO model device:", next(model.model.parameters()).device)
        # print("moving yolo to device, print statements..:", time.time() - staart_2)
        
        start_detect = time.time()
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

        print("face detection time:", time.time() - start_detect)
        
        # start_head = time.time()
        # print("Loading head segmentation model...")
        # # ckpt_path = os.path.join("api_calls_and_functions/register_face_api/preprocessed", "head_segmentation.ckpt")
        # # segmentation_pipeline = seg_pipeline.HumanHeadSegmentationPipeline(
        # #     device=device,
        # #     model_path=ckpt_path
        # # )
        # print("headseg loading time:", time.time() - start_head)
        
        # start_head_proc = time.time()
        segmented_heads = []
        for cur_img in crops:  # 'crops' is a list of face crops from img_np
            segmentation_map = segmentation_pipeline.predict(cur_img)

            segmented_region = cur_img * cv2.cvtColor(segmentation_map, cv2.COLOR_GRAY2RGB)
            segmented_region_rgb = cv2.cvtColor(segmented_region, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(segmented_region_rgb)
            segmented_heads.append(pil_image)

        # print("headseg processing time:", time.time() - start_head_proc)
        return segmented_heads, boxes  #boxes will help us maintain correspndence.

    except Exception as e:
        print("Exception in group2head():", e)
        traceback.print_exc()
        raise e

def group2head_reg(image_data: bytes) -> bytes:
    try:
        # if os.path.exists(f'{local_path}/output'):
        #     shutil.rmtree(f'{local_path}/output')

        grp_image = Image.open(io.BytesIO(image_data))
        grp_image.save(f'{local_path}/recieved_image.jpg')

        # print("Loading YOLO model...")
        # model = YOLO(f'{local_path}/preprocessed/best.pt')

        print("Detecting faces...")
        face_detection_result = model.predict(
            source=grp_image,
            conf=0.5
        )
        img_np = np.array(grp_image)
        boxes = face_detection_result[0].boxes.xyxy.cpu().numpy()
        crops = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            face_crop = img_np[y1:y2, x1:x2]
            crops.append(face_crop)  
        # print("Loading head segmentation model...")
        # ckpt_path = os.path.join(f"{local_path}/preprocessed", "head_segmentation.ckpt")
        # segmentation_pipeline = seg_pipeline.HumanHeadSegmentationPipeline(
        #     device=device,
        #     model_path=ckpt_path
        # )

        segmented_heads = []
        # face_crop_dir = f'{local_path}/output/yolo_results/crops/face'
        segmentation_map = segmentation_pipeline.predict(crops[0]) #assuming only single face during registration

        segmented_region = crops[0] * cv2.cvtColor(segmentation_map, cv2.COLOR_GRAY2RGB)
        segmented_region_rgb = cv2.cvtColor(segmented_region, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(segmented_region_rgb)
        segmented_heads.append(pil_image)

        return segmented_heads

    except Exception as e:
        print("Exception in group2head():", e)
        traceback.print_exc()
        raise e
    

def get_single_image_embeddings(pil_image):
    embedding_model.to(device)
    #convert image to tensor
    tensor_image = transform(pil_image)  #dims -> 3,160,160
    tensor_image = tensor_image.unsqueeze(0) #dims -> 1,3,160,160
    tensor_image = tensor_image.to(device)
    with torch.no_grad():
        embedding = embedding_model(tensor_image)
        
    return embedding.detach().cpu() #dims -> 1,512

def get_head_embs(emb, model_state_dict):
    head_model.to(device)
    head_model.load_state_dict(model_state_dict)
    head_model.eval()
    #convert image to tensor
    tensor_emb = emb.to(device)
    with torch.no_grad():
        embedding = head_model(tensor_emb)
        
    return embedding.detach().cpu() #dims -> 1,64

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

def get_all_student_embeddings(s3_client, s3_bucket):
    # Step 1: List all objects under 'registration/'
    response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix='registration/')
    objects = response.get("Contents", [])

    # Step 2: Collect all embedding file paths
    embedding_files = []
    for obj in objects:
        key = obj["Key"]
        if "_embedding/" in key and key.endswith(".npy"):
            embedding_files.append(key)

    if not embedding_files:
        raise Exception("No embedding files found in S3.")

    # Step 3: Load all embeddings as tensors
    embeddings = []
    for embedding_file in embedding_files:
        buffer = io.BytesIO()
        s3_client.download_fileobj(s3_bucket, embedding_file, buffer)
        buffer.seek(0)
        embedding_np = np.load(buffer)
        student_name = embedding_file.split('/')[1]
        embeddings.append((torch.tensor(embedding_np, dtype=torch.float32), student_name))

    return embeddings  # List of (tensor, student_name) tuples

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
            model = bin_classifier().to(device)  
            state_dict = torch.load(buffer, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            model.load_state_dict(state_dict)
            model.eval()
            
            models.append((student_name, model))
    
    return models
    
    


    
    