from ultralytics import YOLO
import os
import cv2
from PIL import Image
import io
import shutil
import torch
import head_segmentation.segmentation_pipeline as seg_pipeline
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def group2head(image_data : bytes) -> bytes:  #input group photo
    #clear runs/detect/predict
    if os.path.exists('ML_work/our_method/Implementation/group2face/group2face/runs/detect'):
        shutil.rmtree('ML_work/our_method/Implementation/group2face/group2face/runs/detect')
    
    grp_image = Image.open(io.BytesIO(image_data))
    #save grp image on the server
    grp_image.save('ML_work/archive_junk/new_temp_img.jpg')

    model = YOLO('ML_work/our_method/Implementation/preprocessed/best.pt')
    grp_photo_path = 'ML_work/archive_junk/new_temp_img.jpg'
    os.chdir('ML_work/our_method/Implementation/group2face')
    face_detection_result = model.predict(grp_photo_path,save_crop = True, exist_ok=True,conf = 0.5)
    # now they are saved in runs/predict
    segmentation_pipeline = seg_pipeline.HumanHeadSegmentationPipeline(device=device)
    
    num = 0
    segmented_heads = []
    for image_path in os.listdir('ML_work/our_method/Implementation/group2face/runs/detect/predict/crops/face'):
        #apply head segmentaiton here now
        cur_img = cv2.imread(os.path.join('ML_work/our_method/Implementation/group2face/runs/detect/predict/crops/face',image_path))
        segmentation_map = segmentation_pipeline.predict(cur_img)

        segmented_region = cur_img * cv2.cvtColor(segmentation_map, cv2.COLOR_GRAY2RGB)
        segmented_region_rgb = cv2.cvtColor(segmented_region, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(segmented_region_rgb)
        
        segmented_heads.append(pil_image)
        # pil_image.save(f'/interns/iittcseitr24_10/face_attendance/group2face/segmented_heads/{num}.jpg')

        num = num + 1
    
    return segmented_heads
    


    
    