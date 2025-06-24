from ultralytics import YOLO
import os
import cv2
from PIL import Image
import io
import shutil
import torch
import traceback
import head_segmentation.segmentation_pipeline as seg_pipeline
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#reading the env variable-
local_path = os.getenv("LOCAL_STORAGE_PATH", os.getcwd())

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
            f'{local_path}/recieved_image.jpg',
            save_crop=True,
            exist_ok=True,
            conf=0.5,
            project=f'{local_path}/output',
            name='yolo_results'
        )

        print("Loading head segmentation model...")
        ckpt_path = os.path.join("preprocessed", "head_segmentation.ckpt")
        segmentation_pipeline = seg_pipeline.HumanHeadSegmentationPipeline(
            device=device,
            model_path=ckpt_path
        )

        segmented_heads = []
        face_crop_dir = f'{local_path}/output/yolo_results/crops/face'
        for image_path in os.listdir(face_crop_dir):
            cur_img = cv2.imread(os.path.join(face_crop_dir, image_path))
            segmentation_map = segmentation_pipeline.predict(cur_img)

            segmented_region = cur_img * cv2.cvtColor(segmentation_map, cv2.COLOR_GRAY2RGB)
            segmented_region_rgb = cv2.cvtColor(segmented_region, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(segmented_region_rgb)
            segmented_heads.append(pil_image)

        return segmented_heads

    except Exception as e:
        print("Exception in group2head():", e)
        traceback.print_exc()
        raise e
    


    
    