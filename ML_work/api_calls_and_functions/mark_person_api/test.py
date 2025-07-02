from dotenv import load_dotenv
from pathlib import Path
import os
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1].parent / '.env')
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / '.env')
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import io
import torch.nn.functional as F
import traceback
import torch
from PIL import Image
import boto3
from api_calls_and_functions.shared_material.util_functions import group2head, get_single_image_embeddings, get_only_student_embedding, get_all_student_models


access_key = os.getenv('AWS_ACCESS_KEY_ID')
secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
s3_bucket = os.getenv('S3_BUCKET_NAME')

app = FastAPI()


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        contents = await file.read()
        segmented_heads, boxes = group2head(contents)
        #now, check how many people are registered (as detection algorithm will vary if only 1 person is registered)
        response = s3_client.list_objects_v2(
            Bucket=s3_bucket,
            Prefix='registration/',
            Delimiter='/'
        )
        common_prefixes = response.get('CommonPrefixes', [])
        students = [
            prefix['Prefix'].split('/')[1]
            for prefix in common_prefixes
            if 'embedding_dictionary' not in prefix['Prefix']
        ]

        if len(students) == 1:
            #we have only the embedding, so we have to perform cosine similarity
            registered_emb, embedding_key = get_only_student_embedding(s3_client, s3_bucket) #embedding_key is to extract the student name
            img_np = cv2.cvtColor(np.array(Image.open(io.BytesIO(contents))), cv2.COLOR_RGB2BGR)

            #iterate through the head_segmentations and pass each into through inceptionresnetv1
            for idx, person in enumerate(segmented_heads): #list of PIL images
                emb = get_single_image_embeddings(person) #gives [1,512]
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)
                if registered_emb.dim() == 1:
                    registered_emb = registered_emb.unsqueeze(0)

                # Now emb.shape and registered_emb.shape are both [1, 512]
                # Compare along the 512-dim vector axis (dim=1)
                sim = F.cosine_similarity(emb, registered_emb, dim=1).item()
                if sim > 0.66:
                    student_name = embedding_key.split('/')[1]
                    print(f"Detected {student_name} with similarity {sim:.2f}")
                    #add to future frame the current detected bounding box
                    x1, y1, x2, y2 = map(int, boxes[idx])
                    cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_np, f"{student_name} ({sim:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        
        else:
            img_np = cv2.cvtColor(np.array(Image.open(io.BytesIO(contents))), cv2.COLOR_RGB2BGR)
            models = get_all_student_models(s3_client, s3_bucket)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            #now, we have to run all the stored models against all the people in the group photo
            scores_dict = {}
            for idx, person in enumerate(segmented_heads):  # PIL images
                emb = get_single_image_embeddings(person)  # shape: [1, 512]
                for student_name, model in models:
                    with torch.no_grad():
                        _, probs = model(emb.to(device)) 
                        prob = probs[:,0]
                        scores_dict[student_name] = prob.item()
                    max = 0
                    person = ''
                    for k, v in scores_dict.items():
                        if v > max:
                            max = v
                            person = k
            
                    if max > 0.8:     
                        print(f"Detected {person} with confidence {max:.2f} for head {idx}")
                        x1, y1, x2, y2 = map(int, boxes[idx])
                        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img_np, f"{person} ({max:.2f})", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
        # img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        _, img_encoded = cv2.imencode(".jpg", img_np)
        return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")

    except Exception as e:
        print("Error during image processing:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn    
    uvicorn.run(app, host="0.0.0.0", port=8000)     
