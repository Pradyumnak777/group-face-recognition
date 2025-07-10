from dotenv import load_dotenv
from pathlib import Path
import os
import base64
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1].parent / '.env')
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / '.env')
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import io
import torch.nn.functional as F
import traceback
import torch
import tempfile
from PIL import Image
import boto3
from api_calls_and_functions.shared_material.util_functions import group2head, get_single_image_embeddings, get_only_student_embedding, get_all_student_embeddings, get_head_embs


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
        if len(segmented_heads) == 0:
            return {"message": "no faces detected"}
        #now, check how many people are registered (as detection algorithm will vary if only 1 person is registered)
        response = s3_client.list_objects_v2(
            Bucket=s3_bucket,
            Prefix='model/model.pth'
        )
        model_exists = False
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'] == 'model/model.pth':
                    model_exists = True
                    break
        
        #preparing world_neg_array-
        # world_neg_array = np.load('api_calls_and_functions/shared_material/packages/world_neg_array.npy')
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # world_neg_tensor = torch.tensor(world_neg_array, dtype=torch.float32).to(device)

        if model_exists == False: #we CANNOT use that model for prediction, use cosine similarity from inception model
            
            img_np = cv2.cvtColor(np.array(Image.open(io.BytesIO(contents))), cv2.COLOR_RGB2BGR)
            registered_emb, embedding_key = get_only_student_embedding(s3_client, s3_bucket)
            #iterate through the head_segmentations and pass each into through inceptionresnetv1
            for idx, person in enumerate(segmented_heads): #list of PIL images
                emb = get_single_image_embeddings(person) #gives [1,512]
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)
                if registered_emb.dim() == 1:
                    registered_emb = registered_emb.unsqueeze(0)

                # Now emb.shape and registered_emb.shape are both [1, 512]
                # Compare along the 512-dim vector axis (dim=1)
                detected_people = []
                sim = F.cosine_similarity(emb, registered_emb, dim=1).item()
                # neg_sims = F.cosine_similarity(emb, world_neg_tensor, dim=1)
                # avg_world_sim = torch.mean(neg_sims).item()
                # best_world_sim = torch.max(neg_sims).item()

                # print(f'sim: {sim:.2f}, avg_neg_sim: {avg_world_sim:.2f}, highest neg_sim: {best_world_sim:.2f}')
                print(f'sim: {sim:.2f}')

                # if sim > 0.65 and sim > best_world_sim and sim > avg_world_sim + 0.2:
                if sim > 0.65:
                    student_name = embedding_key.split('/')[1]
                    detected_people.append(student_name)
                    print(f"Detected {student_name} with similarity {sim:.2f}")
                    #add to future frame the current detected bounding box
                    x1, y1, x2, y2 = map(int, boxes[idx])
                    cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_np, f"{student_name} ({sim:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            #load the model's state dict
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                s3_client.download_fileobj(s3_bucket, 'model/model.pth', tmp_file)
                tmp_file_path = tmp_file.name

            model_state_dict = torch.load(tmp_file_path, map_location=device)    
            os.remove(tmp_file_path)
            
            # #now parse world_neg thru this head
            # with torch.no_grad():
            #     world_neg_proj = get_head_embs(world_neg_tensor, model_state_dict)
                
            img_np = cv2.cvtColor(np.array(Image.open(io.BytesIO(contents))), cv2.COLOR_RGB2BGR)
            #get all student embeddings-
            embeddings = get_all_student_embeddings(s3_client, s3_bucket)
            detected_people = []
            raw_threshold = 0.5 #our 512d emb
            head_threshold = 0.65 # 64d extension
            for idx, person in enumerate(segmented_heads):  # PIL images
                emb = get_single_image_embeddings(person)  # shape: [1, 512]
                #now pass this through the extended head
                #!!!
                #now, we first need to eliminate 'unregistered' faces. facenet is best at doing this
                scores_raw = {}
                best_raw_sim, best_raw_name = -1, None
                for e,name in embeddings:
                    sim = F.cosine_similarity(emb, e, dim=1).item() #both are 512-d embs , for now
                    scores_raw[name] = sim
                    if sim > best_raw_sim:
                        best_raw_sim = sim
                        best_raw_name = name
                
                # print(f"best match by face_net: {best_raw_name} with score {best_raw_sim}")
                if best_raw_sim < raw_threshold:
                    print(f"ditching this face, due to low score by facenet: {best_raw_sim}")
                    continue #skipping as facenet is better trained to detect unkowns
                
                #!!! NOW, passthrough head
                
                new_emb = get_head_embs(emb, model_state_dict)
                #now, perform cosine similarity against all other student embeddings
                scores_head = {}
                best_head_sim, best_head_name = -1, None
                # best_sim2 = -1
                for e,name in embeddings:
                    with torch.no_grad():
                        stored_e_head = get_head_embs(e, model_state_dict)
                    sim = F.cosine_similarity(new_emb, stored_e_head, dim=1).item() #both are 64-d projected embs now
                    scores_head[name] = sim
                    if sim > best_head_sim:
                        # best_sim2 = best_sim
                        best_head_sim = sim
                        best_head_name = name
                
                print(scores_head)
                if best_head_sim > head_threshold:
                    detected_people.append(best_head_name) 
                    print(f"Detected {best_head_name} with similarity {best_head_sim:.2f}")
                    #add to future frame the current detected bounding box
                    x1, y1, x2, y2 = map(int, boxes[idx])
                    cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_np, f"{best_head_name} ({best_head_sim:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    
                # -----------------
                       
                # neg_sims = F.cosine_similarity(new_emb, world_neg_proj)
                # avg_world_sim = torch.mean(neg_sims).item()#as world has 1000 embeddings frmo lfw wild
                # best_world_sim = torch.max(neg_sims).item()
                # print(f"best student sim: {best_sim:.2f}, avg world sim: {avg_world_sim:.2f}, highest neg_sim: {best_world_sim:.2f}")

                # print(scores)
                # # diff = best_sim - best_sim2
                # if best_sim > 0.6  and best_sim > best_world_sim and best_sim > avg_world_sim + 0.2: #we want to maximize distance between actual sim and similarity to the 'world'
                #     detected_people.append(best_name)
                #     print(f"Detected {best_name} with similarity {best_sim:.2f}")
                #     #add to future frame the current detected bounding box
                #     x1, y1, x2, y2 = map(int, boxes[idx])
                #     cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #     cv2.putText(img_np, f"{best_name} ({best_sim:.2f})", (x1, y1 - 10),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    #create excel sheet with attendance and return (WIP)
                        
                        
        # img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        _, img_encoded = cv2.imencode(".jpg", img_np)
        img_bytes = img_encoded.tobytes()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        return {"names": detected_people, "image": img_base64}

    except Exception as e:
        print("Error during image processing:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn    
    uvicorn.run(app, host="0.0.0.0", port=8000)     
