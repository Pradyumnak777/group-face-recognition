from dotenv import load_dotenv
from pathlib import Path
import os
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1].parent / '.env')
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / '.env')
# print("Access Key:", os.getenv('AWS_ACCESS_KEY_ID'))
# print("Secret Key:", os.getenv('AWS_SECRET_ACCESS_KEY'))
# print("bucket name:", os.getenv('S3_BUCKET_NAME'))

from api_calls_and_functions.shared_material.util_functions import group2head_reg
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from typing import List
from PIL import Image
from torch.utils.data import DataLoader
import torch
import traceback
import boto3
import numpy as np
import io
from api_calls_and_functions.shared_material.util_functions.models_n_training import (
    augment_positive_set,
    get_embeddings_from_image_list,
    custom_head,
    train_head,
    get_single_image_embeddings
)
#cloud is the only option now!
access_key = os.getenv('AWS_ACCESS_KEY_ID')
secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
s3_bucket = os.getenv('S3_BUCKET_NAME')

app = FastAPI()
@app.post("/register-student")
async def upload_imgs(files: List[UploadFile] = File(...), 
                      student_name: str = Form(...)):
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )

        # print("Current Working Directory:", os.getcwd())
        # print("Resolved Storage Path:", local_storage_path)


        for num, file in enumerate(files):
            contents = await file.read()
            output_data = group2head_reg(contents)
            for _, img in enumerate(output_data):
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                img_byte_arr.seek(0)

                # Now upload to S3 using fileobj
                s3_client.upload_fileobj(img_byte_arr, s3_bucket, f"registration/{student_name}/{student_name}_initial.jpg")

        #now, perform 'registering'
        img_embedding = get_single_image_embeddings(img)
        with io.BytesIO() as f:
            np.save(f, img_embedding)
            f.seek(0)
            s3_client.upload_fileobj(f, s3_bucket, f"registration/{student_name}/{student_name}_embedding/{student_name}.npy")
        #embedding has been uploaded
        response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix='registration/')
        
        if len(response.get('Contents', [])) == 2:
            #this is the first registration, extended head(triplet loss) wont be trained now. Only embeddings from facenet will be extracted

            augmented_pos_set = augment_positive_set(img) 
            pos_embeddings = get_embeddings_from_image_list(augmented_pos_set) #this will include the image itself, so a total of 16 positive embeddings
            pos_embeddings = [tensor.cpu().numpy() for tensor in pos_embeddings]
            pos_embeddings = np.stack(pos_embeddings)
            student_embeddings_dict = {}
            student_embeddings_dict[f'{student_name}'] = pos_embeddings
            buffer = io.BytesIO()
            np.savez(buffer, **student_embeddings_dict)
            buffer.seek(0)
            s3_client.upload_fileobj(buffer, s3_bucket, f'embedding_dictionary/student_embeddings_dict.npz') #this is a dictionary of pos student embeddings.

            
        else:
            #this is not the first registration
            #therefore, append embedding of this image to existing student_embedding dict
            if 'Contents' in response and len(response.get('Contents', [])) > 2:
                buffer = io.BytesIO()
                s3_client.download_fileobj(Bucket=s3_bucket, Key='embedding_dictionary/student_embeddings_dict.npz', Fileobj=buffer)
                buffer.seek(0)
                loaded = np.load(buffer)
                student_embeddings_dict = {name: loaded[name] for name in loaded.files}
            
            augmented_pos_set = augment_positive_set(img)
            pos_embeddings = get_embeddings_from_image_list(augmented_pos_set)
            pos_embeddings = [tensor.cpu().numpy() for tensor in pos_embeddings]
            pos_embeddings = np.stack(pos_embeddings) 
            student_embeddings_dict[student_name] = pos_embeddings 
            new_buffer = io.BytesIO()
            np.savez(new_buffer, **student_embeddings_dict)
            new_buffer.seek(0)
            s3_client.upload_fileobj(new_buffer, s3_bucket, 'embedding_dictionary/student_embeddings_dict.npz') #positive embeddings of this image is stored in the dictionary now
            #the above code section performs updation of the embedding dict on amazon s3
            
            #to get the negative embeddings, take embeddings from all OTHER students and concatenate
            neg_embeddings = np.empty((0,512), dtype=np.float32)
            for k,v in student_embeddings_dict.items():
                if k != student_name:
                    neg_embeddings = np.vstack((neg_embeddings, v))
            
            #now train
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model2 = custom_head().to(device)
            #if model.pth already exists, just pull from it-(?)
            model_key = "model/model.pth"
            model_exists = False
            response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=model_key)
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['Key'] == model_key:
                        model_exists = True
                        break

            if model_exists:
                # download and load the model state dict
                model_buffer = io.BytesIO()
                s3_client.download_fileobj(Bucket=s3_bucket, Key=model_key, Fileobj=model_buffer)
                model_buffer.seek(0)
                state_dict = torch.load(model_buffer, map_location=device)
                model2.load_state_dict(state_dict)
                print("Loaded existing model state from S3.")
                        
            optimizer = torch.optim.Adam(model2.parameters(), lr=0.001)
            model2 = train_head(pos_embeddings,neg_embeddings,model2,optimizer)
            
            #this model will now be used(and updated, upon new registrations) globally
            model_buffer = io.BytesIO()
            torch.save(model2.state_dict(), model_buffer)
            model_buffer.seek(0)

            s3_client.upload_fileobj(model_buffer, s3_bucket, f"model/model.pth")
         
                   

        return {"message": "Student registered"}

    except Exception as e:
        print("Error during image processing:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn    
    uvicorn.run(app, host="0.0.0.0", port=8000)