from g2f import group2head
import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from typing import List
from PIL import Image
from torch.utils.data import DataLoader
import torch
import traceback
import boto3
import tempfile
import numpy as np
import io
from api_calls_and_functions.model_training import (
    augment_positive_set,
    get_single_image_embeddings,
    get_embeddings_from_image_list,
    BoostingDataset,
    bin_classifier,
    train_classifier,
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
            output_data = group2head(contents)
            for _, img in enumerate(output_data):
                s3_client.upload_file(img, s3_bucket, f"registration/{student_name}/{student_name}_inital.jpg")

        #now, perform 'registering'

        response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix='registration/')
        img_embedding = get_single_image_embeddings(img)
        with io.BytesIO() as f:
            np.save(f, img_embedding)
            f.seek(0)
            s3_client.upload_fileobj(f, s3_bucket, f"registration/{student_name}/{student_name}_embedding/{student_name}.npy")
        #embedding has been uploaded
        
        if 'Contents' not in response or len(response.get('Contents', [])) == 0:
            #this is the first registration, model wont be trained now. Only embeddings from facenet will be extracted
            #create an empty student_embeddings_dict and save locally in api_calls_and_functions/model_training/required
            student_embeddings_dict = {}
            student_embeddings_dict[f'{student_name}'] = img_embedding
            buffer = io.BytesIO()
            np.savez(buffer, **student_embeddings_dict)
            buffer.seek(0)
            s3_client.upload_fileobj(buffer, s3_bucket, f'embedding_dictionary/student_embeddings_dict.npz')

            
        else:
            #this is not the first registration
            #therefore, append embedding of this image to existing student_embedding dict
            if 'Contents' in response and len(response.get('Contents', [])) > 0:
                buffer = io.BytesIO()
                s3_client.download_fileobj(Bucket=s3_bucket, Key='embedding_dictionary/student_embeddings_dict.npz', Fileobj=buffer)
                buffer.seek(0)
                loaded = np.load(buffer)
                student_embeddings_dict = {name: loaded[name] for name in loaded.files}
            
            student_embeddings_dict[student_name] = img_embedding
            new_buffer = io.BytesIO()
            np.savez(new_buffer, **student_embeddings_dict)
            new_buffer.seek(0)
            s3_client.upload_fileobj(new_buffer, s3_bucket, 'embedding_dictionary/student_embeddings_dict.npz')
            
            #the above code section performs updation of the embedding dict on amazon s3
            
            augmented_pos_set = augment_positive_set(img) #will not be used if there's only 1 registered face. this i primarily for hard negative mining.
            #'augment_pos_set' is now an array of tranformed PIL images
            pos_embeddings = get_embeddings_from_image_list(augmented_pos_set)
            neg_embeddings = np.load('api_calls_and_functions/model_training/required/world_neg_array.npy')
            
            #perform initial training before boosting-
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model2 = bin_classifier().to(device)
            optimizer = torch.optim.Adam(model2.parameters(), lr=0.0001)
            model2 = train_classifier(pos_embeddings,neg_embeddings,student_name,model2,optimizer)
            
            #now, create a dataloader/dataset from the dictionary in s3 (to perform bootstrapping/boosting)
            with tempfile.NamedTemporaryFile() as tmp:
                s3_client.download_fileobj(s3_bucket, 'embedding_dictionary/student_embeddings_dict.npz', tmp)
                tmp.seek(0)
                npz = np.load(tmp)
                embedding_dict = dict(npz)
                train_dataset = BoostingDataset(student_name, embedding_dict)
                trainloader = DataLoader(train_dataset,batch_size=16,shuffle=True) #returns- (embedding, ground_truth-label)
                
                #now, boost-train
                threshold = 0.95
                ###
                metric = 0
                while metric < threshold :
                    model2.to(device)
                    model2.eval()
                    fp = [] #false positives
                    fn = [] #false negatives
                    tp = [] #true positives
                    tn = [] #true negatives
                    #first, pass the data to the model
                    predicted_labels = []
                    true_labels = []
                    with torch.no_grad():
                        for embeddings,labels in trainloader: #running through the student_embedding dictionary
                            embeddings = embeddings.to(device)
                            _,probs = model2(embeddings)
                            predicted_labels.extend(probs[:,0])
                            true_labels.extend(labels[:, 0])
                            for prob, label, embedding in zip(probs[:, 0], labels[:, 0], embeddings):
                                if prob > threshold and label == 0: #it is a false positive
                                    fp.append(embedding)
                                elif prob < threshold and label == 1: #it is a false negative
                                    fn.append(embedding)
                                elif prob > threshold and label == 1:  #true positive
                                    tp.append(embedding)
                                    # correct += (probabilities >= var_threshold).sum().item()
                                elif prob < threshold and label == 0:  #true negative
                                    tn.append(embedding)
                    
                    if (len(tp) + len(fp)) == 0:
                        precision = 0
                    else:
                        precision = len(tp) / (len(tp) + len(fp))
                    
                    if (len(tp) + len(fn)) == 0:
                        recall = 0
                    else:
                        recall = len(tp) / (len(tp) + len(fn))
                    
                    if (precision + recall) == 0:
                        f1 = 0
                    else:
                        f1 = (2 * precision * recall) / (precision + recall)
                        f1 = f1 * 100
                        
                    if (len(tn) + len(fp)) == 0:
                        specificity = 0
                    else:
                        specificity = len(tn) / (len(tn) + len(fp))
                    
                    balanced_accuracy = (recall + specificity) / 2
                    
                    print(f"Precision: {precision:.2f}")
                    print(f"Recall: {recall:.2f}")
                    print(f"F1 Score: {f1:.2f}")
                    print(f"Balanced Accuracy: {balanced_accuracy:.2f}")
                    
                    metric = balanced_accuracy #this can be tweaked/modified.
                    
                    #convert true_labels and predicted_labels into normal lists from tensors
                    new_true_labels = []
                    new_predicted_labels = []
                    for label in predicted_labels:
                        new_predicted_labels.append(label.item())
                    for label in true_labels:
                        new_true_labels.append(int(label.item()))
                    # auc = roc_auc_score(new_true_labels, new_predicted_labels)
                    # print("AUC-ROC:", auc)
                    
                    print("fp: ", len(fp))
                    print("fn: ", len(fn))
                    
                    if metric < threshold:
                        if len(fp)!= 0:
                            fp_arr = np.array([tensor.cpu().numpy() for tensor in fp], dtype=float)
                            print("in loop training")
                            world_neg_person = np.concatenate((world_neg_person,fp_arr))    #add fp to world_neg_person
                            
                        if len(fn) != 0:
                            fn_arr = np.array([tensor.cpu().numpy() for tensor in fn], dtype=float)
                            print("in loop training")
                            embeddings_positive = np.concatenate((embeddings_positive,fn_arr))   #add fn to embeddings_positive
                        
                        
                        model2 = train_classifier(embeddings_positive,world_neg_person,model2,optimizer)


                # os.makedirs(f'/face_attendance/benchmarking/people/{person_name}/{person_name}_world_neg', exist_ok=True)
                # np.save(f'/data/facebiometrics/updated/ML_work/our_method/Implementation/one-person-model/models_lfw/{person_name}/world_neg_{person_name}.npy', world_neg_person)
                
                #push this model to s3 now
                buffer = io.BytesIO()
                torch.save(model2.state_dict(), buffer)
                buffer.seek(0)
                s3_client.upload_fileobj(
                    buffer, 
                    s3_bucket, 
                    f"registration/{student_name}/{student_name}_model/{student_name}.pth"
                )
                ###
                
            #alright, now we have a model ready for a single person (when its NOT the first registration)

        return {"message": "Student registered"}

    except Exception as e:
        print("Error during image processing:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn    
    uvicorn.run(app, host="0.0.0.0", port=8000)