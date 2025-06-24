from fastapi import FastAPI, HTTPException, File, Form
import traceback
import boto3
import os
import io
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import torch
import tempfile
from api_calls_and_functions.model_training import (
    augment_positive_set,
    get_single_image_embeddings,
    get_embeddings_from_image_list,
    BoostingDataset,
    bin_classifier,
    train_classifier,
)
access_key = os.getenv('AWS_ACCESS_KEY_ID')
secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
s3_bucket = os.getenv('S3_BUCKET_NAME')

app = FastAPI()
@app.post('/retrain')

async def retrain_model(student_name: str = Form(...)):
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        student_name = student_name.lower()
        response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=f'registration/{student_name}')
        # response['Contents']
        buffer = io.BytesIO()
        s3_client.download_fileobj(Bucket=s3_bucket, Key=f'registration/{student_name}_initial.jpg', Fileobj=buffer)
        buffer.seek(0)
        img = Image.open(buffer) #this is our initial_image of this person, with which we will create the initial pos set and train
        
        if 'Contents' in response and len(response.get('Contents', [])) > 0:
            buffer = io.BytesIO()
            s3_client.download_fileobj(Bucket=s3_bucket, Key='embedding_dictionary/student_embeddings_dict.npz', Fileobj=buffer)
            buffer.seek(0)
            loaded = np.load(buffer)
            student_embeddings_dict = {name: loaded[name] for name in loaded.files} #this dictionary is for boosting and adding to the pos and neg set 
        
        augmented_pos_set = augment_positive_set(img) #will not be used if there's only 1 registered face. this i primarily for hard negative mining.
        #'augment_pos_set' is now an array of tranformed PIL images
        pos_embeddings = get_embeddings_from_image_list(augmented_pos_set)
        neg_embeddings = np.load('api_calls_and_functions/model_training/required/world_neg_array.npy')
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model2 = bin_classifier().to(device)
        optimizer = torch.optim.Adam(model2.parameters(), lr=0.0001)
        model2 = train_classifier(pos_embeddings,neg_embeddings,student_name,model2,optimizer)
        
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
        
        return {"message": "student model re-trained"}    
        #alright, now we have a model ready for a single person (when its NOT the first registration)
    except Exception as e:
        print("Error during re-training:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))