from PIL import Image
from io import BytesIO
import os
import shutil
import cv2
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader
from benchmarking.utils import get_embeddings,LFWDataset_embedding
from benchmarking.models import bin_classifier
from train_classifier_api import train_classifier


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(img_bytes_list, name):
    train_dir = '/face_attendance/api-calls/training-api/demo_lfw_train'
    dir = '/face_attendance/api-calls/training-api/temp_person_images'
    #clear temp_images
    if os.path.exists('/face_attendance/api-calls/training-api/temp_person_images'):
        shutil.rmtree('/face_attendance/api-calls/training-api/temp_person_images')
    imgs = []
    for idx,image_bytes in enumerate(img_bytes_list):
        #Open the image from bytes
        image = Image.open(BytesIO(image_bytes))
        image.save(f'{dir}/p/{idx}_p.jpg')
        # image.save(f'{train_dir}/{name}/{name}_{idx}.jpg')
        #we need to save the negative version also
        for sample in os.listdir(os.path.join(dir,'p')):
            test_im = cv2.imread(sample)
            inverted = 255 - test_im
            cv2.imwrite(f'{dir}/n/{idx}_n.jpg',inverted)
    
    #initial setup is ready for training model
    embeddings_positive = get_embeddings(f'{dir}/p')
    inverted_embeddings = get_embeddings(f'{dir}/n')
    embeddings_positive = np.array(embeddings_positive, dtype=float)
    inverted_embeddings = np.array(inverted_embeddings, dtype=float)
    
    #initial train
    
    model2 = bin_classifier().to(device)
    optimizer = torch.optim.Adam(model2.parameters(), lr=0.0001)
    
    train_classifier(embeddings_positive,inverted_embeddings,model2,optimizer)
    
    world_neg_array = np.load('/face_attendance/benchamrking/world_neg_array.npy')
    
    world_neg_person = np.concatenate((world_neg_array,inverted_embeddings))
    world_neg_person = np.array(world_neg_person, dtype=float)
    
    with open('/face_attendance/benchamrking/lfw_train_dict.pkl', 'rb') as file:
        lfw_train_dict = pickle.load(file)
        
    #append new person embeddings to this dictionary
    new_person_embeddings = get_embeddings(f'{dir}/p')
    lfw_train_dict[len(lfw_train_dict)] = (name, new_person_embeddings)     #tuple (name, tensor([...]))
    
    #now update this is our new train_dict, save it via pkl module
    
    
    train_dataset = LFWDataset_embedding(lfw_train_dict, name)
    trainloader = DataLoader(train_dataset,batch_size=32,shuffle=True)
    #start boosting
    threshold = 0.9
    accuracy = 0
    
    while accuracy < 90 :
        model2.load_state_dict(torch.load(f'/face_attendance/benchamrking/people/{person_name}/{person_name}_model/model_{person_name}.pth'))
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
            for embeddings,labels in trainloader: #running through embedding train-lfw
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
            b_accuracy = 0
        else:
            b_accuracy = (2 * precision * recall) / (precision + recall)
        accuracy = b_accuracy * 100
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"Balanced Accuracy: {accuracy:.2f}")
        
        #convert true_labels and predicted_labels into normal lists from tensors
        new_true_labels = []
        new_predicted_labels = []
        for label in predicted_labels:
            new_predicted_labels.append(label.item())
        for label in true_labels:
            new_true_labels.append(int(label.item()))
        auc = roc_auc_score(new_true_labels, new_predicted_labels)
        print("AUC-ROC:", auc)
        
        print("fp: ", len(fp))
        print("fn: ", len(fn))
        
        if accuracy < threshold*100:
            if len(fp)!= 0:
                fp_arr = np.array(fp, dtype=float)
                print("in loop training")
                world_neg_person = np.concatenate((world_neg_person,fp_arr))    #add fp to world_neg_person
                
            if len(fn) != 0:
                fn_arr = np.array(fn, dtype=float)
                print("in loop training")
                embeddings_positive = np.concatenate((embeddings_positive,fn_arr))   #add fn to embeddings_positive
            
            
            train_classifier(embeddings_positive,world_neg_person,person_name,model2,optimizer)


    os.makedirs(f'/face_attendance/benchamrking/people/{person_name}/{person_name}_world_neg', exist_ok=True)
    np.save(f'/face_attendance/benchamrking/people/{person_name}/{person_name}_world_neg/world_neg_{person_name}.npy', world_neg_person)