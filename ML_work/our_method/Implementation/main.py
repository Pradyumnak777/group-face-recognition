#IMPORT STATEMENTS

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from training import train_resnet, train_classifier
# from ML_work.our_method.Implementation.aux_functions.world_neg import world_neg_cluster
from utils import get_lfw_embeddings, get_embeddings, LFWDataset_embedding
import os
from models import bin_classifier
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#TRAIN RESNET_EMBEDDER----

#run if pretrained model not present.
# train_resnet() #model will be saved in '/benchmarking'


#----WOKRING ON ONE-PERSON-MODELS

#create world_neg_set

# world_neg_array = world_neg_cluster()  #DO NOT RUN IF WORLD_NEG IS ALREADY PRESENT(TAKES LONG TO RUN)
# np.save('/face_attendance/benchmarking/world_neg_array.npy', world_neg_array)
world_neg_array = np.load('/data/facebiometrics/updated/ML_work/our_method/Implementation/preprocessed/world_neg_array.npy')

#a dicionatry of lfw_embeddings and their corresponding labels

#can be laoded from .pkl file..
# lfw_train_dict = get_lfw_embeddings('/face_attendance/modified_lfw_datasets/face_dataset_lfw_train')
# with open('/face_attendance/benchmarking/lfw_train_dict.pkl', 'wb') as file:
#     pickle.dump(lfw_train_dict, file)

with open('/data/facebiometrics/updated/ML_work/our_method/Implementation/preprocessed/lfw_train_dict.pkl', 'rb') as file:
    lfw_train_dict = pickle.load(file)

#OPM training loop (all 158 people)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

count = 0

for person_name in os.listdir('/data/facebiometrics/updated/ML_work/Datasets/modified_lfw_datasets/face_dataset_lfw_train'):
    # if count == 1:  #do for just one person
    #     break
    # elif person_name == 'Amelie_Mauresmo' or person_name == 'Julianne_Moore' or person_name == 'Tom_Hanks': # we already have her model, check on others
    #     continue
    
    print(person_name)
    #define positive embeddings and negative embeddings
    #opm-train-new =>  has images taken from the lfw-train-set, replicated as neg(inverted) and pos, then converted to embeddings
    embeddings_positive = get_embeddings(f'/data/facebiometrics/updated/ML_work/our_method/Implementation/one-person-model/opm-train-new/{person_name}/{person_name}_p')
    inverted_embeddings = get_embeddings(f'/data/facebiometrics/updated/ML_work/our_method/Implementation/one-person-model/opm-train-new/{person_name}/{person_name}_n')
    embeddings_positive = [tensor.cpu().numpy() for tensor in embeddings_positive]
    inverted_embeddings = [tensor.cpu().numpy() for tensor in inverted_embeddings]
    embeddings_positive = np.array(embeddings_positive, dtype=float)
    inverted_embeddings = np.array(inverted_embeddings, dtype=float)
    
    #initial train
    
    model2 = bin_classifier().to(device)
    optimizer = torch.optim.Adam(model2.parameters(), lr=0.0001)
    
    train_classifier(embeddings_positive,inverted_embeddings,person_name,model2,optimizer)
    
    #preparing for subsequent trains
    
    world_neg_person = np.concatenate((world_neg_array,inverted_embeddings))
    world_neg_person = np.array(world_neg_person, dtype=float)
    
    
    train_dataset = LFWDataset_embedding(lfw_train_dict, person_name)
    trainloader = DataLoader(train_dataset,batch_size=32,shuffle=True)
    #start boosting
    threshold = 0.95
    metric = 0
    while metric < threshold :
        model2.load_state_dict(torch.load(f'/data/facebiometrics/updated/ML_work/our_method/Implementation/one-person-model/models_lfw/{person_name}/model_{person_name}.pth'))
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
        auc = roc_auc_score(new_true_labels, new_predicted_labels)
        print("AUC-ROC:", auc)
        
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
            
            
            train_classifier(embeddings_positive,world_neg_person,person_name,model2,optimizer)


    # os.makedirs(f'/face_attendance/benchmarking/people/{person_name}/{person_name}_world_neg', exist_ok=True)
    np.save(f'/data/facebiometrics/updated/ML_work/our_method/Implementation/one-person-model/models_lfw/{person_name}/world_neg_{person_name}.npy', world_neg_person)
    
    count+=1
    
    # #new model stats
    # model2.load_state_dict(torch.load(f'/data/facebiometrics/updated/ML_work/our_method/Implementation/one-person-model/models_lfw/{person_name}/model_{person_name}.pth'))
    # model2.to(device)
    # model2.eval()
    # fp = [] #false positives
    # fn = [] #false negatives
    # tp = [] #true positives
    # tn = [] #true negatives
    # #first, pass the data to the model
    # predicted_labels = []
    # true_labels = []
    # with torch.no_grad():
    #     for embeddings,labels in trainloader: #running through embedding train-lfw
    #         embeddings = embeddings.to(device)
    #         _,probs = model2(embeddings)
    #         predicted_labels.extend(probs[:,0])
    #         true_labels.extend(labels[:, 0])
    #         for prob, label, embedding in zip(probs[:, 0], labels[:, 0], embeddings):
    #             if prob > threshold and label == 0: #it is a false positive
    #                 fp.append(embedding)
    #             elif prob < threshold and label == 1: #it is a false negative
    #                 fn.append(embedding)
    #             elif prob > threshold and label == 1:  #true positive
    #                 tp.append(embedding)
    #                 # correct += (probabilities >= var_threshold).sum().item()
    #             elif prob < threshold and label == 0:  #true negative
    #                 tn.append(embedding)
    
    # if (len(tp) + len(fp)) == 0:
    #     precision = 0
    # else:
    #     precision = len(tp) / (len(tp) + len(fp))
    
    # if (len(tp) + len(fn)) == 0:
    #     recall = 0
    # else:
    #     recall = len(tp) / (len(tp) + len(fn))
    
    # if (precision + recall) == 0:
    #     b_accuracy = 0
    # else:
    #     b_accuracy = (2 * precision * recall) / (precision + recall)
    # accuracy = b_accuracy * 100
    # print(f"Precision: {precision:.2f}")
    # print(f"Recall: {recall:.2f}")
    # print(f"Balanced Accuracy: {accuracy:.2f}")
    
    # #convert true_labels and predicted_labels into normal lists from tensors
    # new_true_labels = []
    # new_predicted_labels = []
    # for label in predicted_labels:
    #     new_predicted_labels.append(label.item())
    # for label in true_labels:
    #     new_true_labels.append(int(label.item()))
    # auc = roc_auc_score(new_true_labels, new_predicted_labels)
    # print("AUC-ROC:", auc)
    
    # print("fp: ", len(fp))
    # print("fn: ", len(fn))
        








