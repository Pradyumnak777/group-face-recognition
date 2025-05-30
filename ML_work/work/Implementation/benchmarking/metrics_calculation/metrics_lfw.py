import os
import pickle
import sys
import torch
import pandas as pd

sys.path.insert(1,'/data/facebiometrics/updated/ML_work/our_method/Implementation')
from utils import LFWDataset_embedding_test
from torch.utils.data import DataLoader
from models import bin_classifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = []
with open('/data/facebiometrics/updated/ML_work/our_method/Implementation/preprocessed/lfw_test_dict.pkl', 'rb') as file:
    lfw_test_dict = pickle.load(file)

test_dataset = LFWDataset_embedding_test(lfw_test_dict)
testloader = DataLoader(test_dataset,batch_size=32,shuffle=True)

#now start a loop across all one-person-models
opm_dir = '/data/facebiometrics/updated/ML_work/our_method/Implementation/one-person-model/models_lfw'

model = bin_classifier().to(device)

for person in os.listdir(opm_dir):
    true_label = person
    model.load_state_dict(torch.load(f'{opm_dir}/{person}/model_{person}.pth'))
    model.eval()
    fp = 0 #false positives
    fn = 0 #false negatives
    tp = 0 #true positives
    tn = 0 #true negatives
    #first, pass the data to the model
    with torch.no_grad():
        for embeddings,labels in testloader: #running through embedding train-lfw
            embeddings = embeddings.to(device)
            # labels = labels.to(device)
            _,probs = model(embeddings)
            for i,prob in enumerate(probs):
                if prob[0].item() > 0.8 and labels[i]==true_label:
                    tp+=1
                elif prob[0].item() > 0.8 and labels[i] != true_label:
                    fp+=1
                elif prob[0].item() < 0.8 and labels[i]==true_label:
                    fn += 1
                elif prob[0].item() < 0.8 and labels[i]!=true_label:
                    tn += 1
        
    #now calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2
    
    w_pos = tp + fn
    w_neg = tn + fp
    
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0
    weighted_accuracy = (w_pos * recall_pos + w_neg * recall_neg) / (w_pos + w_neg) if (w_pos + w_neg) > 0 else 0
    results.append({
        "Name": true_label,
        "Precision": precision,
        "Recall": recall,
        "Balanced Accuracy": balanced_accuracy,
        "F1 Score": f1_score,
        "weighted accuracy": weighted_accuracy,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn
    })

df_results = pd.DataFrame(results)

# Save to Excel
output_file = "/data/facebiometrics/updated/ML_work/our_method/Implementation/benchmarking/metrics_calculation/metrics_results.xlsx"
df_results.to_excel(output_file, index=False)

print(f"Metrics saved")
    
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
    
