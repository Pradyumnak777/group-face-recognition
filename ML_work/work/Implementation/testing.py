from utils import get_embeddings, LFWDataset_embedding_test, DataLoader
from models import bin_classifier
import torch
import os
from sklearn.metrics import precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model2 = bin_classifier()
model2.load_state_dict(torch.load('benchmarking/people/Julianne_Moore/Julianne_Moore_model/model_Julianne_Moore.pth', map_location=device))
model2.to(device)
model2.eval()

metrics_dict = {}
dict = {}

# for person in os.listdir('modified_lfw_datasets/face_dataset_lfw_test'):
#     lfw_test_dict = get_embeddings(f'modified_lfw_datasets/face_dataset_lfw_test/{person}')

#     # Create the dataset and DataLoader
#     test_dataset = LFWDataset_embedding_test(lfw_test_dict)
#     testloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

#     total = 0
#     correct = 0
#     all_labels = []
#     all_predictions = []

#     with torch.no_grad():
#         for embeddings in testloader:
#             _, probs = model2(embeddings)
#             predictions = (probs[:, 0] > 0.8).int()
            
#             all_labels.extend([1] * embeddings.size(0))  # Assuming 1 for positive label; adjust if needed
#             all_predictions.extend(predictions.tolist())
            
#             total += embeddings.size(0)
#             correct += predictions.sum().item()

#     accuracy = (correct / total) * 100
#     precision = precision_score(all_labels, all_predictions)
#     recall = recall_score(all_labels, all_predictions)
#     f1 = f1_score(all_labels, all_predictions)

#     print(probs)

#     metrics_dict[person] = {
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1_score': f1
#     }
    
#     print(f"{person} - Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

# print(metrics_dict)
fp = fn = tp = tn = 0

for person in os.listdir('modified_lfw_datasets/face_dataset_lfw_test'):
    lfw_test_dict = get_embeddings(f'modified_lfw_datasets/face_dataset_lfw_test/{person}')

    test_dataset = LFWDataset_embedding_test(lfw_test_dict)
    testloader = DataLoader(test_dataset,batch_size=32,shuffle=True)

    
    total = 0
    correct = 0
    with torch.no_grad():
        for embeddings in testloader:
            _,probs = model2(embeddings)
            if person == 'Julianne_Moore':
                print(f"tn: {tn}, tp: {tp}, fp: {fp}, fn:{fn}")
                tp += (probs[:, 0] >= 0.8).sum().item()
                fn += (probs[:, 0] < 0.8).sum().item()
            else:
                print(f"tp: {tn}, tp: {tp}, fp: {fp}, fn:{fn}")
                fp += (probs[:, 0] >= 0.8).sum().item()
                tn += (probs[:, 0] < 0.8).sum().item()

    
    # print(probs)
    # accuracy = (correct/total) * 100
    # print(person, accuracy)
    # dict[person] = accuracy
    
    
accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0  
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
    
# print(dict)