### Implementation

1. Before you run `/main.py`, ensure you have all the required `.npy` and `.pkl` files listed below. If you do not, follow the accompanying steps to generate them:
    - **world_neg_array.npy**:

        - Go to `/aux_functions/world_neg.py`.
        - Run the script.
        - Alternatively, you can change the directory to any dataset of your choice, which follows a structure similar to the datasets given in the Google Drive (e.g., `lfw-deepfunneled`).
        - We have chosen `lfw-deepfunneled` for this implementation.

    - **lfw_train_dict.pkl**:

        - Go to `/aux_functions/train_pkl.py`.
        - Run the script, which will store the lfw_embeddings in a dictionary format in a .pkl file for easy read access.
    
    - **lfw_test_dict.pkl**:

        - Go to `/aux_functions/test_pkl.py`.
        - Run the script, which will store the lfw_embeddings in a dictionary format in a .pkl file for easy read access.

2. Run `main.py` to commence training of one-person-models on out modified `lfw-train` dataset.

3. Then, run `/benchmarking/metrics_lfw.py` to generate an excel file of metrics of the lfw-subjects.
