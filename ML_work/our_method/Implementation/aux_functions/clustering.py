from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

def k_means(face_vector, num=1000):
    # images = []
    # for path in fp:
    #     img = cv2.imread(path)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    #     images.append(img)
        
    # resized_images = [cv2.resize(img, (250, 250)) for img in images]
    # features = np.array([img.flatten() for img in resized_images])
    scaler = StandardScaler()
    # features_normalized = scaler.fit_transform(face_vector)
    kmeans = KMeans(n_clusters=num, random_state=42)
    
    #resize is to row-column type
    # np.reshape(face_vector)
    
    kmeans.fit(face_vector)
    
    cluster_centers_normalized = kmeans.cluster_centers_
    
    cluster_centers_images = []
    for center_normalized in cluster_centers_normalized:
        center = scaler.inverse_transform(center_normalized.reshape(1, -1))
        center = center.reshape(250, 250, 3)
        center = np.clip(center, 0, 255).astype(np.uint8)  
        cluster_centers_images.append(center)
    
    return cluster_centers_images