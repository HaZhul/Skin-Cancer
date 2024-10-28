import cv2
import numpy as np
from sklearn.cluster import KMeans

class Segmentation_Mask:
    def __init__(self) -> None:
        pass
    
    def make_segmentation(self, image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
        segmented_image = cv2.bitwise_and(image, image, mask=cleaned_mask)
        return segmented_image


    def make_knn_segmentation(self, image_path, k=3):
        image = cv2.imread(image_path)
        
        original_shape = image.shape
        image_resized = cv2.resize(image, (original_shape[1] // 2, original_shape[0] // 2))
        
        pixel_values = image_resized.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(pixel_values)
        labels = kmeans.labels_
        centers = np.uint8(kmeans.cluster_centers_)

        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(image_resized.shape)

        segmented_image = cv2.resize(segmented_image, (original_shape[1], original_shape[0]))

        mask = np.zeros((original_shape[0], original_shape[1]), dtype=np.uint8)
        for i in range(k):
            cluster_mask = (labels == i).reshape(image_resized.shape[:2])
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size > (0.05 * cluster_mask.size):
                cluster_mask_resized = cv2.resize(cluster_mask.astype(np.uint8), (original_shape[1], original_shape[0]))
                mask = cv2.bitwise_or(mask, cluster_mask_resized * 255)

        final_segmented_image = cv2.bitwise_and(image, image, mask=mask)
        
        return final_segmented_image