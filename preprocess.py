import cv2
import numpy as np

def run_preprocessing(image):
    grey_image_with_higher_contrast = stretching_the_image_colour_palette(image)
    reduced_gray_image = reduce_image_size(grey_image_with_higher_contrast, 2)
    image_with_padding = add_black_padding(reduced_gray_image, 5)
    mask = segment_area_expansion(image_with_padding, lo_diff=30, up_diff=30)
    mask_after_morphology_operations = run_morphology_operations(mask)
    negated_mask = 1 - mask_after_morphology_operations

    return negated_mask


def stretching_the_image_colour_palette(img):
    img_float = img.astype(np.float32)

    min_val = img_float.min()
    max_val = img_float.max()

    stretched_img = 255 * (img_float - min_val) / (max_val - min_val)
    
    return stretched_img.astype(np.uint8)


def add_black_padding(image, padding_size=10):
    padded_image = np.zeros((image.shape[0] + 2 * padding_size, 
                            image.shape[1] + 2 * padding_size), dtype=image.dtype)

    padded_image[padding_size:padding_size + image.shape[0], 
                padding_size:padding_size + image.shape[1]] = image

    return padded_image


def reduce_image_size(image, pixels_to_reduce):
    if image.shape[0] < pixels_to_reduce * 2 or image.shape[1] < pixels_to_reduce * 2:
        raise ValueError("Image dimensions must be at least {}x{} pixels to reduce by {} pixels.".format(
            pixels_to_reduce * 2, pixels_to_reduce * 2, pixels_to_reduce))

    reduced_image = image[pixels_to_reduce:-pixels_to_reduce, 
                        pixels_to_reduce:-pixels_to_reduce]

    return reduced_image


def segment_area_expansion(image, lo_diff, up_diff):
    h, w = image.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)

    seed_point = (w - 1, 0)
    new_value = 255

    cv2.floodFill(image, mask, seed_point, new_value, lo_diff, up_diff, cv2.FLOODFILL_FIXED_RANGE)

    mask = mask[4:-4, 4:-4]

    return mask

def run_morphology_operations(binary_image):
    binary_image = binary_image*255

    #OPENING
    kernel = np.ones((5, 5), np.uint8)
    opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    #DILATION
    dilated_image = cv2.dilate(opened_image, kernel, iterations=10)

    image_after_morphology_operations = dilated_image // 255

    return image_after_morphology_operations































    
    # def make_segmentation(, image_path):
    #     image = cv2.imread(image_path)
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     _, binary_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #     kernel = np.ones((5, 5), np.uint8)
    #     cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    #     cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
    #     segmented_image = cv2.bitwise_and(image, image, mask=cleaned_mask)
    #     return segmented_image

    # def make_kmeans_segmentation_hsv(, image_path, k=3):
    #     # Wczytaj obraz i konwertuj do przestrzeni kolorów HSV
    #     image = cv2.imread(image_path)
    #     image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
    #     # Zmiana kształtu danych do formatu odpowiedniego dla K-means
    #     pixel_values = image_hsv.reshape((-1, 3))
    #     pixel_values = np.float32(pixel_values)
        
    #     # Wykonaj K-means na pikselach obrazu HSV
    #     kmeans = KMeans(n_clusters=k, random_state=42)
    #     kmeans.fit(pixel_values)
    #     labels = kmeans.labels_
    #     centers = np.uint8(kmeans.cluster_centers_)
    #     print("centers")
        
    #     # Stwórz obraz z segmentacją
    #     segmented_image = centers[labels.flatten()]
    #     segmented_image = segmented_image.reshape(image_hsv.shape)
        
    #     # Konwertuj z powrotem na przestrzeń BGR
    #     segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_HSV2BGR)
        
    #     return segmented_image_bgr