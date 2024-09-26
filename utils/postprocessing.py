import cv2
import os

def save_segmented_objects(image, predictions, save_path='data/segmented_objects/'):
    os.makedirs(save_path, exist_ok=True)
    
    # Ensure predictions contain masks
    if 'masks' not in predictions or len(predictions['masks']) == 0:
        print("No masks found in predictions.")
        return

    for i, mask in enumerate(predictions['masks']):
        mask = mask[0].mul(255).byte().cpu().numpy()  # Make sure this step is correct based on your mask format
        ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        segmented_object = cv2.bitwise_and(image, image, mask=thresh)
        cv2.imwrite(f"{save_path}/object_{i}.png", segmented_object)

# Example usage:
if __name__ == "__main__":
    # Replace with actual image and predictions for testing
    image = cv2.imread('path_to_your_image.jpg')  # Load your image here
    predictions = {
        'masks': []  # Replace with your actual prediction masks
    }
    save_segmented_objects(image, predictions)
