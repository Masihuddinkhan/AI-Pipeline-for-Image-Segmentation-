import torch
import torchvision
from PIL import Image
import torchvision.transforms as T
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import os
import numpy as np

class ImageSegmentation:
    def __init__(self):
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
        self.model.eval()
        self.output_dir = 'D:\\AI Pipeline for Image Segmentation\\data\\segmented_objects'
        os.makedirs(self.output_dir, exist_ok=True)

    def segment_image(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Image open karne mein error: {e}")
            return
        
        transform = T.Compose([T.ToTensor()])
        image_tensor = transform(image)

        with torch.no_grad():
            predictions = self.model([image_tensor])
        
        # Print the number of segmented objects
        print(f"Segmented {len(predictions[0]['masks'])} objects.")

        for idx, (mask, score) in enumerate(zip(predictions[0]['masks'], predictions[0]['scores'])):
            if score > 0.5:  # Threshold to filter out low-confidence detections
                mask = mask.squeeze().cpu().numpy()
                mask_image = Image.fromarray((mask * 255).astype(np.uint8))
                segmented_image_path = os.path.join(self.output_dir, f'object_{idx}.png')
                mask_image.save(segmented_image_path)
                print(f"Segmented image saved at: {segmented_image_path}")  # Print the path of saved image

        return predictions

# Usage
if __name__ == "__main__":
    segmentation = ImageSegmentation()
    # Replace with the path to your input image
    input_image_path = 'D:/AI Pipeline for Image Segmentation/data/input_images/2008_000018.jpg'
    segmentation.segment_image(input_image_path)

    # Final image path check
    final_image_path = "D:\\AI Pipeline for Image Segmentation\\data\\output\\final_image.png"
    if os.path.exists(final_image_path):
        print("Final image found, proceeding to open.")
        # Open the image code here (add your code to open the image)
    else:
        print(f"Final image not found at: {final_image_path}")