import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class ObjectIdentification:
    def __init__(self):
        # Load the pre-trained CLIP model and processor
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

    def identify_object(self, image_path):
        """
        Identifies the object in the given image.

        Args:
            image_path (str): The path to the input image.

        Returns:
            Tensor: Features extracted from the image.
        """
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")

        # Perform the model inference
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)

        print("Image Path:", image_path)  # Debug print
        print("Outputs Shape:", outputs.shape)  # Debug print
        return outputs

if __name__ == "__main__":
    identifier = ObjectIdentification()
    output = identifier.identify_object("D:\\AI Pipeline for Image Segmentation\\data\\input_images\\2008_000018.jpg")
    print("Identification Output:", output)
