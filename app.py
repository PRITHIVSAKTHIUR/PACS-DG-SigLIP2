import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/PACS-DG-SigLIP2"  # Update to your actual model path on Hugging Face
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Label map
id2label = {
    "0": "art_painting",
    "1": "cartoon",
    "2": "photo",
    "3": "sketch"
}

def classify_pacs_image(image):
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    prediction = {
        id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))
    }

    return prediction

# Gradio Interface
iface = gr.Interface(
    fn=classify_pacs_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=4, label="Predicted Domain Probabilities"),
    title="PACS-DG-SigLIP2",
    description="Upload an image to classify its visual domain: Art Painting, Cartoon, Photo, or Sketch."
)

if __name__ == "__main__":
    iface.launch()
