![4.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/2M1HRenGKvzLJiAdaexKs.png)

# **PACS-DG-SigLIP2**

> **PACS-DG-SigLIP2** is a vision-language encoder model fine-tuned from **google/siglip2-base-patch16-224** for **multi-class domain generalization** classification. It is trained to distinguish visual domains such as **art paintings**, **cartoons**, **photos**, and **sketches** using the **SiglipForImageClassification** architecture.

> [!note]
*SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features* https://arxiv.org/pdf/2502.14786

```py
Classification Report:
              precision    recall  f1-score   support

art_painting     0.8538    0.9380    0.8939      2048
     cartoon     0.9891    0.9330    0.9603      2344
       photo     0.9029    0.8635    0.8828      1670
      sketch     0.9990    1.0000    0.9995      3929

    accuracy                         0.9488      9991
   macro avg     0.9362    0.9336    0.9341      9991
weighted avg     0.9509    0.9488    0.9491      9991
```

![download (1).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/lCLDN4U4zT8U2viaJyV1d.png)

---

# **ID2Label Mapping**

```py
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("flwrlabs/pacs")

# Extract unique masterCategory values (assuming it's a string field)
labels = sorted(set(example["domain"] for example in dataset["train"]))

# Create id2label mapping
id2label = {str(i): label for i, label in enumerate(labels)}

# Print the mapping
print(id2label)
```

---

## **Label Space: 4 Domain Categories**

The model predicts the most probable visual domain from the following:

```
Class 0: "art_painting"
Class 1: "cartoon"
Class 2: "photo"
Class 3: "sketch"
```

---

## **Install dependencies**

```bash
pip install -q transformers torch pillow gradio
```

---

## **Inference Code**

```python
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
```

---

## **Intended Use**

The **PACS-DG-SigLIP2** model is designed to support tasks in **domain generalization**, particularly:

- **Cross-domain Visual Recognition** – Identify the domain style of an image.
- **Robust Representation Learning** – Aid in training or evaluating models on domain-shifted inputs.
- **Dataset Characterization** – Use as a tool to explore domain imbalance or drift.
- **Educational Tools** – Help understand how models distinguish between stylistic image variations.
