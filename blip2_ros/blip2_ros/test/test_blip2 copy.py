import sys
import os
import torch
import transformers
import huggingface_hub
from PIL import Image
import requests
import cv2
import numpy as np
from transformers import Blip2Model, AutoProcessor
import torch.nn.functional as F

def main(args=None):
    print("Python Version:", sys.version)
    print("PyTorch Version:", torch.__version__)
    print("Transformers Version:", transformers.__version__)
    print("Hugging Face Hub Version:", huggingface_hub.__version__)
    
    print("CUDA verfügbar:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA Version:", torch.version.cuda)
        print("CUDA Device Name:", torch.cuda.get_device_name(0))

    # Modell und Prozessor laden
    model_name = "Salesforce/blip2-opt-2.7b"
    processor = AutoProcessor.from_pretrained(model_name)
    model = Blip2Model.from_pretrained(model_name, torch_dtype=torch.float16)

    model_path = transformers.utils.hub.cached_file(model_name, "config.json")
    print("Modell wurde geladen von:", os.path.dirname(model_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Bild laden
    url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' 
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')

    # Optional anzeigen
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.imshow('Image', image_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Prompt definieren
    text = "a large statue near water"

    # Eingaben vorbereiten
    image_inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    text_inputs = processor(text=text, return_tensors="pt").to(device, torch.float16)

    with torch.no_grad():
        # Bild-Embedding + Projektion
        image_outputs = model.vision_model(**image_inputs)
        image_embeds = image_outputs.last_hidden_state[:, 0, :]  # CLS-Token
        image_proj = model.vision_proj(image_embeds)  # [1, D]

        # Text-Embedding + Projektion
        text_inputs["output_hidden_states"] = True
        text_inputs["use_cache"] = False
        text_outputs = model.language_model(**text_inputs)
        text_embeds = text_outputs.hidden_states[-1].mean(dim=1)  # [1, 2560]
        text_proj = model.text_proj(text_embeds)  # [1, D]

        # Cosine Similarity im gemeinsamen Raum
        similarity = F.cosine_similarity(image_proj, text_proj)
        print(f"Cosine Similarity: {similarity.item():.4f}")

if __name__ == "__main__":
    main()
