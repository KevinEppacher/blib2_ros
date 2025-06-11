import torch
from transformers import Blip2Processor, Blip2Model
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import requests


def main():
    
    # Bild laden (lokal oder URL)
    url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' 
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')

    # Text-Kandidaten
    texts = "a large statue near water"

    # Processor + Modell laden
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2Model.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        device_map="auto",
        torch_dtype=torch.float16
    )

    # Inputs vorbereiten
    inputs = processor(images=image, text=texts, return_tensors="pt").to(model.device)
    pixel_values = inputs["pixel_values"]

    # Vorwärtsdurchlauf
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = model.get_image_features(pixel_values=pixel_values)
        # text_embeds = model.get_text_features(**inputs)
        print("Image Embeddings Shape:", image_embeds.shape)

        # image_embeds = outputs.vision_outputs.last_hidden_state[:, 0, :]
        # print("Image Embeddings Shape:", image_embeds.shape)
        # text_embeds = outputs.qformer_outputs.last_hidden_state[:, 0, :]
        # print("Text Embeddings Shape:", text_embeds.shape)
        # # image_embeds = outputs.image_embeds
        # # text_embeds = outputs.text_embeds

    # Cosine Similarity berechnen
    # sims = cosine_similarity(image_embeds.cpu(), text_embeds.cpu())
    # print("Cosine Similarity:", sims.flatten())

if __name__ == "__main__":
    main()