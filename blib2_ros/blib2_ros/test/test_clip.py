from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import requests
from sklearn.metrics.pairwise import cosine_similarity

def main():
    # Load model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Input: image + texts
    url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' 
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    texts = ["a cat", "a dog", "a person skating"]

    # Encode inputs
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)

    # Get embeddings
    image_emb = outputs.image_embeds.detach().numpy()
    text_embs = outputs.text_embeds.detach().numpy()

    # Compute cosine similarity
    sims = cosine_similarity(image_emb, text_embs)
    print("Cosine similarities:", sims.flatten())


if __name__ == "__main__":
    main()