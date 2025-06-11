import torch
from transformers import Blip2Processor, Blip2Model
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import requests

def main():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2Model.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.eval()

    url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png'
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    text = "dog"

    inputs = processor(images=image, text=text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        # Bild-Embedding (Text ist leer)
        inputs_img = processor(images=image, text="", return_tensors="pt").to(model.device)
        outputs_img = model(**inputs_img)
        image_embeds = outputs_img['qformer_outputs'].last_hidden_state.mean(dim=1).cpu().numpy()

        # Text-Embedding (Bild ist gleich, Text ist gesetzt)
        inputs_txt = processor(images=image, text=text, return_tensors="pt").to(model.device)
        outputs_txt = model(**inputs_txt)
        text_embeds = outputs_txt['qformer_outputs'].last_hidden_state.mean(dim=1).cpu().numpy()


    similarity = cosine_similarity(image_embeds, text_embeds)
    print("Cosine similarity:", similarity[0][0])

if __name__ == "__main__":
    main()