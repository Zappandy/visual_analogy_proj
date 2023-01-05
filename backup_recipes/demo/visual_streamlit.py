import io
import streamlit as st
import torch
import requests
import timm
from PIL import Image
from utils import display_recipes
from transformers import DetrFeatureExtractor, DetrForObjectDetection

def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        pil_img = Image.open(io.BytesIO(image_data))
        return pil_img


def viz_load_model():
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-101")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")
    return model, feature_extractor

def text_model():
    
    items = [
        "macaroni, butter, salt, bacon, milk, flour, pepper, cream corn",
        "provolone cheese, bacon, bread, ginger"
    ]
    
    items = ["bananas", "pineapple", "avocado"]

    items = [
        "salmon, butter, salt, spinach, milk, flour, pepper, cream corn",
        "provolone cheese, bread, ginger"]
    display_recipes(items)

def objects(model, feature_extractor, outputs, target_sizes):
    results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]

    ingredients = set()
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        # let's only keep detections with score > 0.9
        if score > 0.7:
            ingr = model.config.id2label[label.item()]
            print(
                f"Detected {ingr} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )
            ingredients.add(ingr)
    return ingredients

def main():
    st.title('Image upload demo')
    image = load_image()
    model, feature_extract = viz_load_model()

    inputs = feature_extract(images=image, return_tensors="pt")
    outputs = model(**inputs)
    ner_ingredients = objects(model=model, feature_extractor=feature_extract, outputs=outputs, target_sizes=torch.tensor([image.size[::-1]]))
    print(ner_ingredients)
    text_model()


if __name__ == '__main__':
    main()