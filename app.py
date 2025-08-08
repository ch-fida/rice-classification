import gradio as gr
import numpy as np
import tensorflow as tf
import cv2
import torch
from huggingface_hub import login, hf_hub_download
import os


# Log in to Hugging Face with your token
login(token=os.environ.get('Token', None))

torch.hub.download_url_to_file(
    'https://github.com/ch-fida/Rice-Classification/blob/main/images/1121_14.png?raw=true', '1121_14.png')
torch.hub.download_url_to_file(
    'https://github.com/ch-fida/Rice-Classification/blob/main/images/RH-10_27.png?raw=true', 'RH-10_27.png')
torch.hub.download_url_to_file(
    'https://github.com/ch-fida/Rice-Classification/blob/main/images/Sharbati_146.png?raw=true', 'Sharbati_146.png')
torch.hub.download_url_to_file(
    'https://github.com/ch-fida/Rice-Classification/blob/main/images/Sona%20Masoori_125.png?raw=true', 'Sona Masoori_125.png')
torch.hub.download_url_to_file(
    'https://github.com/ch-fida/Rice-Classification/blob/main/images/Sugandha_128.png?raw=true', 'Sugandha_128.png')

# Load model
model = tf.keras.models.load_model(hf_hub_download(repo_id=os.environ.get('Model', None), filename=os.environ.get('ID', None)))

# Class map
classes = {0: '1121', 1: 'RH-10', 2: 'Sharbati', 3: 'Sona Masoori', 4: 'Sugandha'}

# login(token=os.environ.get('Token', None))


# Prediction function
def classify(image):
    img = np.array(image)
    img = cv2.resize(img, (196, 196), interpolation=cv2.INTER_AREA)
    norm_image = (img - np.min(img)) / (np.max(img) - np.min(img))
    norm_image = np.expand_dims(norm_image, axis=0)
    predictions = model.predict(norm_image)
    return classes[np.argmax(predictions)]

# Examples
examples = [
    ['1121_14.png'],
    ['RH-10_27.png'],
    ['Sharbati_146.png'],
    ['Sona Masoori_125.png'],
    ['Sugandha_128.png']
]

# Build UI with Blocks
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center; font-weight: bold;'>🍚 Rice Classification Model</h1>")

    
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Rice Image")
        label_output = gr.Label(label="Predicted Category")

    classify_button = gr.Button("Classify")
    classify_button.click(fn=classify, inputs=image_input, outputs=label_output)

    gr.Examples(examples=examples, inputs=[image_input])


    # Description at the BOTTOM
    gr.Markdown("""
    ---
    **This model classifies rice into the following categories:**

    - 1121  
    - RH-10  
    - Sharbati  
    - Sona Masoori  
    - Sugandha
    """)

# Launch app
demo.launch(debug=True, share=True, show_api=True, show_error=True)
