# Rice Classifier 🍚 | Image-based Rice Variety Detection with Deep Learning

[![HF Space](https://img.shields.io/badge/Try%20Demo-HuggingFace-blue.svg)](https://huggingface.co/spaces/chfida/Rice-Classification)

This project demonstrates a **deep learning-based image classifier** trained to detect and classify rice grains into five distinct varieties. This model is particularly useful for automated quality checks, research, and agricultural supply chain validation.

---

## 🚀 Live Demo

Test the model live using Hugging Face Spaces:

🔗 **Demo Link**: [Rice Classification Model](https://huggingface.co/spaces/chfida/Rice-Classification)

📦 **Embed iFrame** (for websites or dashboards):

```html
<iframe
  src="https://chfida-Rice-Classification.hf.space"
  frameborder="0"
  width="100%"
  height="500">
</iframe>
```
## 🍚 Classes Detected
The model is trained to classify rice into the following 5 categories:

- 1121
- RH-10
- Sharbati
- Sona Masoori
- Sugandha

## 📊 Dataset Overview
Total Images: ~25000 (after pre processing on original dataset)

Format: .png images with class-based folder structure

Classes: 5 (listed above)

Source: Proprietary collection of rice grain images on a black background for better contrast

🔗 Dataset Link: [Rice Classification](https://www.kaggle.com/datasets/fidachaudhary/rice-classification)

## 🛠️ Preprocessing Pipeline
To ensure high-quality training data and balanced representation across all rice varieties, we applied a robust preprocessing pipeline as follows:

### 1. Grain Extraction from Image
Each image in the dataset originally contained up to 20 rice grains placed on a black background. We used image processing techniques to isolate individual grains:

- Grayscale conversion
- Thresholding + Contour Detection
- Cropping individual grains into separate images
- Padding and resizing to 196x196 pixels

This allowed us to transform bulk images into thousands of single-grain samples.

### 2. Dataset Balancing
After extraction:
- We aimed for 5000 images per class
- For classes with fewer than 5000 samples, we applied data augmentation using:
    - Horizontal/vertical flips
    - Rotation
    - Brightness/contrast adjustment
    - Minor affine transforms

This ensured a balanced dataset across all five categories.

### 3. Normalization
All final images were normalized to improve model training:

```python
norm_image = (image - np.min(image)) / (np.max(image) - np.min(image))
```
>✅ Tip: For consistent prediction results, test your rice samples on a plain black background (e.g., black card or sheet).

## 🧠 Model Summary
- Architecture: CNN-based custom image classifier
- Input Size: 196x196 pixels
- Training Accuracy: 98%
- Validation Accuracy: 96%
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Epochs: 50

## ⚙️ How to Use
### 🖱️ Option 1: Try in Browser
Upload your rice image using the live demo on Hugging Face:

🔗 [Try the Live Demo](https://chfida-Rice-Classification.hf.space)

📌 Tip: Use an image with a black background for best prediction accuracy.
### 🧩 Option 2: Programmatic Access via Gradio Client
```python
from gradio_client import Client, handle_file

client = Client("chfida/Rice-Classification")
result = client.predict(
    image=handle_file("your_image.jpg"),  # Replace with path to your image
    api_name="/classify"
)
print(result)
```
> Note:📦 Make sure you have `gradio_client` installed:
> ```python
>pip install gradio_client
>```
## 📁 Repository Structure
```python
├── app.py             # Gradio app script
├── README.md          # Project documentation
├── images       # Example images for demo
```
> **Note**: This repository does not include model weights or local inference support. All inference runs in the cloud via Hugging Face Spaces.

## ✍️ Author
Fida Muhammad <br>
🔗 [Hugging Face](https://huggingface.co/chfida)<br>
🔗 [Hugging Face Space](https://huggingface.co/chfida/spaces)


## 📜 License
This project is licensed under the MIT License.
You are free to use, modify, and share the code for non-commercial and research purposes.

## 🙏 Acknowledgements
- [Gradio](https://www.gradio.app/)
- [Hugging Face Spaces](https://huggingface.co/spaces)
- [numpy](https://numpy.org/)
