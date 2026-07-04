import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

st.set_page_config(page_title="Pneumonia Detector", layout="centered")
st.title("🫁 Pneumonia Detection from Chest X-Rays")
st.write("Upload a chest X-ray image to check for signs of pneumonia, with explainable AI visualization.")

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load('resnet18_pneumonia.pth', map_location='cpu'))
    model.eval()

    # Disable inplace ReLU for Grad-CAM compatibility
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)
    input_tensor.requires_grad_(True)

    with st.spinner("Analyzing..."):
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()

        classes = ['NORMAL', 'PNEUMONIA']
        prediction = classes[predicted_class]
        confidence = probabilities[predicted_class].item() * 100

        # Grad-CAM
        target_layers = [model.layer4[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor)[0]

        img_np = input_tensor.squeeze(0).detach().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)

        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    st.subheader("Results")
    if prediction == "PNEUMONIA":
        st.error(f"Prediction: **{prediction}** ({confidence:.1f}% confidence)")
    else:
        st.success(f"Prediction: **{prediction}** ({confidence:.1f}% confidence)")

    st.subheader("Grad-CAM Explanation")
    st.image(visualization, caption="Regions influencing the prediction", use_column_width=True)
    st.caption("Red/warm areas indicate regions the model focused on most for its prediction.")