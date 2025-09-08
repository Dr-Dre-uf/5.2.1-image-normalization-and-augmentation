import streamlit as st
import numpy as np
from skimage import transform
import random
from PIL import Image

# Instruction to adjust parameters in the sidebar
st.write("Adjust image processing parameters in the sidebar on the left to see the effects.")

# Sidebar controls with tooltips
st.sidebar.header("Image Processing Parameters")
image_source = st.sidebar.radio(
    "Select Image Source:",
    ("Fluorescence (IF)", "Brightfield (BF)", "Upload Image"),
    help="Choose between pre-loaded images or upload your own."
)
normalization_factor = st.sidebar.slider(
    "Normalization Factor",
    0.0, 1.0, 1.0,
    help="Adjust the intensity of the image. Lower values darken, higher values brighten."
)
rotation_angle = st.sidebar.slider(
    "Rotation Angle (degrees)",
    -30.0, 30.0, 0.0,
    help="Rotate the image clockwise or counterclockwise."
)
flip_horizontal = st.sidebar.checkbox(
    "Flip Horizontal",
    help="Mirror the image horizontally."
)
flip_vertical = st.sidebar.checkbox(
    "Flip Vertical",
    help="Mirror the image vertically."
)

# Disclaimer
st.sidebar.markdown("⚠️ **Disclaimer:** Please do not upload any sensitive or confidential data. This application is for demonstration purposes only.")

# Load images based on selection
if image_source == "Fluorescence (IF)":
    try:
        image = Image.open('assets/IFCells.jpg')
        print(f"Image IF loaded with size: {image.size}")
    except FileNotFoundError as e:
        st.error(f"Image not found: {e}")
        st.stop()
elif image_source == "Brightfield (BF)":
    try:
        image = Image.open('assets/BloodSmear.png')
        print(f"Image BF loaded with size: {image.size}")
    except FileNotFoundError as e:
        st.error(f"Image not found: {e}")
        st.stop()
else:  # Upload Image
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        print(f"Uploaded image loaded with size: {image.size}")
    else:
        st.warning("Please upload an image.")
        st.stop()

# Convert to numpy array once
img_array = np.array(image)

# Normalize the image
img_array = img_array * normalization_factor

# Apply augmentation
angle = rotation_angle
img_array = transform.rotate(img_array, angle, mode='wrap')

if flip_horizontal:
    img_array = np.fliplr(img_array)
if flip_vertical:
    img_array = np.flipud(img_array)

# Convert back to PIL Image
image_processed = Image.fromarray(img_array.astype(np.uint8))

# Streamlit app - Display images side by side
col1, col2 = st.columns(2)

with col1:
    st.subheader(f'Original Image ({image_source})')
    st.image(image, channels='RGB')

with col2:
    st.subheader(f'Processed Image ({image_source})')
    st.image(image_processed, channels='RGB')