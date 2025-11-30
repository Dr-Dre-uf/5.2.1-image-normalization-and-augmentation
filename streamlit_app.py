import streamlit as st
import numpy as np
from PIL import Image
from skimage import transform, img_as_ubyte
from skimage.util import random_noise
import cv2

# ==============================

# App Setup

# ==============================

st.set_page_config(page_title="Image Processing Suite", layout="wide")

st.sidebar.title("Image Processing Suite")
app_mode = st.sidebar.radio(
"Select App",
[
"Image Processing & Augmentation",
"Edge Detection",
"Motion Blur Simulation",
"Salt & Pepper Noise & Denoising"
]
)

# Default image paths

BF_PATH = "assets/BloodSmear.png"
IF_PATH = "assets/IFCells.jpg"

# ==============================

# Helper function to load images

# ==============================

def load_image(uploaded_file=None, choice=None):
if uploaded_file is not None:
return np.array(Image.open(uploaded_file).convert("RGB"))
elif choice is not None:
if choice == "Fluorescence (IFCells)":
return np.array(Image.open(IF_PATH).convert("RGB"))
else:
return np.array(Image.open(BF_PATH).convert("RGB"))
return None

# ==============================

# 1. Image Processing & Augmentation

# ==============================

if app_mode == "Image Processing & Augmentation":
st.title("Image Processing & Augmentation")

```
st.sidebar.header("Image Source")
image_source = st.sidebar.radio(
    "Select Image Source:",
    ("Fluorescence (IF)", "Brightfield (BF)", "Upload Image")
)

normalization_factor = st.sidebar.slider("Normalization Factor", 0.0, 1.0, 1.0)
rotation_angle = st.sidebar.slider("Rotation Angle (degrees)", -30.0, 30.0, 0.0)
flip_horizontal = st.sidebar.checkbox("Flip Horizontal")
flip_vertical = st.sidebar.checkbox("Flip Vertical")

uploaded_file = None
if image_source == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png"])
    if uploaded_file is None:
        st.warning("Please upload an image.")
        st.stop()
    img = np.array(Image.open(uploaded_file))
else:
    img_path = IF_PATH if image_source == "Fluorescence (IF)" else BF_PATH
    img = np.array(Image.open(img_path))

# Normalize & augment
img_processed = img * normalization_factor
img_processed = transform.rotate(img_processed, rotation_angle, mode='wrap')
if flip_horizontal:
    img_processed = np.fliplr(img_processed)
if flip_vertical:
    img_processed = np.flipud(img_processed)
img_processed = Image.fromarray(img_processed.astype(np.uint8))

# Display
col1, col2 = st.columns(2)
col1.subheader("Original Image")
col1.image(img, channels="RGB")
col2.subheader("Processed Image")
col2.image(img_processed, channels="RGB")
```

# ==============================

# 2. Edge Detection

# ==============================

elif app_mode == "Edge Detection":
st.title("Edge Detection with Filters")

```
st.sidebar.header("Image Selection")
use_uploaded = st.sidebar.checkbox("Upload your own image")
uploaded_file = None
if use_uploaded:
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg","png"])
else:
    image_choice = st.sidebar.selectbox("Select Default Image", ["Fluorescence (IFCells)", "Brightfield (BloodSmear)"])

st.sidebar.header("Filter Settings")
horiz_strength = st.sidebar.slider("Horizontal Filter Strength", 0.5, 5.0, 1.0)
vert_strength = st.sidebar.slider("Vertical Filter Strength", 0.5, 5.0, 1.0)
sobel_strength = st.sidebar.slider("Sobel Filter Strength", 0.5, 5.0, 1.0)

# Load image
img = load_image(uploaded_file, image_choice if not use_uploaded else None)
if img is not None:
    # Filters
    base_horiz = np.array([[1,1,1],[0,0,0],[-1,-1,-1]], np.float32)
    base_vert = np.array([[1,0,-1],[1,0,-1],[1,0,-1]], np.float32)
    horiz_filter = horiz_strength * base_horiz
    vert_filter = vert_strength * base_vert

    # Apply
    def apply_filter_rgb(img, kernel):
        channels = cv2.split(img)
        filtered = [cv2.filter2D(c, -1, kernel) for c in channels]
        return cv2.merge(filtered)

    img_horiz = apply_filter_rgb(img, horiz_filter)
    img_vert = apply_filter_rgb(img, vert_filter)
    E_custom = np.sqrt(img_horiz.astype(np.float32)**2 + img_vert.astype(np.float32)**2)
    E_custom = cv2.normalize(E_custom, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F,0,1,ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sobel = cv2.convertScaleAbs(sobel * sobel_strength)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.image(img, caption="Original", use_container_width=True)
    col2.image(img_horiz, caption="Horizontal Edges", use_container_width=True)
    col3.image(img_vert, caption="Vertical Edges", use_container_width=True)
    col4.image(E_custom, caption="Edge Magnitude", use_container_width=True)
    col5.image(sobel, caption="Sobel Edges", use_container_width=True)
else:
    st.info("Please select or upload an image to begin.")
```

# ==============================

# 3. Motion Blur Simulation

# ==============================

elif app_mode == "Motion Blur Simulation":
st.title("Motion Blur Simulation")
st.sidebar.header("Image Selection")
use_uploaded = st.sidebar.checkbox("Upload your own image")
uploaded_file = None
if use_uploaded:
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg","png"])
else:
image_choice = st.sidebar.selectbox("Select Example Image", ["Fluorescence (IFCells)", "Brightfield (BloodSmear)"])

```
st.sidebar.header("Motion Blur Settings")
blur_length = st.sidebar.slider("Blur Length", 3, 50, 20)
blur_angle = st.sidebar.slider("Blur Angle (degrees)", 0, 180, 45)

img = load_image(uploaded_file, image_choice if not use_uploaded else None)
if img is not None:
    kernel = np.zeros((blur_length, blur_length))
    kernel[int((blur_length-1)/2), :] = np.ones(blur_length)
    M = cv2.getRotationMatrix2D((blur_length/2-0.5, blur_length/2-0.5), blur_angle, 1)
    kernel = cv2.warpAffine(kernel, M, (blur_length, blur_length))
    kernel /= blur_length

    img_motion = cv2.filter2D(img, -1, kernel)
    col1, col2 = st.columns(2)
    col1.image(img, caption="Original Image", use_container_width=True)
    col2.image(img_motion, caption="With Motion Blur", use_container_width=True)
else:
    st.info("Please upload an image or select one from the examples to begin.")
```

# ==============================

# 4. Salt & Pepper Noise & Denoising

# ==============================

elif app_mode == "Salt & Pepper Noise & Denoising":
st.title("Salt & Pepper Noise and Denoising")

```
st.sidebar.header("Image Selection")
use_uploaded = st.sidebar.checkbox("Upload your own image")
uploaded_file = None
if use_uploaded:
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg","png"])
else:
    image_choice = st.sidebar.selectbox("Select Example Image", ["Fluorescence (IFCells)", "Brightfield (BloodSmear)"])

st.sidebar.header("Noise Settings")
noise_amount = st.sidebar.slider("Noise Amount", 0.0, 0.2, 0.05)
st.sidebar.header("Denoising Filter")
filter_type = st.sidebar.radio("Filter Type", ["Median", "Gaussian"])
if filter_type == "Median":
    filter_strength = st.sidebar.slider("Kernel Size (odd only)", 3, 11, 3, step=2)
else:
    filter_strength = st.sidebar.slider("Gaussian Sigma", 0.5, 5.0, 1.0, step=0.5)

img = load_image(uploaded_file, image_choice if not use_uploaded else None)
if img is not None:
    noisy = random_noise(img, mode="s&p", amount=noise_amount)
    noisy_u8 = img_as_ubyte(noisy)
    if filter_type=="Median":
        denoised = cv2.medianBlur(noisy_u8, filter_strength)
    else:
        denoised = cv2.GaussianBlur(noisy_u8, (5,5), filter_strength)

    col1, col2, col3 = st.columns(3)
    col1.image(img, caption="Original", use_container_width=True)
    col2.image(noisy, caption="With Salt & Pepper Noise", use_container_width=True)
    col3.image(denoised, caption=f"Denoised ({filter_type})", use_container_width=True)
else:
    st.info("Please upload an image or select one from the examples to begin.")
```
