import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import os
import time

# --- Configuration ---
IMG_HEIGHT, IMG_WIDTH = 224, 224
CONFIDENCE_THRESHOLD = 95.0
MODEL_DIR = "models"
ASSETS_DIR = "assets"
MAX_SINGLE_SIZE = 5 * 1024 * 1024  # 5MB
MAX_BATCH_SIZE = 50 * 1024 * 1024  # 50MB
MAX_BATCH_IMAGES = 10

# --- Page Configuration ---
st.set_page_config(
    page_title="RipeScan: A Fresh Produce Classifier",
    page_icon="🍓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching Functions for Performance ---
@st.cache_resource(max_entries=2)
def load_tf_model(model_name):
    """Loads a TensorFlow model with enhanced caching and error handling."""
    model_file = {
        "VGG16": "vgg16_10class_best (1).keras",
        "ResNet50": "resnet50_best.keras"
    }.get(model_name)
    
    if not model_file:
        st.error(f"Invalid model selection: {model_name}", icon="❌")
        return None

    model_path = os.path.join(MODEL_DIR, model_file)
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model {model_name}: {str(e)}", icon="❌")
        return None

@st.cache_data
def load_class_names():
    """Loads class names from a text file with caching."""
    filepath = os.path.join(ASSETS_DIR, 'class_names.txt')
    try:
        with open(filepath, 'r') as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        st.error(f"Class names file not found at '{filepath}'", icon="❌")
        return []
    except Exception as e:
        st.error(f"Error reading class names: {str(e)}", icon="❌")
        return []

# --- Image Processing Functions ---
def validate_image_file(file, is_batch=False):
    """Validates image file size and type."""
    if file.size > MAX_SINGLE_SIZE and not is_batch:
        st.warning(f"⚠️ {file.name} exceeds 5MB limit. Skipping.")
        return False
    return True

def preprocess_image(image):
    """Preprocesses an uploaded image for model prediction with error handling."""
    try:
        img = Image.open(image).convert('RGB')
        img = img.resize((IMG_HEIGHT, IMG_WIDTH))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        return tf.keras.applications.resnet.preprocess_input(img_array)
    except UnidentifiedImageError:
        st.error(f"🖼️ Cannot identify image file: {image.name}", icon="❌")
    except Exception as e:
        st.error(f"🖼️ Error processing {image.name}: {str(e)}", icon="❌")
    return None

# --- Prediction Functions ---
def batch_predict(model, images, class_names):
    """Performs batch prediction with confidence thresholding."""
    try:
        batch_tensor = tf.stack([tf.expand_dims(img, 0) for img in images])
        batch_tensor = tf.squeeze(batch_tensor, axis=1)
        predictions = model.predict(batch_tensor)
    except Exception as e:
        st.error(f"🤖 Prediction error: {str(e)}", icon="❌")
        return []

    results = []
    for pred in predictions:
        confidence = np.max(pred) * 100
        if confidence < CONFIDENCE_THRESHOLD:
            results.append(("Unknown / Not a fruit", confidence))
        else:
            class_idx = np.argmax(pred)
            results.append((class_names[class_idx], confidence))
    return results

# --- UI Components ---
def display_single_result(image, result):
    """Displays single image result in centered layout."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Your Uploaded Image", use_container_width=True)
        class_name, confidence = result
        if class_name == "Unknown / Not a fruit":
            st.warning(f"**Prediction:** {class_name}\n\n**Confidence:** {confidence:.2f}% (Below {CONFIDENCE_THRESHOLD}% threshold)", icon="❓")
        else:
            st.success(f"**Prediction:** {class_name}\n\n**Confidence:** {confidence:.2f}%", icon="✅")

def display_batch_results(images, results):
    """Displays batch results in responsive grid layout."""
    cols_per_row = 4
    for i in range(0, len(images), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i + j
            if idx < len(images):
                with cols[j]:
                    st.image(
                        images[idx], 
                        caption=images[idx].name,
                        use_container_width=True
                    )
                    class_name, confidence = results[idx]
                    if class_name == "Unknown / Not a fruit":
                        st.warning(f"{class_name} ({confidence:.2f}%)", icon="❓")
                    else:
                        st.success(f"{class_name} ({confidence:.2f}%)", icon="✅")

# --- Main Application Logic ---
def main():
    """Main application function with optimized workflow."""
    # --- NEW, MORE ROBUST CSS FOR UNIFORM IMAGE SIZING ---
    st.markdown("""
        <style>
            /* Target the image container specifically within Streamlit's columns */
            div[data-testid="stImage"] {
                height: 250px; /* Set a fixed height for the image container */
                display: flex;
                justify-content: center;
                align-items: center;
                overflow: hidden; /* Hide parts of the image that overflow */
            }
            /* Style the image itself */
            div[data-testid="stImage"] img {
                height: 100%; /* Make the image fill the container's height */
                width: 100%; /* Make the image fill the container's width */
                object-fit: cover; /* Cover the area, cropping if necessary, without distortion */
                border-radius: 10px; /* Optional: adds rounded corners */
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🍓 RipeScan: A Fresh Produce Classifier")
    st.caption("Upload your fruit images and let our AI identify them!")

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        model_name = st.radio(
            "**Choose a Model**",
            ("VGG16", "ResNet50"),
            index=0,
            help="VGG16: Classic CNN architecture, ResNet50: Deeper residual network"
        )
        
        upload_type = st.radio(
            "**Upload Type**",
            ("Single Image", "Batch of Images"),
            index=0,
            help="Single: Max 5MB, Batch: Max 10 images (50MB total)"
        )
        
        st.markdown("---")
        st.info("Navigate to other pages from the top-left menu.")

    # --- File Upload Section ---
    st.info(f"**Selected Model:** `{model_name}`", icon="🤖")
    uploaded_files = None
    
    if upload_type == "Single Image":
        uploaded_files = st.file_uploader(
            "Upload a fruit image",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=False
        )
    else:
        uploaded_files = st.file_uploader(
            "Upload fruit images (max 10)",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True
        )
        if uploaded_files and len(uploaded_files) > MAX_BATCH_IMAGES:
            st.warning(f"Maximum {MAX_BATCH_IMAGES} images allowed. Using first {MAX_BATCH_IMAGES}.", icon="⚠️")
            uploaded_files = uploaded_files[:MAX_BATCH_IMAGES]

    # --- Processing Pipeline ---
    if uploaded_files and st.button("🔍 Classify Fruit(s)", type="primary", use_container_width=True):
        files = [uploaded_files] if not isinstance(uploaded_files, list) else uploaded_files
        
        is_batch = upload_type == "Batch of Images"
        valid_files = [f for f in files if validate_image_file(f, is_batch)]
        total_size = sum(f.size for f in valid_files)
        
        if not valid_files:
            st.warning("No valid images to process", icon="⚠️")
            return
            
        if is_batch and total_size > MAX_BATCH_SIZE:
            st.error(f"Total size {total_size/1024/1024:.2f}MB exceeds 50MB limit!", icon="❌")
            return

        model = load_tf_model(model_name)
        class_names = load_class_names()
        
        if not model or not class_names:
            st.error("Initialization failed. Cannot proceed.", icon="❌")
            return

        progress_bar = st.progress(0, text="Preprocessing images...")
        processed_images = []
        valid_images = []
        
        for i, file in enumerate(valid_files):
            progress_bar.progress((i+1)/len(valid_files), text=f"Processing {file.name}...")
            img_tensor = preprocess_image(file)
            if img_tensor is not None:
                processed_images.append(img_tensor)
                valid_images.append(file)
        
        if not processed_images:
            progress_bar.empty()
            st.error("All images failed preprocessing", icon="❌")
            return

        progress_bar.progress(0, text="Making predictions...")
        results = batch_predict(model, processed_images, class_names)
        progress_bar.progress(100, text="Complete!")
        time.sleep(0.5)
        progress_bar.empty()

        st.markdown("---")
        st.subheader("Classification Results")
        
        if upload_type == "Single Image" and len(valid_images) == 1:
            display_single_result(valid_images[0], results[0])
        else:
            display_batch_results(valid_images, results)
        
        st.balloons()

if __name__ == "__main__":
    main()
