import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Model Details",
    page_icon="🤖",
    layout="wide"
)

# --- Asset Paths ---
ASSETS_DIR = "assets"
PLOTS_DIR = os.path.join(ASSETS_DIR, "plots")
REPORTS_DIR = os.path.join(ASSETS_DIR, "reports")

# --- Helper Functions ---
@st.cache_data
def load_history(model_prefix):
    """Loads the training history CSV for a specific model."""
    history_file = os.path.join(REPORTS_DIR, f"{model_prefix}_history.csv")
    try:
        return pd.read_csv(history_file)
    except FileNotFoundError:
        st.error(f"History file not found. Make sure '{history_file}' exists.", icon="❌")
        return None

def display_architecture(model_name, code_snippet):
    """Displays the model architecture explanation and code."""
    st.subheader(f"{model_name} Architecture")
    if model_name == "VGG16":
        st.markdown("""
        **VGG16** is a classic Convolutional Neural Network (CNN) known for its simplicity and depth. It uses a consistent pattern of `3x3` convolutional layers stacked on top of each other.
        
        - **Base Model**: We use the pre-trained VGG16 model from Keras, initially freezing its layers to leverage its learned features from the large ImageNet dataset.
        - **Custom Head**: We add our own classification head on top of the base model. This includes:
            - `GlobalAveragePooling2D`: To reduce the spatial dimensions to a single vector.
            - `Dense` layers with `ReLU` activation for learning complex patterns.
            - `Dropout`: To prevent overfitting by randomly dropping neurons during training.
            - **Final `Dense` Layer**: A layer with `softmax` activation to output probabilities for our 10 fruit classes.
        - **Fine-Tuning**: After initial training of the head, we unfreeze some of the top layers of the base VGG16 model and retrain the entire network with a very low learning rate. This allows the model to adapt its pre-trained features specifically to our fruit dataset.
        """)
    else: # ResNet50
        st.markdown("""
        **ResNet50** is a more modern and deeper CNN that introduced the concept of "residual connections" or shortcuts. These connections help mitigate the vanishing gradient problem, allowing for much deeper networks.

        - **Base Model**: We use the pre-trained ResNet50 model, which has 50 layers. Its layers are also initially frozen.
        - **Residual Blocks**: The core idea is that some layers are skipped, allowing the gradient to flow more easily during backpropagation. This helps the model learn identity functions, meaning it can easily pass the input through if a layer is not needed.
        - **Custom Head**: Similar to the VGG16 setup, we add a custom head with `GlobalAveragePooling2D`, `Dense` layers, `Dropout`, and a final `softmax` layer for classification.
        - **Fine-Tuning**: The fine-tuning process is also similar, where we unfreeze the top layers of ResNet50 and continue training with a low learning rate to specialize the model for fruit classification.
        """)
    st.code(code_snippet, language='python')

def display_plots(model_prefix):
    """Displays the pre-generated plot images for the model."""
    st.subheader("Plots & Metrics")
    st.markdown("**Evaluation Visuals**")

    # Define paths for all plot images
    cm_path = os.path.join(PLOTS_DIR, f'{model_prefix}_confusion_matrix.png')
    curves_path = os.path.join(PLOTS_DIR, f'{model_prefix}_accuracy_loss_curves.png')
    aug_path = os.path.join(PLOTS_DIR, 'augmentation_samples.png')

    col1, col2, col3 = st.columns(3)

    with col1:
        if os.path.exists(cm_path):
            st.image(cm_path, caption=f'{model_prefix.upper()} Confusion Matrix')
            st.markdown("Shows correct vs. incorrect predictions for each class.")
        else:
            st.warning(f"Confusion matrix not found at '{cm_path}'")
    
    with col2:
        if os.path.exists(curves_path):
            st.image(curves_path, caption=f'{model_prefix.upper()} Accuracy & Loss')
            st.markdown("Tracks the model's learning progress over time.")
        else:
            st.warning(f"Accuracy/Loss curves not found at '{curves_path}'")

    with col3:
        if os.path.exists(aug_path):
            st.image(aug_path, caption="Augmentation Samples")
            st.markdown("Examples of artificially modified training images.")
        else:
            st.warning(f"Augmentation samples not found at '{aug_path}'")


def display_reports(history_df, model_prefix):
    """Displays classification report and training history."""
    st.subheader("Reports")
    
    # Classification Report
    report_path = os.path.join(REPORTS_DIR, f'{model_prefix}_classification_report.txt')
    if os.path.exists(report_path):
        st.markdown("**Classification Report**")
        with open(report_path, 'r') as f:
            report_text = f.read()
        st.code(report_text, language='text')
        st.info("""
        - **Precision**: Of all the predictions for a class, how many were correct?
        - **Recall**: Of all the actual instances of a class, how many did the model correctly identify?
        - **F1-Score**: The harmonic mean of precision and recall, providing a single metric for performance.
        - **Support**: The number of actual occurrences of each class in the test set.
        """)
    else:
        st.warning(f"Classification report not found at '{report_path}'")
        
    # Training History Table
    st.markdown("**Training History Data**")
    st.dataframe(history_df)


# --- Main App ---
def model_page():
    """Renders the model details page."""
    st.title("🤖 Model Deep Dive")
    st.markdown("Explore the architecture, performance, and reports for each of our trained models.")

    # --- Model Code Snippets from Notebook ---
    vgg16_code = """
# Base Model: VGG16
base_model_vgg = tf.keras.applications.VGG16(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)
base_model_vgg.trainable = False # Freeze base layers

# Custom Head
inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base_model_vgg(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
model_vgg = tf.keras.Model(inputs, outputs)
    """

    resnet50_code = """
# Base Model: ResNet50
base_model_resnet = tf.keras.applications.ResNet50(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)
base_model_resnet.trainable = False # Freeze base layers

# Custom Head
inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base_model_resnet(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
model_resnet = tf.keras.Model(inputs, outputs)
    """

    # --- Tabs for each model ---
    tab1, tab2 = st.tabs(["VGG16", "ResNet50"])

    with tab1:
        st.header("VGG16 Model Details")
        vgg16_history = load_history('vgg16')
        if vgg16_history is not None:
            # Display key metrics
            final_acc = vgg16_history['val_accuracy'].iloc[-1]
            final_loss = vgg16_history['val_loss'].iloc[-1]
            col1, col2 = st.columns(2)
            col1.metric("Final Validation Accuracy", f"{final_acc:.2%}")
            col2.metric("Final Validation Loss", f"{final_loss:.4f}")

            with st.expander("Show Architecture Details", expanded=True):
                display_architecture("VGG16", vgg16_code)
            
            with st.expander("Show Plots & Metrics", expanded=True):
                display_plots('vgg16')
                
            with st.expander("Show Reports", expanded=True):
                display_reports(vgg16_history, 'vgg16')
            
    with tab2:
        st.header("ResNet50 Model Details")
        resnet50_history = load_history('resnet50')
        if resnet50_history is not None:
            # Display key metrics
            final_acc = resnet50_history['val_accuracy'].iloc[-1]
            final_loss = resnet50_history['val_loss'].iloc[-1]
            col1, col2 = st.columns(2)
            col1.metric("Final Validation Accuracy", f"{final_acc:.2%}")
            col2.metric("Final Validation Loss", f"{final_loss:.4f}")

            with st.expander("Show Architecture Details", expanded=True):
                display_architecture("ResNet50", resnet50_code)

            with st.expander("Show Plots & Metrics", expanded=True):
                display_plots('resnet50')

            with st.expander("Show Reports", expanded=True):
                display_reports(resnet50_history, 'resnet50')

if __name__ == "__main__":
    # To make this page runnable standalone, we need to ensure the asset paths are correct
    if not os.path.exists(ASSETS_DIR):
        # Adjust path if running from inside the `pages` directory
        ASSETS_DIR = os.path.join("..", ASSETS_DIR)
        PLOTS_DIR = os.path.join(ASSETS_DIR, "plots")
        REPORTS_DIR = os.path.join(ASSETS_DIR, "reports")
    
    model_page()