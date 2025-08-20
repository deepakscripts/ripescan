import streamlit as st
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="About RipeScan",
    page_icon="ℹ️",
    layout="wide"
)

# --- Asset Paths ---
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
TECH_DIAGRAM_PATH = os.path.join(ASSETS_DIR, "tech_diagram.png") # You will need to create this image

# --- Main Page ---
def about_page():
    """Renders the about page."""

    st.title("ℹ️ About RipeScan")
    st.markdown("### From a curious idea to a smart solution in your pocket.")
    st.write("---")

    # --- Our Mission ---
    st.header("Our Mission: Clarity & Confidence in Every Bite")
    st.write(
        """
        Have you ever stood in a grocery store, holding a fruit, and wondered, "What is this, really?" We have. 
        Our mission with **RipeScan** is to eliminate that moment of uncertainty. We believe that understanding our food is the first step to enjoying it more. 
        We're passionate about using technology to make fresh produce more accessible, understandable, and fun for everyone.
        """
    )
    st.write("---")

    # --- How It Works ---
    st.header("🔬 The Magic Behind the Scan")
    st.write(
        """
        RipeScan isn't magic, but it's close! It's powered by **Deep Learning**, a type of artificial intelligence that learns to recognize patterns from thousands of images. 
        Here’s a simplified look at the journey your image takes:
        """
    )
    
    # Updated to use the local image file
    if os.path.exists(TECH_DIAGRAM_PATH):
        st.image(TECH_DIAGRAM_PATH, caption="From Your Camera to a Confident Classification")
    else:
        st.warning(f"Tech diagram not found. Please add 'tech_diagram.png' to the 'assets' folder.")


    with st.expander("Want the nerdy details? Click here!"):
        st.markdown(
            """
            1.  **Image Preprocessing**: Your uploaded image is resized to `224x224` pixels and normalized to match the format our models were trained on.
            2.  **Feature Extraction**: The image is passed to a pre-trained **Convolutional Neural Network (CNN)**—either VGG16 or ResNet50. The model's deep layers, trained on millions of images, are experts at identifying fundamental features like textures, colors, and shapes.
            3.  **Classification Head**: We've added custom layers on top of the base model. These layers take the extracted features and learn to associate them specifically with our 10 fruit classes.
            4.  **Softmax Activation**: The final layer uses a softmax function to convert the model's output into a set of probabilities, telling us how confident it is for each fruit. The highest probability is your result!
            """
        )
    
    st.write("---")

    # --- Interactive Quiz ---
    st.header("🍓 Fun Fruit Quiz!")
    st.write("Think you're a fruit expert? Let's see! Which fruit is known as the 'king of fruits' in many tropical regions?")

    # Quiz Questions
    quiz_options = ["Apple", "Mango", "Banana", "Pomegranate"]
    user_answer = st.radio("Select your answer:", quiz_options, index=None)

    if st.button("Submit Answer"):
        if user_answer == "Mango":
            st.success("You got it! The Mango is often called the 'king of fruits.' 👑")
            st.balloons()
        elif user_answer is None:
            st.warning("Please select an answer!")
        else:
            st.error(f"Not quite! While the {user_answer} is delicious, the correct answer is Mango. �")

    st.write("---")
    st.info("This project was created with a passion for technology and fresh food. We hope you enjoy using RipeScan.")

if __name__ == "__main__":
    about_page()
