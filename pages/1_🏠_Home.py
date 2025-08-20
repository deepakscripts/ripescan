import streamlit as st
import os
import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="RipeScan Home",
    page_icon="�",
    layout="wide"
)

# --- Asset Paths ---
# Corrected path to go up one directory from 'pages' to the root
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
VIDEO_PATH = os.path.join(ASSETS_DIR, "RipeScan_Promotional_Video_Script.mp4")

# --- Icon Paths ---
QUESTION_ICON_PATH = os.path.join(ASSETS_DIR, "question_icon.png")
CAMERA_ICON_PATH = os.path.join(ASSETS_DIR, "camera_icon.png")
GRID_ICON_PATH = os.path.join(ASSETS_DIR, "grid_icon.png")
SHIELD_ICON_PATH = os.path.join(ASSETS_DIR, "shield_icon.png")


# --- Main Page ---
def home_page():
    """Renders the home page."""

    # --- Header and Video Section ---
    st.title("🍓 Welcome to RipeScan")
    st.markdown("### Know your food. Love your food.")
    st.write("---")

    # The st.video function handles file paths correctly, no need for base64
    if os.path.exists(VIDEO_PATH):
        st.video(VIDEO_PATH)
    else:
        st.error("Video file not found. Please ensure 'RipeScan_Promotional_Video_Script.mp4' is in the 'assets' folder.", icon="📹")


    # --- What Problem Does It Solve? ---
    st.header("🤔 Ever felt puzzled at the fruit market?")
    col1, col2 = st.columns([1, 2])
    with col1:
        if os.path.exists(QUESTION_ICON_PATH):
            st.image(QUESTION_ICON_PATH, width=200)
    with col2:
        st.write(
            """
            You're exploring a vibrant market, and you see a delicious-looking fruit you've never encountered before. 
            What is it? How do you even begin to search for it online?
            
            **RipeScan removes the guesswork.** It's your personal fruit expert, ready to give you instant clarity, right from your phone.
            """
        )

    st.write("---")

    # --- Use Case & Features ---
    st.header("Your Pocket-Sized Produce Pro")
    st.write("RipeScan is designed to be simple, fast, and fun. Here’s what you can do:")

    # Feature 1: Instant Identification
    st.subheader("1. Snap & Identify")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(
            """
            Just point your camera at any of our 10 supported fruits, and our advanced models will tell you what it is in seconds. 
            Choose between the classic **VGG16** for speed or the powerful **ResNet50** for deeper analysis.
            """
        )
    with col2:
        if os.path.exists(CAMERA_ICON_PATH):
            st.image(CAMERA_ICON_PATH, width=150)

    # Feature 2: Batch Upload
    st.subheader("2. Analyze Your Entire Haul")
    col1, col2 = st.columns([1, 2])
    with col1:
        if os.path.exists(GRID_ICON_PATH):
            st.image(GRID_ICON_PATH, width=150)
    with col2:
        st.write(
            """
            Just got back from shopping? Upload up to 10 images at once and get a full report on your fresh finds. 
            It’s perfect for cataloging, learning, or just satisfying your curiosity!
            """
        )

    # Feature 3: Smart Confidence Score
    st.subheader("3. Trustworthy Results")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(
            """
            Our app doesn't just guess; it gives you a **confidence score** with each prediction. 
            If it sees something that isn't a fruit (like a person!), it will smartly label it as "Unknown" instead of making a mistake.
            """
        )
    with col2:
        if os.path.exists(SHIELD_ICON_PATH):
            st.image(SHIELD_ICON_PATH, width=150)

    st.write("---")

    # --- Example Scenario ---
    st.header("A Day at the Market with RipeScan")
    st.write(
        """
        Imagine Priya at her local market. She spots a vibrant red fruit she thinks is a tomato but isn't sure.
        
        1.  She opens **RipeScan**.
        2.  She snaps a quick picture.
        3.  *Voilà!* The app confirms: **"Pomegranate - 99.8% Confidence."**
        
        Now she can confidently buy it, knowing exactly what she's adding to her fruit bowl. It’s that easy!
        """
    )

    st.info("Ready to try it yourself? Navigate to the **Classifier** page from the sidebar.")


if __name__ == "__main__":
    home_page()