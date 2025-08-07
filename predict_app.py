import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
from PIL import Image

# Load the trained model
with open("finger_count_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Finger Count Predictor", layout="centered")

st.title("🧠 Finger Count Prediction")
st.write("Show your hand to the camera, and we'll predict how many fingers are up!")

img_file = st.camera_input("📸 Capture hand image")

if img_file is not None:
    # Convert to numpy array (RGB)
    img = Image.open(img_file)
    img_array = np.array(img)

    # MediaPipe processing
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        result = hands.process(img_array)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]

            # Draw landmarks for display
            img_copy = img_array.copy()
            mp.solutions.drawing_utils.draw_landmarks(
                img_copy, hand, mp_hands.HAND_CONNECTIONS
            )
            st.image(img_copy, caption="Detected Hand", use_container_width=True)

            # Flatten landmarks into input vector
            coords = []
            for lm in hand.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            # Predict
            prediction = model.predict([coords])[0]
            st.success(f"🖐️ Predicted Finger Count: **{prediction}**")

        else:
            st.warning("⚠️ No hand detected in the image.")
