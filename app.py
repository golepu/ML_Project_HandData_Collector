import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Hand Data Collector", layout="centered")

st.title("🖐️ Hand Landmark Data Collector")
st.write("Take a snapshot of your hand and label how many fingers are up. Then download the collected data as a CSV.")

data_file = "hand_data.csv"

# Dropdown for labeling
label = st.selectbox("How many fingers are up?", options=list(range(6)), index=0)

# Camera input
img_file = st.camera_input("📸 Take a picture of your hand")

if img_file is not None:
    img = Image.open(img_file)
    img_array = np.array(img)  # Already RGB

    # Run MediaPipe
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        result = hands.process(img_array)

        if result.multi_hand_landmarks:
            st.success("✅ Hand detected!")
            hand = result.multi_hand_landmarks[0]

            # Show the image with landmarks
            img_copy = img_array.copy()
            mp.solutions.drawing_utils.draw_landmarks(
                img_copy, hand, mp_hands.HAND_CONNECTIONS)
            st.image(img_copy, caption="Detected Hand", use_column_width=True)

            # Flatten and prepare data
            coords = []
            for lm in hand.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            # Save to CSV
            if st.button("💾 Save This Sample"):
                row = coords + [label]
                df = pd.DataFrame([row])
                try:
                    existing = pd.read_csv(data_file, header=None)
                    df = pd.concat([existing, df], ignore_index=True)
                except FileNotFoundError:
                    pass
                df.to_csv(data_file, index=False, header=False)
                st.success("Sample saved!")
        else:
            st.warning("⚠️ No hand detected in the image.")

# CSV Download
if st.button("📥 Download All Collected Data"):
    try:
        with open(data_file, "rb") as f:
            st.download_button("⬇️ Download hand_data.csv", f, "hand_data.csv")
    except FileNotFoundError:
        st.info("No data collected yet.")
