import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image, ImageOps
from transformers import CLIPProcessor, CLIPModel
import tempfile
import os
import dotenv


# Load environment variables
dotenv.load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")


# Load CLIP model using token
MODEL_NAME = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)


# Function to extract frames
def extract_frames(video_path, frame_skip=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()
        if not ret:
            break
        
        if int(frame_pos) % frame_skip == 0:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append((cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, image))  # Timestamp in seconds
    
    cap.release()
    return frames

# Function to encode frames
def encode_frames(frames):
    """
    Encode a list of video frames into CLIP embeddings.

    Args:
        frames (list): A list of tuples where each tuple contains a timestamp (in seconds)
                       and a PIL Image object representing a video frame.

    Returns:
        list: A list of tuples where each tuple contains a timestamp and a normalized
              CLIP embedding tensor for the corresponding frame.
    """
    timestamps, images = zip(*frames)
    resized_images = [ImageOps.fit(image, (224, 224)) for image in images]  # Resize for CLIP

    inputs = processor(images=resized_images, return_tensors="pt", padding=True)

    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

    return list(zip(timestamps, embeddings))

# Function to encode text query
def encode_text(query):
    inputs = processor(text=[query], return_tensors="pt", padding=True)

    with torch.no_grad():
        text_emb = model.get_text_features(**inputs)
        text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)

    return text_emb.squeeze()

# Function to find the best matching frame
def find_best_frame(frame_embeddings, text_embedding):
    similarities = [(ts, torch.cosine_similarity(embed, text_embedding, dim=0).item()) for ts, embed in frame_embeddings]

    if not similarities:
        return None, None

    best_timestamp, best_score = max(similarities, key=lambda x: x[1])
    return best_timestamp, best_score

# Streamlit UI
st.title("🔍 Video Search with CLIP")
st.write("Upload a video and enter a query to find the best matching frame.")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
query = st.text_input("Enter your search query:")

if uploaded_file and query:
    # Save the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        video_path = temp_video.name

    st.write("### Extracting frames from video...")
    frames = extract_frames(video_path)

    st.write("### Encoding frames...")
    frame_embeddings = encode_frames(frames)

    st.write("### Encoding your query...")
    text_embedding = encode_text(query)

    st.write("### Searching best match...")
    best_ts, score = find_best_frame(frame_embeddings, text_embedding)

    if best_ts is not None:
        st.success(f"Best match found at {best_ts:.2f} seconds (Score: {score:.4f})")

        # Find the corresponding frame image
        matched_frame = next((img for ts, img in frames if ts == best_ts), None)
        if matched_frame:
            st.image(matched_frame, caption=f"Matched Frame at {best_ts:.2f} sec", use_column_width=True)
    else:
        st.error("No matching frame found.")

    # Cleanup temp video file
    os.remove(video_path)
