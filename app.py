import streamlit as st
import cv2
import torch
import numpy as np
import tempfile
import os
from transformers import TimesformerForVideoClassification, AutoImageProcessor
from groq import Groq

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VisionAct AI",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 VisionAct AI")
st.markdown("#### Video Action Recognition + AI Description")
st.markdown("Upload a video and get a detailed AI-powered description of what's happening!")
st.divider()

# ─── Constants ─────────────────────────────────────────────────────────────────
HF_REPO    = "yash2024/visionact-timesformer-ucf101"
NUM_FRAMES = 8
IMAGE_SIZE = 224
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Load Model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(HF_REPO)
    model     = TimesformerForVideoClassification.from_pretrained(HF_REPO)
    model     = model.to(DEVICE)
    model.eval()
    return model, processor

# ─── Frame Extraction ──────────────────────────────────────────────────────────
def extract_frames(video_path, num_frames=NUM_FRAMES):
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total == 0:
        cap.release()
        return [np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)] * num_frames

    indices = np.linspace(0, total - 1, num_frames).astype(int)
    frames  = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
            frames.append(frame)
        else:
            frames.append(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))

    cap.release()
    while len(frames) < num_frames:
        frames.append(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))
    return frames

# ─── Predict Action ────────────────────────────────────────────────────────────
def predict_action(video_path, model, processor):
    frames  = extract_frames(video_path)
    inputs  = processor(images=frames, return_tensors="pt")
    inputs  = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=-1)
        top5    = torch.topk(probs, 5)

    id2label = model.config.id2label
    results  = [
        {"action": id2label[idx.item()], "confidence": score.item()}
        for score, idx in zip(top5.values[0], top5.indices[0])
    ]
    return results

# ─── Generate Description ──────────────────────────────────────────────────────
def generate_description(top_predictions):
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
    if not GROQ_API_KEY:
        return "⚠️ GROQ_API_KEY not set in Streamlit secrets."

    client = Groq(api_key=GROQ_API_KEY)

    top1 = top_predictions[0]["action"]
    conf = top_predictions[0]["confidence"] * 100
    top3 = ", ".join([f'{p["action"]} ({p["confidence"]*100:.1f}%)' for p in top_predictions[:3]])

    prompt = f"""You are an expert video analyst. A video has been analyzed and the AI model detected the following actions:

Top prediction: {top1} (confidence: {conf:.1f}%)
Top 3 predictions: {top3}

Based on this, write a detailed, vivid, and accurate description of what is likely happening in the video. Include:
- What action is being performed
- How the person is performing it (body movements, technique, posture)
- The likely environment or setting
- Any equipment or objects involved
- The intensity or style of the action

Write 3-4 sentences in a natural, descriptive tone. Do not mention confidence scores or AI in your description."""

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

# ─── Main App ──────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a video file",
    type=["avi", "mp4", "mov", "mkv"],
    help="Supported formats: .avi, .mp4, .mov, .mkv"
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.video(tmp_path)
    st.divider()

    with st.spinner("🔄 Loading model..."):
        model, processor = load_model()

    with st.spinner("🧠 Analyzing video..."):
        predictions = predict_action(tmp_path, model, processor)

    top_action = predictions[0]["action"]
    top_conf   = predictions[0]["confidence"] * 100

    st.markdown(f"### 🎯 Detected Action: `{top_action}`")
    st.progress(int(top_conf), text=f"{top_conf:.1f}% confident")
    st.divider()

    st.markdown("#### 📊 Top-5 Predictions")
    for i, pred in enumerate(predictions):
        col1, col2 = st.columns([3, 7])
        with col1:
            st.markdown(f"**{i+1}. {pred['action']}**")
        with col2:
            st.progress(int(pred['confidence'] * 100), text=f"{pred['confidence']*100:.1f}%")

    st.divider()

    st.markdown("#### 📝 AI Description")
    with st.spinner("✍️ Generating description..."):
        description = generate_description(predictions)
    st.info(description)

    with st.expander("🎞️ View Extracted Frames"):
        frames = extract_frames(tmp_path)
        cols   = st.columns(4)
        for i, frame in enumerate(frames):
            with cols[i % 4]:
                st.image(frame, caption=f"Frame {i+1}", use_column_width=True)

    os.unlink(tmp_path)

else:
    st.markdown("""
    ### How it works:
    1. 📤 **Upload** any action video
    2. 🧠 **TimeSformer AI** analyzes the video
    3. 📝 **Llama-3 70B** generates a detailed description
    4. 🎯 Get **top-5 predictions** with confidence scores

    > Model trained on **UCF-101** — 101 action classes, **92.84% accuracy**
    """)
