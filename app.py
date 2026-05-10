import os
import time
import tempfile
import numpy as np
import torch
import streamlit as st
from scipy.io import wavfile
from transformers import (
    pipeline,
    AutoProcessor,
    MusicgenForConditionalGeneration,
)

# ============================================================
# App Configuration
# ============================================================

st.set_page_config(
    page_title="MoodBeat: Caption-to-Music Generator",
    page_icon="🎵",
    layout="wide",
)

MOOD_MODEL_NAME = "facebook/bart-large-mnli"
MUSIC_MODEL_NAME = "facebook/musicgen-small"

MOOD_LABELS = [
    "calm",
    "cheerful",
    "energetic",
    "romantic",
    "mysterious",
    "melancholic",
    "cozy",
]

MOOD_STYLE_MAP = {
    "calm": "soft, peaceful, relaxing acoustic background music",
    "cheerful": "happy, bright, playful pop background music",
    "energetic": "upbeat, fast, energetic electronic background music",
    "romantic": "soft, warm, emotional romantic acoustic music",
    "mysterious": "dark, cinematic, atmospheric synth background music",
    "melancholic": "slow, emotional, soft piano background music",
    "cozy": "warm, cozy, lo-fi jazz background music",
}


# ============================================================
# Device Setup
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return "cuda", 0
    return "cpu", -1


DEVICE, PIPELINE_DEVICE = get_device()


# ============================================================
# Model Loading
# ============================================================

@st.cache_resource
def load_mood_classifier():
    classifier = pipeline(
        task="zero-shot-classification",
        model=MOOD_MODEL_NAME,
        device=PIPELINE_DEVICE,
    )
    return classifier


@st.cache_resource
def load_musicgen():
    processor = AutoProcessor.from_pretrained(MUSIC_MODEL_NAME)
    model = MusicgenForConditionalGeneration.from_pretrained(MUSIC_MODEL_NAME)
    model = model.to(DEVICE)
    model.eval()
    return processor, model


# ============================================================
# Core Functions
# ============================================================

def predict_mood(text: str, classifier):
    result = classifier(
        sequences=text,
        candidate_labels=MOOD_LABELS,
        hypothesis_template="This social media post has a {} mood.",
        multi_label=False,
    )

    top_mood = result["labels"][0]
    top_score = float(result["scores"][0])
    all_scores = list(zip(result["labels"], result["scores"]))

    return top_mood, top_score, all_scores


def build_music_prompt(post_text: str, mood: str, platform: str):
    mood_style = MOOD_STYLE_MAP.get(
        mood,
        "pleasant and suitable background music",
    )

    prompt = (
        f"Generate a short {mood_style} for a social media post. "
        f"The post text is: {post_text}. "
        f"The emotional mood is {mood}. "
        f"The music should be suitable for {platform}, "
        f"with a clear atmosphere, a polished sound, and no vocals."
    )

    return prompt


def generate_music(prompt: str, max_new_tokens: int):
    processor, model = load_musicgen()

    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    start_time = time.time()

    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            guidance_scale=3.0,
        )

    generation_time = time.time() - start_time
    sampling_rate = model.config.audio_encoder.sampling_rate

    audio = audio_values[0].detach().cpu().float().numpy()

    if audio.ndim == 2:
        audio = audio.T

    audio = np.clip(audio, -1.0, 1.0)

    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "generated_music.wav")

    wavfile.write(
        audio_path,
        rate=sampling_rate,
        data=audio,
    )

    return audio_path, generation_time


# ============================================================
# Streamlit UI
# ============================================================

st.title("🎵 MoodBeat: AI Caption-to-Music Generator")

st.markdown(
    """
    MoodBeat is a prototype application that generates short background music for social media posts.

    Workflow:

    **Social Media Caption → Mood Classification → Music Prompt → MusicGen Audio**
    """
)

with st.sidebar:
    st.header("Settings")

    platform = st.selectbox(
        "Target platform",
        [
            "TikTok",
            "Instagram",
            "WeChat Moments",
            "Xiaohongshu",
            "General social media",
        ],
    )

    max_new_tokens = st.slider(
        "Music generation length",
        min_value=128,
        max_value=512,
        value=256,
        step=64,
        help="A higher value usually creates longer music, but generation becomes slower.",
    )

    generate_audio = st.checkbox(
        "Generate audio with MusicGen",
        value=True,
        help="Turn this off if you only want to test mood detection and prompt generation.",
    )

    st.caption(f"Current device: {DEVICE}")


example_captions = {
    "Calm beach post": "Finally spent the afternoon by the sea. The wind was soft, the waves were gentle, and everything felt peaceful.",
    "Cheerful party post": "Had the best night with my friends. We laughed, danced, and made memories I will never forget.",
    "Energetic fitness post": "Just finished an intense workout. Feeling powerful, focused, and ready for the rest of the day.",
    "Romantic sunset post": "Watching the sunset with someone special. The sky was golden and the moment felt timeless.",
    "Mysterious city post": "Walking through the city at night, surrounded by neon lights and quiet streets.",
    "Melancholic rainy post": "A rainy evening alone with my thoughts. Some memories feel closer when the world is quiet.",
    "Cozy cafe post": "A warm cup of coffee, soft lights, and a quiet corner in my favorite cafe.",
}

selected_example = st.selectbox(
    "Choose an example caption or write your own:",
    list(example_captions.keys()),
)

default_text = example_captions[selected_example]

post_text = st.text_area(
    "Social media caption",
    value=default_text,
    height=140,
)

run_button = st.button("Generate Music Prompt and Audio", type="primary")


if run_button:
    if not post_text.strip():
        st.error("Please enter a caption first.")
        st.stop()

    with st.spinner("Predicting mood..."):
        mood_classifier = load_mood_classifier()
        mood, mood_score, all_scores = predict_mood(post_text, mood_classifier)

    music_prompt = build_music_prompt(post_text, mood, platform)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Input Caption")
        st.write(post_text)

        st.subheader("2. Predicted Mood")
        st.success(f"Predicted mood: **{mood}**")
        st.write(f"Confidence: `{mood_score:.4f}`")

        st.subheader("Mood Scores")
        for label, score in all_scores:
            st.write(f"{label}: `{score:.4f}`")

    with col2:
        st.subheader("3. MusicGen Prompt")
        st.code(music_prompt, language="text")

        st.subheader("4. Business Use Case")
        st.write(
            f"This generated music is designed for **{platform}** posts. "
            f"It uses the detected **{mood}** mood to create background music that matches the user's caption."
        )

    if generate_audio:
        with st.spinner("Generating music with facebook/musicgen-small. This may take some time..."):
            audio_path, generation_time = generate_music(
                prompt=music_prompt,
                max_new_tokens=max_new_tokens,
            )

        st.subheader("5. Generated Audio")
        st.audio(audio_path, format="audio/wav")
        st.write(f"Generation time: `{generation_time:.2f}` seconds")

        with open(audio_path, "rb") as audio_file:
            st.download_button(
                label="Download generated music",
                data=audio_file,
                file_name="moodbeat_generated_music.wav",
                mime="audio/wav",
            )
    else:
        st.info("Audio generation is turned off. Only mood and prompt were generated.")
