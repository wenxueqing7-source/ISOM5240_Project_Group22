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
# Basic App Configuration
# ============================================================

st.set_page_config(
    page_title="MoodBeat: Caption-to-Music Generator",
    page_icon="🎵",
    layout="wide",
)

# Reduce CPU thread usage on cloud deployment
torch.set_num_threads(1)

# Lightweight zero-shot mood classifier
MOOD_MODEL_NAME = "typeform/distilbert-base-uncased-mnli"

# Music generation model
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

EXAMPLE_CAPTIONS = {
    "Calm beach post": (
        "Finally spent the afternoon by the sea. "
        "The wind was soft, the waves were gentle, and everything felt peaceful."
    ),
    "Cheerful party post": (
        "Had the best night with my friends. "
        "We laughed, danced, and made memories I will never forget."
    ),
    "Energetic fitness post": (
        "Just finished an intense workout. "
        "Feeling powerful, focused, and ready for the rest of the day."
    ),
    "Romantic sunset post": (
        "Watching the sunset with someone special. "
        "The sky was golden and the moment felt timeless."
    ),
    "Mysterious city post": (
        "Walking through the city at night, surrounded by neon lights and quiet streets."
    ),
    "Melancholic rainy post": (
        "A rainy evening alone with my thoughts. "
        "Some memories feel closer when the world is quiet."
    ),
    "Cozy cafe post": (
        "A warm cup of coffee, soft lights, and a quiet corner in my favorite cafe."
    ),
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
# Cached Model Loading
# ============================================================

@st.cache_resource(show_spinner=False)
def load_mood_classifier():
    """
    Load a lightweight zero-shot classifier.
    This is used before fine-tuning is added.
    """
    classifier = pipeline(
        task="zero-shot-classification",
        model=MOOD_MODEL_NAME,
        device=PIPELINE_DEVICE,
    )
    return classifier


@st.cache_resource(show_spinner=False)
def load_musicgen():
    """
    Load facebook/musicgen-small.
    This may be slow on Streamlit Cloud.
    The model is only loaded when the user explicitly clicks audio generation.
    """
    processor = AutoProcessor.from_pretrained(MUSIC_MODEL_NAME)
    model = MusicgenForConditionalGeneration.from_pretrained(MUSIC_MODEL_NAME)
    model = model.to(DEVICE)
    model.eval()
    return processor, model


# ============================================================
# Core Logic
# ============================================================

def predict_mood(text: str):
    """
    Predict the mood of a social media caption using zero-shot classification.
    """
    classifier = load_mood_classifier()

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


def build_music_prompt(post_text: str, mood: str, platform: str) -> str:
    """
    Build an English prompt for MusicGen using the original caption and predicted mood.
    """
    mood_style = MOOD_STYLE_MAP.get(
        mood,
        "pleasant and suitable background music",
    )

    prompt = (
        f"Generate a short {mood_style} for a social media post. "
        f"The post text is: {post_text}. "
        f"The emotional mood is {mood}. "
        f"The music should be suitable for {platform}. "
        f"The music should have a clear atmosphere, a polished sound, "
        f"and no vocals."
    )

    return prompt


def generate_music(prompt: str, max_new_tokens: int = 64, guidance_scale: float = 2.0):
    """
    Generate a short audio clip using facebook/musicgen-small.
    Lower max_new_tokens is safer for Streamlit Cloud.
    """
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
            guidance_scale=guidance_scale,
        )

    generation_time = time.time() - start_time

    sampling_rate = model.config.audio_encoder.sampling_rate

    # MusicGen output shape is usually [batch, channels, samples]
    audio = audio_values[0, 0].detach().cpu().float().numpy()
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "generated_music.wav")

    wavfile.write(
        audio_path,
        rate=sampling_rate,
        data=audio,
    )

    return audio_path, generation_time


def initialize_session_state():
    """
    Store intermediate results so Streamlit reruns do not erase them.
    """
    default_values = {
        "post_text": "",
        "mood": None,
        "mood_score": None,
        "all_scores": None,
        "music_prompt": None,
        "audio_path": None,
        "generation_time": None,
    }

    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# Streamlit UI
# ============================================================

initialize_session_state()

st.title("🎵 MoodBeat: AI Caption-to-Music Generator")

st.markdown(
    """
    MoodBeat is a prototype application for generating short background music
    for social media captions.

    **Workflow:**

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

    st.divider()

    st.subheader("Audio Generation Settings")

    enable_musicgen = st.checkbox(
        "Enable MusicGen audio generation",
        value=False,
        help=(
            "MusicGen can be slow on Streamlit Cloud. "
            "Keep this off if you only want to test mood and prompt generation."
        ),
    )

    max_new_tokens = st.slider(
        "Music generation length",
        min_value=32,
        max_value=1024,
        value=64,
        step=32,
        help="Higher values create longer audio but require more time and memory.",
    )

    guidance_scale = st.slider(
        "Guidance scale",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.5,
        help="Higher values make the model follow the prompt more strongly.",
    )

    st.divider()
    st.caption(f"Device: {DEVICE}")
    st.caption(f"Mood model: {MOOD_MODEL_NAME}")
    st.caption(f"Music model: {MUSIC_MODEL_NAME}")


selected_example = st.selectbox(
    "Choose an example caption or write your own:",
    list(EXAMPLE_CAPTIONS.keys()),
)

default_text = EXAMPLE_CAPTIONS[selected_example]

post_text = st.text_area(
    "Social media caption",
    value=default_text,
    height=140,
)

col_button_1, col_button_2 = st.columns([1, 1])

with col_button_1:
    analyze_button = st.button(
        "Analyze Mood and Generate Prompt",
        type="primary",
        use_container_width=True,
    )

with col_button_2:
    clear_button = st.button(
        "Clear Results",
        use_container_width=True,
    )


if clear_button:
    for key in [
        "post_text",
        "mood",
        "mood_score",
        "all_scores",
        "music_prompt",
        "audio_path",
        "generation_time",
    ]:
        st.session_state[key] = None
    st.rerun()


if analyze_button:
    if not post_text.strip():
        st.error("Please enter a caption first.")
        st.stop()

    st.session_state.post_text = post_text

    with st.spinner("Predicting mood..."):
        mood, mood_score, all_scores = predict_mood(post_text)

    music_prompt = build_music_prompt(
        post_text=post_text,
        mood=mood,
        platform=platform,
    )

    st.session_state.mood = mood
    st.session_state.mood_score = mood_score
    st.session_state.all_scores = all_scores
    st.session_state.music_prompt = music_prompt
    st.session_state.audio_path = None
    st.session_state.generation_time = None


# ============================================================
# Display Mood and Prompt Results
# ============================================================

if st.session_state.mood is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Input Caption")
        st.write(st.session_state.post_text)

        st.subheader("2. Predicted Mood")
        st.success(f"Predicted mood: **{st.session_state.mood}**")
        st.write(f"Confidence: `{st.session_state.mood_score:.4f}`")

        st.subheader("Mood Scores")
        for label, score in st.session_state.all_scores:
            st.write(f"{label}: `{float(score):.4f}`")

    with col2:
        st.subheader("3. MusicGen Prompt")
        st.code(st.session_state.music_prompt, language="text")

        st.subheader("4. Business Use Case")
        st.write(
            f"This application can help users on **{platform}** generate "
            f"background music that matches the emotional tone of their posts. "
            f"The detected mood is **{st.session_state.mood}**, so the generated "
            f"music prompt is designed to support that atmosphere."
        )

    st.divider()

    st.subheader("5. Audio Generation")

    if not enable_musicgen:
        st.info(
            "MusicGen audio generation is currently disabled. "
            "Enable it in the sidebar if you want to generate audio. "
            "This is recommended for Streamlit Cloud stability."
        )
    else:
        generate_audio_button = st.button(
            "Generate Audio with MusicGen",
            type="primary",
            use_container_width=True,
        )

        if generate_audio_button:
            try:
                with st.spinner(
                    "Generating audio with facebook/musicgen-small. "
                    "This may take some time..."
                ):
                    audio_path, generation_time = generate_music(
                        prompt=st.session_state.music_prompt,
                        max_new_tokens=max_new_tokens,
                        guidance_scale=guidance_scale,
                    )

                st.session_state.audio_path = audio_path
                st.session_state.generation_time = generation_time

            except RuntimeError as error:
                st.error("Audio generation failed due to a runtime error.")
                st.code(str(error), language="text")
                st.warning(
                    "This is likely caused by limited CPU or memory on Streamlit Cloud. "
                    "Try reducing the music generation length to 32 tokens, or run the "
                    "MusicGen part locally or in Colab."
                )

            except Exception as error:
                st.error("Audio generation failed.")
                st.code(str(error), language="text")
                st.warning(
                    "If this happens on Streamlit Cloud, the app may not have enough "
                    "resources to run MusicGen in real time."
                )

        if st.session_state.audio_path is not None:
            st.success("Audio generated successfully.")
            st.audio(st.session_state.audio_path, format="audio/wav")
            st.write(
                f"Generation time: `{st.session_state.generation_time:.2f}` seconds"
            )

            with open(st.session_state.audio_path, "rb") as audio_file:
                st.download_button(
                    label="Download generated music",
                    data=audio_file,
                    file_name="moodbeat_generated_music.wav",
                    mime="audio/wav",
                    use_container_width=True,
                )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Prototype version: zero-shot mood classification + prompt template + "
    "facebook/musicgen-small. A fine-tuned mood classifier can be added later "
    "for the final project version."
)
