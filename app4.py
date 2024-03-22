import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
from openai import OpenAI

# Streamlit page configuration and theme customization
# Streamlit page configuration and theme customization
st.set_page_config(page_title="Pocket Therapist", layout="wide", initial_sidebar_state="collapsed")
primary_color = "#4E2A84"  # A purple shade
background_color = "#EDEAFD"  # A light purple shade
secondary_background_color = "#CAB8FF"  # A softer purple


st.markdown("""
    <style>
        :root {
            --primary-color: #0A75BC;
            --bg-color: #F0F2F6;
            --text-color: #333333;
            --font: "Helvetica Neue", Helvetica, Arial, sans-serif;
        }
        .css-18e3th9 {
            background-color: var(--bg-color);
        }
        .stButton>button {
            color: var(--text-color);
            border-radius: 8px;
            border: 1px solid var(--primary-color);
        }
        .st-bb {
            background-color: var(--primary-color);
        }
        .css-1d391kg {
            background-color: var(--primary-color);
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize global variables to manage state
if 'camera_on' not in st.session_state:
    st.session_state['camera_on'] = False

if 'text_received' not in st.session_state:
    st.session_state['text_received'] = ""

# Load your TensorFlow model
model_path = '/Users/user/Desktop/MoodMender/converted_savedmodel/model.savedmodel'
model = tf.keras.models.load_model(model_path)

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def get_emotion(frame, model):
    processed_frame = preprocess_image(frame)
    prediction = model.predict(processed_frame)
    emotion = np.argmax(prediction)
    return emotion  # Or map this to an emotion label

def webcam_feed():
    FRAME_WINDOW = st.empty()
    camera = cv2.VideoCapture(0)

    while st.session_state['camera_on']:
        ret, frame = camera.read()
        if not ret:
            st.write("Failed to grab frame")
            break

        emotion = get_emotion(frame, model)
        st.write("emotion is = ",emotion)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    camera.release()
    FRAME_WINDOW.empty()

def main():
    st.title("Pocket Therapist with Live Webcam")

    with st.sidebar:
        st.write("## Controls")
        if st.button('Toggle Camera'):
            st.session_state['camera_on'] = not st.session_state['camera_on']

    if st.session_state['camera_on']:
        webcam_feed()

    st.write("## Convert speech to text:")
    text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')

    if text:
        st.session_state['text_received'] = text

    if st.session_state['text_received']:
        print("text_received")
        st.text(st.session_state['text_received'])

        # Replace 'your_openai_api_key' with your actual OpenAI API key
        OPENAI_API_KEY = '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'
        client = OpenAI(api_key=OPENAI_API_KEY)
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a trained psychotherapist, specializing in providing stress management strategies."},
                {"role": "user", "content": st.session_state['text_received']}
            ]
        )
        response_text = completion.choices[0].message.content
        st.write(response_text)

        tts = gTTS(response_text, lang='en')
        tts.save('response.mp3')
        st.audio('response.mp3')

if __name__ == "__main__":
    main()
