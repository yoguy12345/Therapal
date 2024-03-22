import cv2
import streamlit as st
import numpy as np
import speech_recognition as sr
from openai import OpenAI
import os
import requests
from streamlit_mic_recorder import mic_recorder
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS

import tensorflow as tf
import streamlit.components.v1 as components

import speech_recognition as sr


print("Current working directory:", os.getcwd())


model_path = '/Users/user/Desktop/MoodMender/converted_savedmodel/model.savedmodel'
model =  tf.keras.models.load_model(model_path)




# Set your OpenAI API key
OPENAI_API_KEY = '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'

# Initialize global variables to manage state
if 'camera_on' not in st.session_state:
    st.session_state['camera_on'] = False

if 'text_received' not in st.session_state:
    st.session_state['text_received'] = ""

def preprocess_image(image):
    if len(image.shape) == 2:
        image = tf.expand_dims(image, axis=-1)
    image = tf.image.resize(image, [224, 224])
    if image.shape[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)

    # Add a batch dimension
    image = tf.expand_dims(image, axis=0)
    return image



def get_emotion(frame):
    # Preprocess the frame from the webcam for your model
    # This preprocessing will depend on how your model expects the input
    processed_frame = cv2.resize(frame, (48, 48))  # Example resize, adjust to your model's input size
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
    processed_frame = np.expand_dims(processed_frame, axis=0)
    processed_frame = np.expand_dims(processed_frame, axis=-1)

    processed_frame = preprocess_image(frame)

    # Predict the emotion on the processed frame
    prediction = model.predict(processed_frame)
    st.write(prediction)

    emotion = np.argmax(prediction)
    return emotion  # Or map this to an emotion label based on your model's output

# Function to display and handle the webcam feed
def webcam_feed():
    FRAME_WINDOW = st.empty()
    camera = cv2.VideoCapture(0)  # Use default webcam

    while st.session_state['camera_on']:
        ret, frame = camera.read()
        if not ret:
            st.write("Failed to grab frame")
            break
        val = get_emotion(frame)
        #print(val)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    camera.release()
    FRAME_WINDOW.empty()  # Clear the webcam feed

def main():
    st.title("Pocket Therapist with Live Webcam")

        

    # Webcam control
    if st.button('Toggle Camera'):
        st.session_state['camera_on'] = not st.session_state['camera_on']
        if st.session_state['camera_on']:
            webcam_feed()  # Start the webcam feed

    # Speech-to-text conversion
    st.write("Convert speech to text:")
    text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')

    if text:
        st.session_state.text_received = text

    if st.session_state.text_received:
        # Display the transcribed text
        st.text(st.session_state.text_received)

        # OpenAI API interaction
        client = OpenAI(api_key=OPENAI_API_KEY)
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a trained psychotherapist, specializing in providing stress management strategies for people with ADHD. Give short responses for every query, less than 4 sentences."},
                {"role": "user", "content": st.session_state.text_received}
            ]
        )
        response_text = completion.choices[0].message.content
        st.write(response_text)  # Display the response

        # Convert the response text to speech and play it
        tts = gTTS(response_text, lang='en')
        tts.save('response.mp3')
        st.audio('response.mp3')

if __name__ == "__main__":
    main()