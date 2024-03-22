import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import streamlit.components.v1 as components

import speech_recognition as sr
from openai import OpenAI
import os
import requests
from streamlit_mic_recorder import mic_recorder
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS


# Assuming you have a pre-trained model for emotion detection
# Load your pre-trained model (update the path to your model)
#model = tf.keras.models.load_model('path_to_your_model')

# Set your OpenAI API key
OPENAI_API_KEY = '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'

# Initialize global variables to manage state
if 'camera_on' not in st.session_state:
    st.session_state['camera_on'] = False

if 'text_received' not in st.session_state:
    st.session_state['text_received'] = ""

def get_emotion(frame):
    # Preprocess the frame from the webcam for your model
    # This preprocessing will depend on how your model expects the input
    processed_frame = cv2.resize(frame, (48, 48))  # Example resize, adjust to your model's input size
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
    processed_frame = np.expand_dims(processed_frame, axis=0)
    processed_frame = np.expand_dims(processed_frame, axis=-1)

    # Predict the emotion on the processed frame
    prediction = model.predict(processed_frame)
    emotion = np.argmax(prediction)
    return emotion  # Or map this to an emotion label based on your model's output

def webcam_feed():
    FRAME_WINDOW = st.empty()
    camera = cv2.VideoCapture(0)  # Use default webcam

    while st.session_state['camera_on']:
        ret, frame = camera.read()
        if not ret:
            st.write("Failed to grab frame")
            break

        # Detect emotion here
        emotion = get_emotion(frame)
        # You can now use the detected emotion to display or further processing

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    camera.release()
    FRAME_WINDOW.empty()  # Clear the webcam feed

def main():
    st.title("Pocket Therapist with Live Webcam")

    # Embed the Teachable Machine model and webcam functionality in the app
    html_code = """
    <div id="webcam-container"></div>
    <div id="label-container"></div>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
    <script src="https://cdn.jsdelivr.net/npm/@teachablemachine/image"></script>
    <script type="text/javascript">
        const URL = "https://teachablemachine.withgoogle.com/models/LzEDAvy5I/";
        let model, webcam, labelContainer, maxPredictions;

        async function init() {
            const modelURL = URL + 'model.json';
            const metadataURL = URL + 'metadata.json';
            model = await tmImage.load(modelURL, metadataURL);
            maxPredictions = model.getTotalClasses();
            
            webcam = new tmImage.Webcam(200, 200, true); // width, height, flip
            await webcam.setup(); // request access to the webcam
            document.getElementById("webcam-container").appendChild(webcam.canvas);
            await webcam.play();
            window.requestAnimationFrame(loop);

            labelContainer = document.getElementById("label-container");
            for (let i = 0; i < maxPredictions; i++) {
                labelContainer.appendChild(document.createElement("div"));
            }
        }

        async function loop() {
            webcam.update(); // update the webcam frame
            await predict();
            window.requestAnimationFrame(loop);
        }

        async function predict() {
            const prediction = await model.predict(webcam.canvas);
            for (let i = 0; i < maxPredictions; i++) {
                const classPrediction = prediction[i].className + ": " + prediction[i].probability.toFixed(2);
                labelContainer.childNodes[i].innerHTML = classPrediction;
            }
        }

        init();
    </script>
    """
    components.html(html_code, height=600)

    #st.title("Pocket Therapist with Live Webcam")

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