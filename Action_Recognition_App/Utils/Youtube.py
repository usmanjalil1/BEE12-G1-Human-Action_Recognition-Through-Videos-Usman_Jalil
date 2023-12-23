import os
import cv2
import pafy
import math
import random
import numpy as np
import time
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
from moviepy.editor import *
from sklearn.model_selection import train_test_split
from pytube import YouTube
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
import streamlit as st
from tqdm import tqdm
IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20

CLASSES_LIST = ['BaseballPitch',
 'Basketball',
 'BenchPress',
 'Biking',
 'Billiards',
 'BreastStroke',
 'CleanAndJerk',
 'Diving',
 'Drumming',
 'Fencing']

from pytube import YouTube
import streamlit as st

def download_video(url, download_path):
    try:
        yt = YouTube(url)

        # Get the highest resolution stream
        video_stream = yt.streams.get_highest_resolution()
        video_title = video_stream.title

        # Download the video
        st.text("Downloading...")
        video_stream.download(download_path)
        st.success(f"Download complete! Video title: {video_title}")

        return video_title

    except Exception as e:
        st.error(f"Error: {str(e)}")



def predict_single_action(video_file_path, model):
    '''
    This function will perform single action recognition prediction on a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''
    video_reader = cv2.VideoCapture(video_file_path)
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_list = []
    predicted_class_name = ''
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)
    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read() 
        if not success:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_list.append(normalized_frame)
    predicted_labels_probabilities = model.predict(np.expand_dims(frames_list, axis = 0))[0]
    predicted_label = np.argmax(predicted_labels_probabilities)
    predicted_class_name = CLASSES_LIST[predicted_label]
    # st.text("Prediction Result:")
    # st.write(predicted_class_name)
    # st.write(f'Confidence: {predicted_labels_probabilities[predicted_label]}')
    result_placeholder = st.empty()

    
    # Function to simulate a prediction
    def simulate_prediction():
    # Simulating a prediction delay
        time.sleep(3)
        return predicted_class_name
   

    # Use spinner for animation
    with st.spinner("Predicting..."):
        prediction_result = simulate_prediction()
        result_placeholder.success("Prediction complete!")
        time.sleep(2)  # Add a short delay to showcase the success message

    # Display the actual prediction result
    result_placeholder.write(prediction_result)
    print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')
    video_reader.release()