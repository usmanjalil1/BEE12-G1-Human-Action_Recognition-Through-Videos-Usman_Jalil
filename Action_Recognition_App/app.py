import streamlit as st
from Utils.Youtube import download_video, predict_single_action
from tensorflow.keras.models import load_model
import os
st.title("Action Recognition in Videos")
CLASSES_LIST = ['BaseballPitch', 'Basketball', 'BenchPress', 'Biking', 'Billiards', 'BreastStroke', 'CleanAndJerk', 'Diving', 'Drumming', 'Fencing']
st.markdown("<p style='color:red'>Note: Our classifier is limited to these classes due to hardware constraints:</p>", unsafe_allow_html=True)
st.write(CLASSES_LIST)
video_url = st.text_input("Enter YouTube Video URL:")
if st.button("Download"):
    if not video_url:
        st.warning("Please enter a valid YouTube video URL.")
    else:
        test_videos_directory = 'Action_Recognition_App/youtube_videos'
        os.makedirs(test_videos_directory, exist_ok = True)
        video_title = download_video(video_url, test_videos_directory)
        input_video_file_path = f'{test_videos_directory}/{video_title}.mp4'

        model = load_model('/Users/mac/Desktop/CV_Project/convlstm_model___Date_Time_2023_12_23__02_58_07___Loss_0.38356760144233704___Accuracy_0.8710691928863525.h5')
        predict_single_action(input_video_file_path, model)

        

