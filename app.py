import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import librosa
import os
from os import path

import json
import numpy as np
import pathlib
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import initializers, regularizers
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam
from AppReadme import markdown_text

import streamlit as st


def load_wav(x):
    '''This return the array values of audio with sampling rate of 22050 and Duration'''
    #loading the wav file with sampling rate of 22050
    samples, sample_rate = librosa.load(x, sr=22050)    # sample_rate = 22050
    return samples
        

# Creating Padding
def pad_sequence(data, max_length=17640):
    truncated = data[:max_length]
    pad_this_much = max_length - len(truncated)
    X_pad_seq = np.pad(truncated, ((0,pad_this_much)), mode='constant', constant_values=0)
    X_pad_seq = np.array(X_pad_seq)
    return X_pad_seq


def convert_to_spectrogram(raw_data):
    '''converting to spectrogram'''
    spectrum = librosa.feature.melspectrogram(y=raw_data, sr=22050, n_mels=64)
    logmel_spectrum = librosa.power_to_db(S=spectrum, ref=np.max)
    return logmel_spectrum
    
    
def get_spectogram_data(X_pad_seq):
    X_spectrogram = []
    spectogram_data = convert_to_spectrogram(X_pad_seq)
    return np.array(spectogram_data)
    
    
def preprocess_audio(path): # E.g. path = recordings/5_theo_28.wav
    s_d = load_wav(path)
    X_pad_seq = pad_sequence(s_d)
    X_spectrogram = get_spectogram_data(X_pad_seq)
    return X_spectrogram


def f1_score_micro(y_true, y_pred):
    return f1_score(y_true, y_pred, average = 'micro')
    
def micro_f1(y_true, y_proba):
    y_pred = tf.math.argmax(y_proba,axis=1)
    return tf.py_function(f1_score_micro, (y_true,y_pred), tf.float32)
    
    
@st.cache
def load_audio_model():
    best_models_path = "model/"
    best_model = tf.keras.models.load_model(best_models_path+'BEST_MODEL.hdfs', custom_objects={'f1_score_micro':f1_score_micro,'micro_f1':micro_f1})
    return best_model


def main():
    st.sidebar.title("Welcome!")
    selection = st.sidebar.selectbox("Select an option", ["Readme","Use default audio","Upload your audio"])

    model_loading_state = st.text('Getting things ready! Please wait...')
    model = load_audio_model()
    model.summary()
    model_loading_state.text('The AI is ready!')
    
    if selection=="Readme":
        st.markdown(markdown_text)
    elif selection=="Use default audio":
        st.title("Spoken Digit Recognition using AI")
        st.subheader("Please choose an audio using the buttons below")
        radio = st.radio(label="", options=["Audio 1", "Audio 2", "Audio 3", "Audio 4"])
        processing_text= st.empty()
        if radio == "Audio 1":
            default_audio = "default_audio/Audio_1.wav"
            audio_file = open(default_audio, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/ogg')
            
            processing_text.text("Processing the audio...")
            input_audio = preprocess_audio(default_audio)
            input_audio = np.expand_dims(input_audio, axis=0)
            print("input_audio =",input_audio.shape)
            processing_text.text("Recognizing the audio...")
            predicted_num = model.predict(input_audio)[0]
            processing_text.text("")
            st.success("You spoke: "+str(np.argmax(predicted_num)))
        elif radio == "Audio 2":
            default_audio = "default_audio/Audio_2.wav"
            audio_file = open(default_audio, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/ogg')
            
            processing_text.text("Processing the audio...")
            input_audio = preprocess_audio(default_audio)
            input_audio = np.expand_dims(input_audio, axis=0)
            print("input_audio =",input_audio.shape)
            processing_text.text("Recognizing the audio...")
            predicted_num = model.predict(input_audio)[0]
            processing_text.text("")
            st.success("You spoke: "+str(np.argmax(predicted_num)))
        elif radio == "Audio 3":
            default_audio = "default_audio/Audio_3.wav"
            audio_file = open(default_audio, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/ogg')
            
            processing_text.text("Processing the audio...")
            input_audio = preprocess_audio(default_audio)
            input_audio = np.expand_dims(input_audio, axis=0)
            print("input_audio =",input_audio.shape)
            processing_text.text("Recognizing the audio...")
            predicted_num = model.predict(input_audio)[0]
            processing_text.text("")
            st.success("You spoke: "+str(np.argmax(predicted_num)))
        elif radio == "Audio 4":
            default_audio= "default_audio/Audio_4.wav"
            audio_file = open(default_audio, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/ogg')
            
            processing_text.text("Processing the audio...")
            input_audio = preprocess_audio(default_audio)
            input_audio = np.expand_dims(input_audio, axis=0)
            print("input_audio =",input_audio.shape)
            processing_text.text("Recognizing the audio...")
            predicted_num = model.predict(input_audio)[0]
            processing_text.text("")
            st.success("You spoke: "+str(np.argmax(predicted_num)))
            
    elif selection=="Upload your audio":
        st.title("Spoken Digit Recognition using AI")
        uploaded_file = st.file_uploader("Upload audio with .wav extension...", type=["wav"])
        upload_button = st.empty()
        
        processing_text= st.empty()
        if uploaded_file is not None:
            #uploaded_file = uploaded_file.name
            #audio_file = open(uploaded_file, 'rb')
            #audio_bytes = audio_file.read()
            #st.audio(audio_bytes, format='audio/ogg')
            st.audio(uploaded_file, format='audio/ogg')   ###
            
            processing_text.text("Processing the audio...")
            input_audio = preprocess_audio(uploaded_file)
            input_audio = np.expand_dims(input_audio, axis=0)
            print("input_audio =",input_audio.shape)
            processing_text.text("Recognizing the audio...")
            predicted_num = model.predict(input_audio)[0]
            processing_text.text("")
            st.success("You spoke: "+str(np.argmax(predicted_num)))
            upload_button.text("")
    


if __name__ == '__main__':
    main()