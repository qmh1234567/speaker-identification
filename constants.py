#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# __author__: Qmh
# __file_name__: constants.py
# __time__: 2019:06:27:16:18

# MEL-SPECTROGRAM
SR = 16000
DURA = 3  # 3s
N_FFT = 512
N_MELS = 128  
HOP_LEN = 161
SAMPLE_LENGTH = 59049
# SIGNAL PROCESSING
PREEMPHASIS_ALPHA = 0.97
FRAME_LEN = 0.025  
FRAME_STEP = 0.01  

# DATASET
TRAIN_DEV_SET = '/home/dsp/Documents/wav_data/SpeakerRecognition_dataset/AIShell/trainset/'
# PICKLE_TRAIN_DIR = '/home/dsp/Documents/wav_data/SpeakerRecognition_dataset/Library_speech_100/train-clean-100/LibriSpeech/train-clean-100/pickle'
# PICKLE_TRAIN_DIR = '/home/dsp/Documents/wav_data/SpeakerRecognition_dataset/TIMIT/lisa/data/timit/raw/TIMIT/TRAIN/pickle'


TEST_SET = '/home/range/Documents/SI_dataset/'
# TEST_SET_TIMIT = '/home/dsp/Documents/wav_data/SpeakerRecognition_dataset/TIMIT/lisa/data/timit/raw/TIMIT/TEST'
# PICKLE_TEST_DIR = '/home/dsp/Documents/wav_data/SpeakerRecognition_dataset/Library_speech_100/test-clean/LibriSpeech/test-clean/pickle'
# PICKLE_TEST_DIR = '/home/dsp/Documents/wav_data/SpeakerRecognition_dataset/TIMIT/lisa/data/timit/raw/TIMIT/TEST/pickle'


TARGET = 'SI'
ENROLL_NUMBER = 20

# CLASS = 251


DATASET_DIR = './dataset'

# MODEL
WEIGHT_DECAY = 0.00001 #0.00001

REDUCTION_RATIO = 8 
BLOCK_NUM = 2
DROPOUT= 0.1
ALPHA = 0.1

# TRAIN
BATCH_SIZE = 32
INPUT_SHPE = (299,40,1)
# INPUT_SHPE = (512,299,1)
TRAIN_AUDIO_NUM = 100


MODEL_DIR ='./checkpoint'
LEARN_RATE = 0.01  # 0.01
LEARN_DECAY = 0.1
MIN_LEARN_RATE = 0.00001
STAGE = 5
NPY_FILE_NAME = 'npy_fb'

#TEST
ANNONATION_FILE = './dataset/annonation.csv'
TEST_TEXT_SV = './dataset/SV_dataset.txt'
TEST_TEXT_SI = './dataset/SI_dataset.txt'