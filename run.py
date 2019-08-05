#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# __author__: Qmh
# __file_name__: run.py
# __time__: 2019:06:27:20:53

import pandas as pd
import constants as c
import os
import tensorflow as tf
from collections import Counter
import numpy as np
from progress.bar import Bar
import models
from keras.layers import Dense
from keras import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import sys
import keras.backend.tensorflow_backend as KTF
from keras.layers import Flatten
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedKFold
from keras import optimizers
import glob
import pickle
import seaborn as sns
import argparse

# OPTIONAL: control usage of GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
sess = tf.Session(config=config)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enroll_dataset",required=True,help="which dataset to enroll with")
    parser.add_argument("--test_dataset",required=True,help="which dataset to test with")
    parser.add_argument("--mode_type",required=True,choices=['train','test'],help='train or test')
    return parser.parse_args()

def shuffle_data(paths, labels):
    length = len(labels)
    shuffle_index = np.arange(0, length)
    shuffle_index = np.random.choice(shuffle_index, size=length, replace=False)
    paths = np.array(paths)[shuffle_index]
    labels = np.array(labels)[shuffle_index]
    return paths, labels


# 标准化
def normalization_frames(m,epsilon=1e-12):
    return np.array([(v-np.mean(v))/max(np.std(v),epsilon) for v in m ])


def CreateDataset(args, split_ratio=0.2, target='SV'):
    typeName = args.mode_type
    train_paths, val_paths = [], []
    train_labels, val_labels = [], []
    seed = 42
    np.random.seed(seed)
    # shuffle
    if typeName.startswith('train'):
        old_audio_paths = [pickle for pickle in glob.iglob(c.TRAIN_DEV_SET + "/pickle/*.pickle")]
        old_audio_paths.sort()
        current_speaker = ""
        current_count = 0
        audio_paths = []
        for pickle in old_audio_paths:
            speaker = os.path.basename(pickle).split('_')[0]
            if current_speaker != speaker:
                current_count = 0
                current_speaker = speaker
            elif current_count < c.TRAIN_AUDIO_NUM:
                audio_paths.append(pickle)
                current_count += 1

        audio_labels = [os.path.basename(pickle).split("_")[0] for pickle in audio_paths]

        train_paths, val_paths, train_labels, val_labels = train_test_split(
            audio_paths, audio_labels, stratify=audio_labels, test_size=split_ratio, random_state=42)
    else:
        df = pd.read_csv('./dataset/annotation.csv')
        audio_labels = list(df['SpeakerID'])

        audio_paths = []
        for f in list(df['FileID']):
            file_name ="pickle_" + f +'.pickle'
            audio_paths.append(os.path.join(args.test_dataset+'/pickle',file_name))

        df = pd.read_csv('./dataset/enrollment.csv')
        val_labels = list(df['SpeakerID'])
        val_paths = []
        for f in list(df['FileID']):
            file_name = "pickle_"+ f +'.pickle'
            val_paths.append(os.path.join(args.enroll_dataset+'/pickle',file_name))
    
        train_paths,train_labels = [],[]
        for index,x in enumerate(audio_paths):
            if x not in val_paths:
                train_paths.append(x)
                train_labels.append(audio_labels[index])
        

    train_dataset = (train_paths, train_labels)
    val_dataset = (val_paths, val_labels)
    print("len(train_paths)=", len(train_paths))
    print("len(val_paths)=", len(val_paths))
    # print("len(audio_paths)=", len(audio_labels))
    print("len(set(train_labels))=", len(set(train_labels)))

    return train_dataset,val_dataset


def Map_label_to_dict(labels):
    labels_to_id = {}
    i = 0
    for label in np.unique(labels):
        labels_to_id[label] = i
        i += 1
    return labels_to_id


def load_validation_data(dataset, labels_to_id, num_class):
    (path, labels) = dataset
    path, labels = shuffle_data(path, labels)
    X, Y = [], []
    bar = Bar('loading data', max=len(labels),
              fill='#', suffix='%(percent)d%%')
    for index, pk in enumerate(path):
        bar.next()
        try:
            with open(pk, "rb") as f:
                load_dict = pickle.load(f)
                x = load_dict["LogMel_Features"]
                x = x[:, :, np.newaxis]
                x = normalization_frames(x) 
                X.append(x)
                Y.append(labels_to_id[labels[index]])
        except Exception as e:
            print(e)
    X = np.array(X)
    Y = np.eye(num_class)[Y]
    bar.finish()
    return (X, Y)


def load_all_data(dataset, typeName):
    (path, labels) = dataset
    X, Y = [], []
    bar = Bar('loading data', max=len(path), fill='#', suffix='%(percent)d%%')
    for index, audio in enumerate(path):
        bar.next()
        try:
            with open(audio, "rb") as f:
                load_dict = pickle.load(f)
                x = load_dict["LogMel_Features"]
                x = x[:, :, np.newaxis]
                x = normalization_frames(x) 
                X.append(x)
                Y.append(labels[index])
        except Exception as e:
            print(e)
    bar.finish()
    return (np.array(X), np.array(Y))


def load_each_batch(dataset, labels_to_id, batch_start, batch_end, num_class):
    (paths, labels) = dataset
    X, Y = [], []
    for i in range(batch_start, batch_end):
        try:
            with open(paths[i], "rb") as f:
                load_dict = pickle.load(f)
                x = load_dict["LogMel_Features"]
                x = x[:, :, np.newaxis]
                x = normalization_frames(x) 
                X.append(x)
                Y.append(labels_to_id[labels[i]])
        except Exception as e:
            print(e)
    X = np.array(X)
    Y = np.eye(num_class)[Y]
    return X, Y


def Batch_generator(dataset, labels_to_id, batch_size, num_class):
    (paths, labels) = dataset
    length = len(labels)
    while True:
        # shuffle
        paths, labels = shuffle_data(paths, labels)
        shuffle_dataset = (paths, labels)
        batch_start = 0
        batch_end = batch_size
        while batch_end < length:
            X, Y = load_each_batch(
                shuffle_dataset, labels_to_id, batch_start, batch_end, num_class)
            yield (X, Y)
            batch_start += batch_size
            batch_end += batch_size


def caculate_distance(enroll_dataset, enroll_pre, test_pre):
    print("enroll_pre.shape=", enroll_pre.shape)
    dict_count = Counter(enroll_dataset[1])
    print(dict_count)
    # each person get a enroll_pre
    speakers_pre = []
    # remove repeat
    enroll_speakers = list(set(enroll_dataset[1]))
    enroll_speakers.sort(key=enroll_dataset[1].index)
    for speaker in enroll_speakers:
        start = enroll_dataset[1].index(speaker)
        speaker_pre = enroll_pre[start:dict_count[speaker]+start]
        speakers_pre.append(np.mean(speaker_pre, axis=0))

    enroll_pre = np.array(speakers_pre)
    print("new_enroll_pre.shape=", enroll_pre.shape)
    # caculate distance
    distances = []
    print("test_pre.shape=", test_pre.shape)
    for i in range(enroll_pre.shape[0]):
        temp = []
        for j in range(test_pre.shape[0]):
            x = enroll_pre[i]
            y = test_pre[j]
            s = np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))
            temp.append(s)
        distances.append(temp)
    distances = np.array(distances)
    print("distances.shape=", distances.shape)
    return distances


def speaker_identification(enroll_dataset, distances, enroll_y):
    #  remove repeat
    new_enroll_y = list(set(enroll_y))
    new_enroll_y.sort(key=list(enroll_y).index)
    #  return the index of max distance of each sentence
    y_pre_top = []
    socre_index = distances.argmax(axis=0)
    distances[socre_index] = -1
    y_pre = []
    for i in socre_index:
        y_pre.append(new_enroll_y[i])
    y_pre_top.append(y_pre)
    return y_pre_top



def compute_result(y_pre_top, y_true):
    dict_all = {'FileID':y_true}
    result = []
    for index,x in enumerate(y_true):
        pres = []
        for y_pre in y_pre_top:
            pres.append(y_pre[index])
        if x in pres:
            result.append(1)
        else:
            result.append(0)
            
    for index,y_pre in enumerate(y_pre_top):
        dict_all[f'top_{index+1}'] = y_pre

    dict_all['result'] = result

    df = pd.DataFrame(dict_all)
    df.to_csv('./result.csv',index=0)
    return result


def main(args):
    typeName = args.mode_type
    if typeName.startswith('train'):
        if not os.path.exists(c.MODEL_DIR):
            os.mkdir(c.MODEL_DIR)
        train_dataset, val_dataset = CreateDataset(args, split_ratio=0.1)
        nclass = len(set(train_dataset[1]))
        print("nclass = ",nclass)
        labels_to_id = Map_label_to_dict(labels=train_dataset[1])
        # load the model
        model = models.SE_ResNet(c.INPUT_SHPE)
        # model = models.Deep_speaker_model(c.INPUT_SHPE)
        # add softmax layer
        x = model.output
        x = Dense(nclass, activation='softmax', name=f'softmax')(x)
        model = Model(model.input, x)
        # model.summary()
        # exit()

        # 加载预训练模型
        filenames = os.listdir(f'{c.MODEL_DIR}/aishell')
        filenames = [hfile for hfile in glob.iglob(c.TRAIN_DEV_SET + "/*.h5")]
        if len(filenames):
            acc_lists = [os.path.splitext(f)[0].split("-")[1].split("_")[1] for f in filenames]
            optimal_model_index = acc_lists.index(min(acc_lists))
            model.load_weights(f'{c.MODEL_DIR}/aishell/{filenames[optimal_model_index]}')

         # train model
        sgd = optimizers.SGD(lr=c.LEARN_RATE,momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,
                        metrics=['accuracy'])
        model.fit_generator(Batch_generator(train_dataset, labels_to_id, c.BATCH_SIZE, nclass),
                            steps_per_epoch=len(train_dataset[0])//c.BATCH_SIZE, epochs=30,
                            validation_data=load_validation_data(
                                val_dataset, labels_to_id, nclass),
                            validation_steps=len(val_dataset[0])//c.BATCH_SIZE,
                            callbacks=[
            ModelCheckpoint(f'{c.MODEL_DIR}/aishell/best.h5',
                            monitor='val_loss', save_best_only=True, mode='min'),
            ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=10,mode='min'),
            EarlyStopping(monitor='val_loss', patience=10),
        ])

    else:
        test_dataset, enroll_dataset = CreateDataset(args,split_ratio=0,target=c.TARGET)
        # load weights
        model_se = models.SE_ResNet(c.INPUT_SHPE)
        model_se.load_weights(f'{c.MODEL_DIR}/aishell/seresnet/acc_0.707-eer_0.292.h5', by_name='True')

        model_dp = models.Deep_speaker_model(c.INPUT_SHPE)
        model_dp.load_weights(f'{c.MODEL_DIR}/aishell/deepspeaker/acc_0.685-eer_0.313.h5',by_name='True')
         # load all data
        print("loading data...")
        (enroll_x, enroll_y) = load_all_data(enroll_dataset, 'enroll')
        (test_x, test_y) = load_all_data(test_dataset, 'test')

        def distance_of_model(model):
            enroll_pre = np.squeeze(model.predict(enroll_x))
            test_pre = np.squeeze(model.predict(test_x))
            distances = caculate_distance(enroll_dataset, enroll_pre, test_pre)
            return distances

        distances_dp = distance_of_model(model_dp)
        distances_se = distance_of_model(model_se)
        distances = 0.3*normalization_frames(distances_dp) + 0.7*normalization_frames(distances_se)
        
        # speaker identification
        test_y_pre = speaker_identification(enroll_dataset, distances, enroll_y)
        #  compute result
        result = compute_result(test_y_pre, test_y)
        score = sum(result)/len(result)
        print(f"score={score}")


if __name__ == "__main__":
    # if len(sys.argv) < 2 or sys.argv[1] not in ['train', 'test']:
    #     print('Usage: python run.py [run_type]\n',
    #           '[run_type]: train | test')   
    #     exit()
    
    args = get_arguments()

    # # 加载数据集到表格中
    # enroll_audio_paths = [os.path.splitext(os.path.basename(wav))[0] for wav in glob.iglob(args.enroll_dataset + "/pickle/*.pickle")]
    # enroll_spkID = [os.path.basename(audio)[:14] for audio in enroll_audio_paths]
    # dict_enroll = {
    #     'FileID':enroll_audio_paths,
    #     'SpeakerID':enroll_spkID,
    # }
    # df = pd.DataFrame(dict_enroll)
    # df.to_csv('./dataset/enrollment.csv',index=0)

    # test_audio_paths,test_spkID = [],[]
    # with open('./data/verify/label.txt','r') as f:
    #     lines = f.read().split('\n')
    #     for line in lines:
    #         if len(line):
    #             test_audio_paths.append(line.split(' ')[0])
    #             test_spkID.append(line.split(' ')[1])
    # dict_test = {
    #     'FileID':test_audio_paths,
    #     'SpeakerID':test_spkID,
    # }
    # df = pd.DataFrame(dict_test)
    # df.to_csv('./dataset/test.csv',index=0)
    # # 主函数
    main(args)
