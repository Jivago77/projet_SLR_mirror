from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor, Lambda
from typing import Tuple
#from tensorflow.keras.utils import to_categorical
import torch
import torch.nn as nn

import cv2
import numpy as np
import os


class Preprocess():
    def __init__(self, processType: str, actions, DATA_PATH: str, sequence_length: int):
        if(DATA_PATH is None):
            self.processType = processType
            process = actions
            self.X_train = process["X_train"]
            self.y_train = process["y_train"]
            self.X_test = process["X_test"]
            self.y_test = process["y_test"]

        else:
            self.actions = actions
            self.DATA_PATH = DATA_PATH
            self.sequence_length = sequence_length
            self.processType = processType

            label_map = {label: num for num, label in enumerate(self.actions)}
            hot_sequences, sequences, = [], []

            for idx_action, action in enumerate(self.actions):

                for sequence in np.array(os.listdir(os.path.join(self.DATA_PATH, action))).astype(int):
                    hot_sequences.append(idx_action)
                    window = []
                    for frame_num in range(self.sequence_length):
                        res = np.load(os.path.join(self.DATA_PATH, action, str(
                            sequence), "{}.npy".format(frame_num)))
                        # window.extend(res)
                        window.append(res)
                    sequences.append(window)
                    # datas.append(sequences)

            self.X = sequences

            self.y = hot_sequences  # .to(torch.float64)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.05, shuffle=False)

            flatten = nn.Flatten()
            self.X_train = torch.tensor(self.X_train, dtype=torch.float)
            self.y_train = torch.tensor(self.y_train, dtype=torch.long)
            self.X_test = torch.tensor(self.X_test, dtype=torch.float)
            self.y_test = torch.tensor(self.y_test, dtype=torch.long)

    def __getitem__(self, idx_seq: int) -> Tuple[torch.Tensor, int]:

        if(idx_seq == "X_train"):
            return self.X_train
        elif(idx_seq == "X_test"):
            return self.X_test
        elif(idx_seq == "y_train"):
            return self.y_train
        elif(idx_seq == "y_test"):
            return self.y_test
        else:
            if (self.processType == "train"):
                data = self.X_train[idx_seq], self.y_train[idx_seq]
            elif (self.processType == "test"):
                data = self.X_test[idx_seq], self.y_test[idx_seq]

            return data

    def __len__(self):
        if (self.processType == "train"):
            dataLen = len(self.X_train)
        elif (self.processType == "test"):
            dataLen = len(self.X_test)
        return dataLen
