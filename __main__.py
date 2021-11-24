#!pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1 opencv-python mediapipe sklearn matplotlib

import numpy as np
import os
import time
import mediapipe as mp
import torch

from projet_SLR_mirror.preprocess import Preprocess
from projet_SLR_mirror.LSTM import LSTM
from projet_SLR_mirror.test import launch_test
from projet_SLR_mirror.load_LSTM import load_LSTM
from projet_SLR_mirror.dataset import CustomImageDataset
from projet_SLR_mirror.tuto import Tuto
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
# Gives easier dataset managment by creating mini batches etc.
from torch.utils.data import DataLoader
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules



class IntelVideoReader:
    """
    (Thread)
    * Reads frames from the intel Realsense D435I Camera (color and depth frames)
    """

    def __init__(self):
        import pyrealsense2 as rs

        self.pipe = rs.pipeline()
        config = rs.config()

        # ctx = rs.context()
        # devices = ctx.query_devices()
        # for dev in devices:
        #     dev.hardware_reset()

        self.width = 848
        self.height = 480

        config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, 30
        )
        config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, 30
        )

        profile = self.pipe.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        clipping_distance_in_meters = 3
        clipping_distance = clipping_distance_in_meters / self.depth_scale

        # device = profile.get_device()
        # depth_sensor = device.first_depth_sensor()
        # device.hardware_reset()

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)
        self.dec_filter = rs.decimation_filter()
        self.temp_filter = rs.temporal_filter()
        self.spat_filter = rs.spatial_filter()

    def next_frame(self):
        """Collects color and frames"""
        frameset = self.pipe.wait_for_frames()

        aligned_frames = self.align.process(frameset)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        self.depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        self.color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

        depth_frame = self.depth_to_disparity.process(depth_frame)
        depth_frame = self.dec_filter.process(depth_frame)
        depth_frame = self.temp_filter.process(depth_frame)
        depth_frame = self.spat_filter.process(depth_frame)
        depth_frame = self.disparity_to_depth.process(depth_frame)
        depth_frame = depth_frame.as_depth_frame()

        color_frame = np.fliplr(np.asanyarray(color_frame.get_data()))
        depth_frame = np.fliplr(np.asanyarray(depth_frame.get_data()))

        return [color_frame, depth_frame]

def launch_LSTM(input_size, output_size, train):
    # Set device cuda for GPU if it's available otherwise run on the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hyperparameters of our neural network which depends on the dataset, and
    # also just experimenting to see what works well (learning rate for example).

    learning_rate = 0.001  # how much to update models parameters at each batch/epoch
    batch_size = 32  # number of data samples propagated through the network before the parameters are updated
    NUM_WORKERS = 4
    num_epochs = 1000  # number times to iterate over the dataset
    DECAY = 1e-4
    hidden_size = 128  # number of features in the hidden state h
    num_layers = 2

    #print("len X_train:",len(X_train)*len(X_train[0])*len(X_train[0][0]))
    # l'input_size doit être 30*30*1662

    train_loader = DataLoader(datas_train, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True)

    test_loader = DataLoader(datas_test, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
                             pin_memory=True)
    # Initialize network
    model = LSTM(input_size,  hidden_size, num_layers, output_size).to(device)

    if(train): # On verifie si on souhaite reentrainer le modele
        model = train_launch(model, learning_rate, DECAY, num_epochs, train_loader, test_loader)
    else:
        try:
            model.load_state_dict(torch.load("actionNN.pth"))
        except:
            model = train_launch(model, learning_rate, DECAY, num_epochs, train_loader, test_loader)

    return model  # ,logits

def train_launch(model, learning_rate, DECAY, num_epochs, train_loader, test_loader):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=DECAY)
    # Train Network

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        model.train_loop(train_loader, model, criterion, optimizer)

        print(
            f"Accuracy on training set: {model.test_loop(train_loader, model)*100:.2f}")
        print(
            f"Accuracy on test set: {model.test_loop(test_loader, model)*100:.2f}")
    print("Done!")
    # model = models.vgg16(pretrained=True).cuda()
    torch.save(model.state_dict(), 'actionNN.pth')
    return model


# on crée des dossiers dans lequels stocker les positions des points que l'on va enregistrer
# Chemin pour les données
DATA_PATH = os.path.join('MP_Data')



# Thirty videos worth of data
nb_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# dataset making : (à décommenter si vous voulez creer votre dataset)

actionsToAdd = ["nothing"] #actions à refaire ou à ajouter
CustomImageDataset(actionsToAdd, nb_sequences, sequence_length, DATA_PATH).__getitem__()

# Actions that we try to detect
actions = np.array(["nothing", 'hello', 'thanks', 'iloveyou', "what's up", "hey", "my", "name"])

# reprocess
datas_train = Preprocess("train", actions, DATA_PATH, sequence_length)
datas_test = Preprocess("test", datas_train, None, None)

input_size = len(datas_train["X_train"][0][0])

#Appel du modele
train = False
model = launch_LSTM(input_size,len(actions), train)

cap = IntelVideoReader()
for action in actions:  
    if (action != "nothing"):
        Tuto(actions,action).launch_tuto()
        launch_test(actions, model, action)
