import cv2
import os
import mediapipe as mp
import numpy as np
from projet_SLR_mirror.test import mediapipe_detection, draw_styled_landmarks, extract_keypoints

mp_holistic = mp.solutions.holistic  # Holistic model


class CustomImageDataset():
    # class CustomImageDataset(Dataset):
    def __init__(self, actionsToAdd, nb_sequences, sequence_length, DATA_PATH):
        self.actionsToAdd = actionsToAdd
        self.nb_sequences = nb_sequences
        self.sequence_length = sequence_length
        self.DATA_PATH = DATA_PATH
        print('dataset init')

    def __len__(self):
        return len(self.actionsToAdd)*len(self.nb_sequences)

    def __getitem__(self):
        for action in self.actionsToAdd:
            for sequence in range(self.nb_sequences):
                try:
                    os.makedirs(os.path.join(
                        self.DATA_PATH, action, str(sequence)))
                    print("sucess")
                except:
                    pass

        cap = cv2.VideoCapture(0)
        # Set mediapipe model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            # Loop through actionsToAdd
            for action in self.actionsToAdd:
                # Loop through sequences aka videos
                for sequence in range(self.nb_sequences):
                    # Loop through video length aka sequence length
                    for frame_num in range(self.sequence_length):

                        # Read feed
                        ret, frame = cap.read()

                        # Make detections
                        image, results = mediapipe_detection(frame, holistic)

                        # Draw landmarks
                        draw_styled_landmarks(image, results)

                        # NEW Apply wait logic
                        if frame_num == 0:
                            cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)
                            cv2.waitKey(2000)
                        else:
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)

                        # NEW Export keypoints
                        keypoints = extract_keypoints(results)
                        npy_path = os.path.join(
                            self.DATA_PATH, action, str(sequence), str(frame_num))
                        np.save(npy_path, keypoints)

                        # Break gracefully
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break

            cap.release()
            cv2.destroyAllWindows()
