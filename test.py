from scipy import stats
import torch
import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def mediapipe_detection(image, model,cap):
    
    image, _ = cap.next_frame()
    # COLOR CONVERSION BGR 2 RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60+num*40),
                      (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return output_frame


def draw_styled_landmarks(image, results):
    # Draw face connections
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
    #                          mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
    #                          mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    #                          )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(80, 44, 121), thickness=2, circle_radius=2)
                              )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(121, 44, 250), thickness=2, circle_radius=2)
                              )


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten(
    ) if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten(
    ) if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
    ) if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
    ) if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60+num*40),
                      (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return output_frame


def launch_test(actions, model, action, cap):

    colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

    sentence = []
    threshold = 0.9

    count_valid = 0
    RESOLUTION_Y = int(1920*9/10)  # Screen resolution in pixel
    RESOLUTION_X = int(1080*9/10)
    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()
            

            # Make detections
            image, results = mediapipe_detection(frame, holistic,cap)
            image = cv2.resizeWindow()
            # print(results)
            image = cv2.resize(image,(RESOLUTION_X,RESOLUTION_Y))
            # Draw landmarks
            draw_styled_landmarks(image, results)
            image = cv2.flip(image, 1)
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
    #         sequence.insert(0,keypoints)
    #         sequence = sequence[:30]
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                #res = model.predict(np.expand_dims(sequence, axis=0))[0]
                res = torch.softmax(
                    model(torch.tensor(
                        sequence, dtype=torch.float).cuda().unsqueeze(0)),
                    dim=1
                ).cpu().detach().numpy()[0]
                print(actions[np.argmax(res)])

            # 3. Viz logic
                if np.max(res) > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Viz probabilities
                image = prob_viz(res, actions, image, colors)
                if(actions[np.argmax(res)] == action):
                    count_valid +=1
                else : count_valid = 0

                if(count_valid ==10):
                    print("VALIDATED")
                    break    
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) 

            cv2.imshow('Raw Webcam Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    indCol = 0

    prob = res[np.argmax(res)]
    cv2.rectangle(output_frame, (0, 60),
                      (int(prob*100), 90),
                      colors[indCol], -1)
    cv2.putText(output_frame, actions[np.argmax(res)], (0, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # for num, prob in enumerate(res):
    #     #si on dépasse la taille du tableau de couleur on revient au début
    #     if(indCol >= len(colors)):
    #         indCol = indCol-len(colors) 
    #     print(num)
    #     cv2.rectangle(output_frame, (0, 60+num*40),
    #                   (int(prob*100), 90+num*40),
    #                   colors[indCol], -1)
    #     cv2.putText(output_frame, actions[num], (0, 85+num*40),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    #     indCol+=1
    
    return output_frame
