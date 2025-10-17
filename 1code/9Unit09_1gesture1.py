#9Unit09_1Face1.py
import cv2
import mediapipe as mp
import math

mpd = mp.solutions.drawing_utils
mpd_styles = mp.solutions.drawing_styles
mph = mp.solutions.hands

def vector_2d_angle(v1, v2):  # Calculate the angle based on the coordinates of two points
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    try:
        angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle_ = 180
    return angle_

def hand_angle(hand_): # Calculate the angle of the finger based on the 21 node coordinates
    angle_list = []
    angle_ = vector_2d_angle(((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
                             ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1]))))
    angle_list.append(angle_) # thumb angle
    angle_ = vector_2d_angle(((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
                             ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1]))))
    angle_list.append(angle_) # index finger angle
    angle_ = vector_2d_angle(((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
                             ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1]))))
    angle_list.append(angle_)# middle finger angle
    angle_ = vector_2d_angle(((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
                             ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1]))))
    angle_list.append(angle_) # ring finger angle
    angle_ = vector_2d_angle(((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
                             ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1]))))
    angle_list.append(angle_) # pink finger angle
    return angle_list

def hand_pos(finger_angle): # return the corresponding gesture name.
    f1 = finger_angle[0]    # thumb angle
    f2 = finger_angle[1]    # index finger angle
    f3 = finger_angle[2]    # middle finger angle
    f4 = finger_angle[3]    # ring finger angle
    f5 = finger_angle[4]    # pink finger angle
    if f1<50 and f2>=50 and f3>=50 and f4>=50 and f5>=50:# <50 straight finger,>=50 curled finger
        return 'good'
    elif f1>=50 and f2>=50 and f3<50 and f4>=50 and f5>=50:
        return 'no!!!'
    elif f1<50 and f2<50 and f3>=50 and f4>=50 and f5<50:
        return 'ROCK!'
    elif f1>=50 and f2>=50 and f3>=50 and f4>=50 and f5>=50:
        return '0'
    elif f1>=50 and f2>=50 and f3>=50 and f4>=50 and f5<50:
        return 'pinky'
    elif f1>=50 and f2<50 and f3>=50 and f4>=50 and f5>=50:
        return '1'
    elif f1>=50 and f2<50 and f3<50 and f4>=50 and f5>=50:
        return '2'
    elif f1>=50 and f2>=50 and f3<50 and f4<50 and f5<50:
        return 'ok'
    elif f1<50 and f2>=50 and f3<50 and f4<50 and f5<50:
        return 'ok'
    elif f1>=50 and f2<50 and f3<50 and f4<50 and f5>50:
        return '3'
    elif f1>=50 and f2<50 and f3<50 and f4<50 and f5<50:
        return '4'
    elif f1<50 and f2<50 and f3<50 and f4<50 and f5<50:
        return '5'
    elif f1<50 and f2>=50 and f3>=50 and f4>=50 and f5<50:
        return '6'
    elif f1<50 and f2<50 and f3>=50 and f4>=50 and f5>=50:
        return '7'
    elif f1<50 and f2<50 and f3<50 and f4>=50 and f5>=50:
        return '8'
    elif f1<50 and f2<50 and f3<50 and f4<50 and f5>=50:
        return '9'
    else:
        return ''

cap = cv2.VideoCapture(0)
hands = mph.Hands(model_complexity=0,  max_num_hands=2)
w, h = 540, 310
while cap.isOpened():
    success, image = cap.read()
    img = cv2.resize(image, (w,h))
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img2)
    if results.multi_hand_landmarks:
        for h_landmarks in results.multi_hand_landmarks:
            finger_points = []               #Set the list to record finger joint coordinates
            for i in h_landmarks.landmark:# Convert to coordinates and record them in the 'finger_points'
                x = i.x*w
                y = i.y*h
                finger_points.append((x,y))
            if finger_points:
                finger_angle = hand_angle(finger_points) # Calculate finger angles and return a list of length 5
                text = hand_pos(finger_angle)            # Retrieve the content returned by the gesture
                cv2.putText(img, text, (30,120), 1, 5, (0,0,255), 10, 1)
    cv2.imshow('Unit09_1 | StudentID | gesture1', img)
    if cv2.waitKey(5)& 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()