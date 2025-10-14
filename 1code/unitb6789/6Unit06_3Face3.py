import cv2
import mediapipe as mp
mouth_normal = cv2.imread("pic/m6.png")                # pic of mouth
mpd = mp.solutions.drawing_utils
mpfm = mp.solutions.face_mesh
dspec = mpd.DrawingSpec((128, 128, 128), 1, 1)
cspec = mpd.DrawingSpec((255, 255, 0), 1, 1)
cpoint = mpfm.FACEMESH_TESSELATION
fm = mpfm.FaceMesh(max_num_faces=1,min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    image = cv2.resize(image, (600, 360))
    imgrgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, d = image.shape                              #find the image shape
    results = fm.process(imgrgb)
    if results.multi_face_landmarks:
        for f_landmarks in results.multi_face_landmarks:
            mpd.draw_landmarks(image, landmark_list=f_landmarks, connections=cpoint,
                               landmark_drawing_spec=dspec, connection_drawing_spec=cspec)
            # point #0 and #17 are the top and button of mouth
            mouth_len = int((f_landmarks.landmark[17].y * h)-int(f_landmarks.landmark[0].y * h))
            mouth = cv2.resize(mouth_normal, (mouth_len * 3, mouth_len)) # resize the pic (m6.png) into the right size
            mouth_gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)         # convert pic to gray
            _, mouth_mask = cv2.threshold(mouth_gray, 25, 255, cv2.THRESH_BINARY_INV) # make a mask
            img_height, img_width, _ = mouth.shape  # find the img_height and img_width
            # the center is between point #13 and #14
            x, y = int(f_landmarks.landmark[13].x * w - img_width/2), \
                   int(((f_landmarks.landmark[13].y + f_landmarks.landmark[14].y)/2) * h - img_height/2)
            mouth_area = image[y: y + img_height, x: x + img_width]
            try:
                mouth_area_no_mouth = cv2.bitwise_and(mouth_area, mouth_area, mask=mouth_mask)
                mouth = cv2.add(mouth_area_no_mouth, mouth)    # put the mouth pic on top of image
                image[y: y+img_height, x: x+img_width] = mouth # at (x, y)
            except:
                print("An error occurred.")
    cv2.imshow("Unit06_3 | StudentID | faceLM3", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
