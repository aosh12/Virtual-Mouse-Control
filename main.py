import cv2
import mediapipe as mp
import pyautogui
import math
import time
import speech_recognition as sr
import threading

# VOICE COMMAND FUNCTION

recognizer = sr.Recognizer()

def voice_control():

    while True:

        try:
            with sr.Microphone() as source:

                recognizer.adjust_for_ambient_noise(source)

                print("Listening...")

                audio = recognizer.listen(source)

                command = recognizer.recognize_google(audio)

                command = command.lower()

                print("Command:", command)

                # Scroll
                if "scroll up" in command:
                    for _ in range(8):
                        pyautogui.scroll(300)

                elif "scroll down" in command:
                    for _ in range(8):
                        pyautogui.scroll(-300)

                # Clicks
                elif "left click" in command:
                    pyautogui.click()

                elif "right click" in command:
                    pyautogui.rightClick()

                elif "double click" in command:
                    pyautogui.doubleClick()

                # Volume
                elif "volume up" in command:
                    for _ in range(5):
                        pyautogui.press("volumeup")

                elif "volume down" in command:
                    for _ in range(5):
                        pyautogui.press("volumedown")

        except:
            pass

# Run voice control in separate thread
voice_thread = threading.Thread(target=voice_control)
voice_thread.daemon = True
voice_thread.start()

# HAND TRACKING SECTION

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

screen_width, screen_height = pyautogui.size()

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

prev_x = 0
prev_y = 0

smoothening = 5

click_threshold = 30
right_click_threshold = 35
drag_threshold = 25

left_clicked = False
right_clicked = False
dragging = False

drag_start_time = 0

while True:

    success, img = cap.read()

    if not success:
        print("Failed to access webcam")
        break

    img = cv2.flip(img, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    frame_height, frame_width, _ = img.shape

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            middle_tip = hand_landmarks.landmark[12]

            ix = int(index_tip.x * frame_width)
            iy = int(index_tip.y * frame_height)

            tx = int(thumb_tip.x * frame_width)
            ty = int(thumb_tip.y * frame_height)

            mx = int(middle_tip.x * frame_width)
            my = int(middle_tip.y * frame_height)

            cv2.circle(img, (ix, iy), 10, (0, 255, 0), -1)
            cv2.circle(img, (tx, ty), 10, (255, 0, 0), -1)
            cv2.circle(img, (mx, my), 10, (0, 0, 255), -1)

            screen_x = screen_width / frame_width * ix
            screen_y = screen_height / frame_height * iy

            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening

            pyautogui.moveTo(curr_x, curr_y)

            prev_x = curr_x
            prev_y = curr_y

            left_distance = math.hypot(tx - ix, ty - iy)

            right_distance = math.hypot(tx - mx, ty - my)

            # LEFT CLICK
            if left_distance < click_threshold and not left_clicked:

                pyautogui.click()

                left_clicked = True

            if left_distance > click_threshold:
                left_clicked = False

            # RIGHT CLICK
            if right_distance < right_click_threshold and not right_clicked:

                pyautogui.rightClick()

                right_clicked = True

            if right_distance > right_click_threshold:
                right_clicked = False

            # DRAG
            if left_distance < drag_threshold:

                if drag_start_time == 0:
                    drag_start_time = time.time()

                elif time.time() - drag_start_time > 1 and not dragging:

                    pyautogui.mouseDown()

                    dragging = True

            else:

                drag_start_time = 0

                if dragging:

                    pyautogui.mouseUp()

                    dragging = False

    cv2.imshow("AI Virtual Mouse + Voice", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()