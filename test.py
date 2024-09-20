import cv2
import mediapipe as mp
import numpy as np

# Video capture settings
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Increase width
cap.set(4, 720)   # Increase height

# Hand configurations (change this when there are multiple people)
max_num_hands = 2

# Hand definitions
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=max_num_hands, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Calculated distances for finger counting
calculated_distances = [[5, 4], [6, 8], [10, 12], [14, 16], [18, 20]]

def is_ok_gesture(hand_landmarks):
    if hand_landmarks:
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        distance = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
        return distance < 0.05
    return False

while cap.isOpened():
    success, img = cap.read()
    
    if success:
        # Mirror effect
        img = cv2.flip(img, 1)
        
        # BGR to RGB Color conversion
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process hands to count
        results = hands.process(img_rgb)
        
        # Finger counter
        counter = 0

        # Check if hands are detected
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                # Draw 21 landmarks
                mp_drawing.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
                
                motions = []
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    # Convert ratios to real positions
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    
                    # If it"s the thumb finger, adjust y position
                    if id == 4:
                        cy = ((cy + motions[3][2]) / 2) + cap.get(4) / 30 if len(motions) > 3 else cy
                        
                    # Add finger landmark position [id, coordinate x, coordinate y]
                    motions.append([id, cx, cy])
        
                for item in calculated_distances:
                    downFingerPosY = motions[item[0]][2]
                    upperFingerPosY = motions[item[1]][2]
                    # Check if the finger is open
                    isFingerOpen = downFingerPosY > upperFingerPosY
                    counter += 1 if isFingerOpen else 0

                # Check for "OK" gesture
                if is_ok_gesture(handLms):
                    # Save the image without text
                    cv2.imwrite("ok_gesture.jpg", img)

                    # Now add text to display on screen (but not on saved image)
                    text = "OK Gesture Detected!"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    text_x = (img.shape[1] - text_size[0]) // 2
                    text_y = (img.shape[0] + text_size[1]) // 2
                    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # debugged in 20/9
                    break

        # Draw rectangle and put text for counting operation                        
        cv2.rectangle(img, (0, 0), (250, 80), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, str(counter), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)  # Green big text
        
        # Show all of these
        cv2.imshow("Capture", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()