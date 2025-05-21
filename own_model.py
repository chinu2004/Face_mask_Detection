# scaleFactor=1.2, minNeighbors=7, minSize=(60, 60)
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your face mask detection model
model = load_model("face_mask_cnn_model.h5")

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(r"C:\Users\USER\Desktop\facial_expression\chinu_frontal_face_haar_cascade.xml")

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=9, minSize=(128, 128))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        face_img = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_img, (128, 128))  # Resize to model input size
        face_array = np.expand_dims(face_resized, axis=0) / 255.0  # Normalize

        # Predict using the model
        prediction = model.predict(face_array)
        label = "Mask" if prediction[0][0] > 0.5 else "No Mask"

        # Display the results
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show the frame
    cv2.imshow("Face Mask Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
