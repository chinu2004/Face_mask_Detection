from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("face_mask_cnn_model.h5")
face_cascade = cv2.CascadeClassifier(r"C:\Users\USER\Desktop\mask detection\chinu_frontal_face_haar_cascade.xml")

# Global flag to control camera
streaming = False
camera = None

def gen_frames():
    global camera
    while streaming:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(136, 136))

            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face_img, (128, 128))
                face_array = np.expand_dims(face_resized, axis=0) / 255.0
                prediction = model.predict(face_array)
                label = "Mask" if prediction[0][0] > 0.5 else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', streaming=streaming)

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start():
    global streaming, camera
    streaming = True
    camera = cv2.VideoCapture(0)
    return redirect(url_for('index'))

@app.route('/stop')
def stop():
    global streaming, camera
    streaming = False
    if camera:
        camera.release()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
