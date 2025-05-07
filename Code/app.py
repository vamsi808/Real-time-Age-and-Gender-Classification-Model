from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load models
faceProto = "models/deploy.prototxt"
faceModel = "models/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

genderProto = "models/gender_deploy.prototxt"
genderModel = "models/gender_net.caffemodel"
genderNet = cv2.dnn.readNet(genderModel, genderProto)

ageProto = "models/age_deploy.prototxt"
ageModel = "models/age_net.caffemodel"
ageNet = cv2.dnn.readNet(ageModel, ageProto)

genderList = ['Male', 'Female']
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

def detect_age_gender(frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), MODEL_MEAN_VALUES, swapRB=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    result_frame = frame.copy()

    h, w = frame.shape[:2]

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype(int)
            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            genderNet.setInput(face_blob)
            gender_preds = genderNet.forward()
            gender = genderList[gender_preds[0].argmax()]

            ageNet.setInput(face_blob)
            age_preds = ageNet.forward()
            age = ageList[age_preds[0].argmax()]

            label = f"{gender}, {age}"
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return result_frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(request.url)

    image = request.files['image']
    if image.filename == '':
        return redirect(request.url)

    filepath = os.path.join('static', 'uploads', image.filename)
    image.save(filepath)

    img = cv2.imread(filepath)
    result_img = detect_age_gender(img)

    result_path = os.path.join('static', 'results', 'result.jpg')
    cv2.imwrite(result_path, result_img)

    return render_template('result.html', result_image='results/result.jpg')

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            result_frame = detect_age_gender(frame)
            ret, buffer = cv2.imencode('.jpg', result_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/results', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    app.run(debug=True)
