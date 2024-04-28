from flask import Flask, Response
import cv2
import numpy as np
from PIL import ImageGrab

app = Flask(__name__)

import cv2
import numpy as np
from PIL import ImageGrab

import time

def generate_frames():
    target_fps = 5
    frame_interval = 1 / target_fps
    last_frame_time = time.time()

    while True:
        # Capture the screen frame
        img = ImageGrab.grab()
        img_np = np.array(img)
        frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        # Resize the frame to 480p resolution
        frame = cv2.resize(frame, (640, 360))  # 480p resolution is 854x480

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame as an HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Calculate the time elapsed since the last frame
        current_time = time.time()
        elapsed_time = current_time - last_frame_time

        # If the elapsed time is less than the desired frame interval, wait
        if elapsed_time < frame_interval:
            time.sleep(frame_interval - elapsed_time)

        # Update the last frame time for the next iteration
        last_frame_time = time.time()

@app.route('/')
def index():
    return '''
        <html>
            <head>
                <title>Screen Sharing</title>
            </head>
            <body>
                <h1>Screen Sharing</h1>
                <img src="/screen_feed">
            </body>
        </html>
    '''

@app.route('/screen_feed')
def screen_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)