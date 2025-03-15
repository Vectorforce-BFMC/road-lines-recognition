from flask import Flask, Response, render_template_string
import cv2
import numpy as np
from functions import process_frame, CONFIG

app = Flask(__name__)

def generate_frames():
    capture1 = cv2.VideoCapture('vid1.mp4')
    
    while True:
        ret, frame = capture1.read()
        if not ret:
            break
        
        canny_output, masked_output, warped_binary, result, debug_img = process_frame(frame)
        rsimg5 = cv2.resize(result, (600, 400))  # Use the lane detection result
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', rsimg5)
        frame_bytes = buffer.tobytes()
        
        # Yield frame in a multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    capture1.release()

@app.route('/')
def index():
    """Homepage with link to the video feed."""
    return render_template_string("""
        <h1>Live Video Feed</h1>
        <img src="{{ url_for('video_feed') }}" width="600">
    """)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
