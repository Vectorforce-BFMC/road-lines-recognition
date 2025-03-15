from flask import Flask, Response, render_template_string
import cv2
import numpy as np
from functions import process_frame

app = Flask(__name__)

def generate_frames():
    capture1 = cv2.VideoCapture('vid1.mp4')
    
    while True:
        ret, frame = capture1.read()
        if not ret:
            break
        
        canny_output, masked_output, warped_binary, result, debug_img = process_frame(frame)
        
        # Resize images for display
        canny_resized = cv2.resize(canny_output, (300, 200))
        masked_resized = cv2.resize(masked_output, (300, 200))
        warped_resized = cv2.resize(warped_binary, (300, 200))
        result_resized = cv2.resize(result, (300, 200))
        
        # Convert grayscale images to BGR for proper display
        canny_resized = cv2.cvtColor(canny_resized, cv2.COLOR_GRAY2BGR)
        masked_resized = cv2.cvtColor(masked_resized, cv2.COLOR_GRAY2BGR)
        warped_resized = cv2.cvtColor(warped_resized, cv2.COLOR_GRAY2BGR)
        
        # Stack images in a 2x2 grid
        top_row = np.hstack((canny_resized, masked_resized))
        bottom_row = np.hstack((warped_resized, result_resized))
        combined_view = np.vstack((top_row, bottom_row))
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', combined_view)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    capture1.release()

@app.route('/')
def index():
    """Homepage with video feed."""
    return render_template_string("""
        <h1>Lane Detection Pipeline</h1>
        <img src="{{ url_for('video_feed') }}" width="600">
    """)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
