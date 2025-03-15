import cv2
import numpy as np
from functions import process_frame, CONFIG

def main():
    capture1 = cv2.VideoCapture('vid1.mp4')
    
    while True:
        ret, frame = capture1.read()
        if not ret:
            break
        
        canny_output, masked_output, warped_binary, result, debug_img = process_frame(frame)
        
        rsimg1 = cv2.resize(frame, (600, 400))
        rsimg2 = cv2.resize(canny_output, (600, 400))
        rsimg3 = cv2.resize(masked_output, (600, 400))
        rsimg4 = cv2.resize(warped_binary, (600, 400))
        rsimg5 = cv2.resize(result, (600, 400))
        
        # Display results
        cv2.imshow('Original Frame', rsimg1)
        cv2.imshow('Canny Edges', rsimg2)
        cv2.imshow('Masked Edges', rsimg3)
        cv2.imshow('Warped Binary', rsimg4)
        cv2.imshow('Lane Detection Result', rsimg5)
        
        if debug_img is not None:
            rsimg6 = cv2.resize(debug_img, (600, 400))
            cv2.imshow('Debug: Sliding Windows', rsimg6)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    capture1.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
