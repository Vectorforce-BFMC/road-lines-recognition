import cv2
import numpy as np

# Configuration dictionary for easy parameter tuning
CONFIG = {
    'canny_low': 30,
    'canny_high': 150,
    'blur_kernel': 5,
    'n_windows': 9,
    'window_margin': 100,
    'min_pixels': 50,
    'white_lower': [0, 200, 0],
    'white_upper': [255, 255, 255],
    'yellow_lower': [15, 38, 115],
    'yellow_upper': [35, 204, 255],
    'lane_color': (0, 255, 0),
    'lane_thickness': 10,
    'debug': False  # Set to True for visualization
}

# Global variables for lane tracking
prev_left_fit = None
prev_right_fit = None

def canny(img):
    """Convert image to grayscale and apply Canny edge detection."""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (CONFIG['blur_kernel'], CONFIG['blur_kernel']), 0)
    edges = cv2.Canny(blur, CONFIG['canny_low'], CONFIG['canny_high'])
    return edges

def define_roi_points(height, width):
    """Define ROI trapezoid points based on image dimensions."""
    return np.array([
        [(int(width * 0.2), height),           # Bottom left
         (int(width * 0.45), int(height * 0.75)), # Top left
         (int(width * 0.55), int(height * 0.75)), # Top right
         (int(width * 0.8), height)]           # Bottom right
    ], np.int32)

def apply_roi(img, polygon):
    """Apply region of interest mask to the image."""
    mask = np.zeros_like(img)
    if len(img.shape) == 3:
        cv2.fillPoly(mask, [polygon], (255, 255, 255))
    else:
        cv2.fillPoly(mask, [polygon], 255)
    return cv2.bitwise_and(img, mask)

def perspective_transform(img, src_points):
    """Apply perspective transform to get a bird's-eye view."""
    height, width = img.shape[:2]
    dst_points = np.array([
        [width * 0.2, height],
        [width * 0.2, 0],
        [width * 0.8, 0],
        [width * 0.8, height]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points)
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped, M

def enhance_lane_marks(img):
    """Enhance lane markings with adaptive thresholding."""
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    
    # Static color thresholds
    white_mask = cv2.inRange(hls, np.array(CONFIG['white_lower']), np.array(CONFIG['white_upper']))
    yellow_mask = cv2.inRange(hls, np.array(CONFIG['yellow_lower']), np.array(CONFIG['yellow_upper']))
    
    # Adaptive thresholding for robustness
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adapt_thresh = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    combined_mask = cv2.bitwise_or(combined_mask, adapt_thresh)
    return combined_mask

def sliding_window_lane_detection(binary_warped, debug_img=None):
    """Detect lane lines using sliding window method."""
    global prev_left_fit, prev_right_fit
    
    if len(binary_warped.shape) > 2:
        binary_warped = cv2.cvtColor(binary_warped, cv2.COLOR_BGR2GRAY)
    
    # Histogram of bottom half
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    if np.max(histogram) < 10:
        return prev_left_fit, prev_right_fit  # Use previous fits if detection fails
    
    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = binary_warped.shape[0] // CONFIG['n_windows']
    nonzero = binary_warped.nonzero()
    nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
    
    leftx_current, rightx_current = leftx_base, rightx_base
    left_lane_inds, right_lane_inds = [], []
    
    for window in range(CONFIG['n_windows']):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        
        win_xleft_low = leftx_current - CONFIG['window_margin']
        win_xleft_high = leftx_current + CONFIG['window_margin']
        win_xright_low = rightx_current - CONFIG['window_margin']
        win_xright_high = rightx_current + CONFIG['window_margin']
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                         (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > CONFIG['min_pixels']:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > CONFIG['min_pixels']:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
        
        # Debug visualization
        if CONFIG['debug'] and debug_img is not None:
            cv2.rectangle(debug_img, (win_xleft_low, win_y_low), 
                         (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(debug_img, (win_xright_low, win_y_low), 
                         (win_xright_high, win_y_high), (0, 255, 0), 2)
    
    left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else []
    right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else []
    
    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]
    
    if len(leftx) < 10 or len(rightx) < 10:
        return prev_left_fit, prev_right_fit
    
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        # Smooth with previous fits if available
        if prev_left_fit is not None and prev_right_fit is not None:
            left_fit = 0.7 * left_fit + 0.3 * prev_left_fit
            right_fit = 0.7 * right_fit + 0.3 * prev_right_fit
        
        prev_left_fit, prev_right_fit = left_fit, right_fit
        return left_fit, right_fit
    except Exception as e:
        print(f"Lane fitting error: {e}")
        return prev_left_fit, prev_right_fit

def draw_lanes(img, left_fit, right_fit, Minv):
    """Draw lanes on the original image with inverse perspective transform."""
    height = img.shape[0]
    result = img.copy()
    
    if left_fit is None or right_fit is None:
        return result
    
    ploty = np.linspace(0, height - 1, 20)
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
    
    # Warp points back to original perspective
    warp_points = np.stack((left_fitx, ploty), axis=1)
    left_points = cv2.perspectiveTransform(np.array([warp_points], dtype=np.float32), Minv)[0]
    warp_points = np.stack((right_fitx, ploty), axis=1)
    right_points = cv2.perspectiveTransform(np.array([warp_points], dtype=np.float32), Minv)[0]
    
    for i in range(len(ploty) - 1):
        cv2.line(result, 
                (int(left_points[i, 0]), int(left_points[i, 1])), 
                (int(left_points[i+1, 0]), int(left_points[i+1, 1])), 
                CONFIG['lane_color'], CONFIG['lane_thickness'])
        cv2.line(result, 
                (int(right_points[i, 0]), int(right_points[i, 1])), 
                (int(right_points[i+1, 0]), int(right_points[i+1, 1])), 
                CONFIG['lane_color'], CONFIG['lane_thickness'])
    
    # Draw lane area
    lane_pts = np.vstack((left_points, np.flipud(right_points))).astype(np.int32)
    mask = np.zeros_like(result)
    cv2.fillPoly(mask, [lane_pts], (0, 100, 0))
    result = cv2.addWeighted(result, 1, mask, 0.3, 0)
    
    return result

def process_frame(frame):
    """Process a single video frame for lane detection."""
    # Define ROI
    roi_points = define_roi_points(frame.shape[0], frame.shape[1])
    
    # Edge detection and ROI
    canny_output = canny(frame)
    masked_output = apply_roi(canny_output, roi_points)
    
    # Enhance lane markings and apply ROI
    lane_binary = enhance_lane_marks(frame)
    lane_binary_masked = apply_roi(lane_binary, roi_points)
    
    # Perspective transform
    warped_binary, M = perspective_transform(lane_binary_masked, roi_points)
    Minv = np.linalg.inv(M)
    
    # Lane detection
    debug_img = frame.copy() if CONFIG['debug'] else None
    left_fit, right_fit = sliding_window_lane_detection(warped_binary, debug_img)
    
    result = draw_lanes(frame, left_fit, right_fit, Minv)
    
    return canny_output, masked_output, warped_binary, result, debug_img


