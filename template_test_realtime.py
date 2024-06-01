import cv2
import numpy as np
import threading
import time

# Function to process an image with template matching
def process_image(img_rgb, template_gray, sift, flann, prev_dst, alpha=0.5, match_threshold=12):
    # Convert images to grayscale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    # Find keypoints and descriptors with SIFT for the image
    keypoints2, descriptors2 = sift.detectAndCompute(img_gray, None)
    
    # Check if descriptors are found
    if descriptors2 is None or len(descriptors2) == 0:
        return img_rgb, prev_dst, False

    # Match descriptors
    matches = flann.knnMatch(template_descriptors, descriptors2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.85 * n.distance:
            good_matches.append(m)

    template_detected = len(good_matches) >= match_threshold

    # If enough matches are found, we extract the locations of matched keypoints in both images
    if template_detected:
        src_pts = np.float32([template_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        # Compute homography matrix
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
        matches_mask = mask.ravel().tolist()

        # Get the corners from the template image
        h, w = template_gray.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

        # Apply perspective transform to get the corners in the main image
        dst = cv2.perspectiveTransform(pts, M)

        # Smooth the detected region
        if prev_dst is not None:
            dst = alpha * dst + (1 - alpha) * prev_dst

        if detection_count > detection_threshold:
            # Draw the detected region in the main image
            img_rgb = cv2.polylines(img_rgb.copy(), [np.int32(dst)], True, (0, 255, 0), 2, cv2.LINE_AA)

        prev_dst = dst
    else:
        prev_dst = None

    return img_rgb, prev_dst, template_detected

# Class to handle frame capture in a separate thread
class FrameCaptureThread(threading.Thread):
    def __init__(self, src=0, scale=0.5):
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(src)
        self.ret = False
        self.frame = None
        self.stopped = False
        self.scale = scale

    def run(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()
            if self.ret and self.scale != 1.0:
                self.frame = cv2.resize(self.frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

    def stop(self):
        self.stopped = True
        self.cap.release()

# Read the template
template = cv2.imread('template.png')
assert template is not None, "Template file could not be read, check with os.path.exists()"

# Convert template to grayscale
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Initialize the SIFT detector
sift = cv2.SIFT_create(nfeatures=5000)

# Find keypoints and descriptors with SIFT for the template
template_keypoints, template_descriptors = sift.detectAndCompute(template_gray, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Start frame capture thread with downsampling scale factor
frame_thread = FrameCaptureThread(scale=1)
frame_thread.start()

prev_dst = None  # Previous detected region corners
detection_count = 0
detection_threshold = 3  # Number of frames the detection must persist
frame_skip = 5

# Initialize variables for FPS calculation
frame_counter = 0
fps = 0
fps_display_interval = 1  # seconds
frame_count_time = time.time()

try:
    while True:

        if frame_thread.ret:
            frame = frame_thread.frame

            # Process only every N-th frame
            if frame_counter % frame_skip == 0:
                detected_img, prev_dst, template_detected = process_image(frame, template_gray, sift, flann, prev_dst)

            # Update detection count based on template_detected
            if template_detected:
                detection_count += 1
            else:
                detection_count = 0

            # Display detection status
            if detection_count >= detection_threshold:
                cv2.putText(detected_img, 'Special CNIC detected!', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                # Draw detected template
                # if prev_dst is not None:
                #     detected_img = cv2.polylines(detected_img.copy(), [np.int32(prev_dst)], True, (0, 255, 0), 2, cv2.LINE_AA)

            # Update FPS calculation
            frame_counter += 1
            if (time.time() - frame_count_time) >= fps_display_interval:
                fps = frame_counter / (time.time() - frame_count_time)
                frame_counter = 0
                frame_count_time = time.time()

            # Display FPS
            cv2.putText(detected_img, f'FPS: {fps:.2f}', (800, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Display the results
            cv2.imshow("Special CNIC Recognition System", detected_img)

            frame_counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the frame capture thread and close windows
    frame_thread.stop()
    frame_thread.join()
    cv2.destroyAllWindows()