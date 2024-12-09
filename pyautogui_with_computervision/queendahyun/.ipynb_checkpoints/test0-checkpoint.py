import cv2
import time
import threading

time.sleep(10)

url = "http://localhost:3000/video_feed/"


# Initialize VideoCapture object
cap = cv2.VideoCapture(url)

# Get the default frame size
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output00.avi', fourcc, 10, (frame_width, frame_height))

# Function to continuously read frames and write them to the output video
def read_frames():
    start_time = time.time()
    frame_count = 0
    while frame_count < 100:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break
        frame_count += 1
        out.write(frame)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time taken to get frames:", elapsed_time, "seconds")
    print("Frame count:", frame_count)

# Start reading frames in a separate thread
frame_thread = threading.Thread(target=read_frames)
frame_thread.start()

# Wait for the frame thread to finish
frame_thread.join()

# Release everything when done
cap.release()
out.release()
