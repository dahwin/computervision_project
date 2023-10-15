import cv2

# Open the video file
cap = cv2.VideoCapture(r"D:\video\Video\Video\twice\twice.mp4")

# Define a function to calculate the variance of an image
def calculate_variance(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate the variance of the grayscale image
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance

# Initialize variables
previous_frame = None
stable_frames = []

# Loop through each frame of the video
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    # If the frame could not be read, break out of the loop
    if not ret:
        break

    # Calculate the variance of the current frame
    variance = calculate_variance(frame)

    # If the variance of the current frame is less than a threshold
    if variance < 1000:
        # If this is the first stable frame or the current frame is significantly different from the previous stable frame
        if previous_frame is None or cv2.absdiff(frame, previous_frame).mean() > 5:
            # Add the current frame to the list of stable frames
            stable_frames.append(frame)
            # Update the previous frame
            previous_frame = frame

# Release the video file and close all windows
cap.release()
cv2.destroyAllWindows()

# Save the stable frames as images
for i, frame in enumerate(stable_frames):
    cv2.imwrite(f'frame_{i}.jpg', frame)
