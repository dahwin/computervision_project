'''dahyun+darwin = dahwin'''
import datetime
from PIL import ImageGrab
import numpy as np
import cv2
from win32api import GetSystemMetrics

width = GetSystemMetrics(0)
height = GetSystemMetrics(1)
time_stamp = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
print(time_stamp)
file_name = f'{time_stamp}.mp4'
fourcc = cv2.VideoWriter_fourcc('m' , 'p' , '4' , 'v')
captured_video = cv2.VideoWriter(file_name , fourcc , 10.0 , (width , height))
import time
start_time = time.time()
frame_count = 0
while True:
    img = ImageGrab.grab()# bbox=(0, 0, width, height)
    img = np.array(img)
    img_final = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img_final)
    # cv2.imshow('Dawin', img_final)
    captured_video.write(img_final)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    frame_count += 1
    if time.time() - start_time >= 5:
        break

# Calculate FPS
fps = frame_count / (time.time() - start_time)
print("Estimated FPS:", fps)

# Release the VideoWriter and close all windows
captured_video.release()
cv2.destroyAllWindows()

# with sound
# import datetime
# import numpy as np
# import cv2
# import pyaudio
# import wave
# from PIL import ImageGrab
# from win32api import GetSystemMetrics
#
# # Set up audio recording parameters
# CHUNK = 1024  # Number of audio samples per frame
# FORMAT = pyaudio.paInt16  # Audio format
# CHANNELS = 1  # Mono audio
# RATE = 44100  # Sample rate (Hz)
# RECORD_SECONDS = 10  # Duration of audio recording (seconds)
# WAVE_OUTPUT_FILENAME = 'output.wav'  # Output audio file name
#
# # Set up video recording parameters
# width = GetSystemMetrics(0)
# height = GetSystemMetrics(1)
# time_stamp = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
# print(time_stamp)
# file_name = f'{time_stamp}.mp4'
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# captured_video = cv2.VideoWriter(file_name, fourcc, 10.0, (width, height))
#
# # Set up audio recording stream
# audio = pyaudio.PyAudio()
# stream = audio.open(format=FORMAT, channels=CHANNELS,
#                     rate=RATE, input=True,
#                     frames_per_buffer=CHUNK)
#
# # Start recording
# frames = []
# for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#     data = stream.read(CHUNK)
#     frames.append(data)
#
# # Stop recording and save audio file
# stream.stop_stream()
# stream.close()
# audio.terminate()
# wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
# wf.setnchannels(CHANNELS)
# wf.setsampwidth(audio.get_sample_size(FORMAT))
# wf.setframerate(RATE)
# wf.writeframes(b''.join(frames))
# wf.close()
#
# # Start screen recording
# while True:
#     img = ImageGrab.grab()  # bbox=(0, 0, width, height)
#     img = np.array(img)
#     img_final = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     cv2.imshow('Dawin', img_final)
#     captured_video.write(img_final)
#
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
#
# # Clean up
# cv2.destroyAllWindows()
# captured_video.release()


# with webcam
# import datetime
# from PIL import ImageGrab
# import numpy as rt
# import cv2
# from win32api import GetSystemMetrics
#
# width = GetSystemMetrics(0)
# height = GetSystemMetrics(1)
# time_stamp = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
# print(time_stamp)
# file_name = f'{time_stamp}.mp4'
# fourcc = cv2.VideoWriter_fourcc('m' , 'p' , '4' , 'v')
# captured_video = cv2.VideoWriter(file_name , fourcc , 20.0 , (width , height))
#
# ratul_webcam = cv2.VideoCapture(0)
#
# while True:
#     img = ImageGrab.grab(bbox=(0, 0, width, height))
#     img_rt = rt.array(img)
#     img_final = cv2.cvtColor(img_rt, cv2.COLOR_BGR2RGB)
#     try:
#         _, frame = ratul_webcam.read()
#         fr_height, fr_width, _ = frame.shape
#         img_final[0:fr_height, 0:fr_width:] = frame[0:fr_height, 0:fr_width, :]
#         cv2.imshow('ratul_webcam', frame)
#     except:
#         None
#     cv2.imshow('Dahwin',img_final)
#     captured_video.write(img_final)
#
#     if cv2.waitKey(10) == ord(("r")):
#         break
