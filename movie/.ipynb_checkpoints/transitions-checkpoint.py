'''dahyun+darwin= dahwin'''

# from moviepy.editor import VideoFileClip
# from moviepy.editor import VideoFileClip, clips_array, vfx
# from moviepy.video import fx
# x = dir(VideoFileClip)
#
# print('\n'.join(x))


from moviepy.editor import *

clips = [VideoFileClip('fancy.mp4'), VideoFileClip('feel.mp4'), VideoFileClip('cv2imagesfacny.mp4')]
padding = 2
video_fx_list = [clips[0]]

idx = clips[0].duration - padding
for video in clips[1:]:
    video_fx_list.append(video.set_start(idx).crossfadeout(padding))
    idx += video.duration - padding

final_video = CompositeVideoClip(video_fx_list)
final_video.write_videofile('crossfade.mp4', fps=24) # add any remaining params







