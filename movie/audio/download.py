from yt_dlp import YoutubeDL
with YoutubeDL({'overwrites':True, 'format':'bestaudio[ext=m4a]', 'outtmpl':'audio.m4a'}) as ydl:
    ydl.download("https://youtu.be/SY2kZjVEd54")

