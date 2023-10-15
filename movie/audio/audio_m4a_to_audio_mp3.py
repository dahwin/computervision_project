# from pydub import AudioSegment
#
# # Load the M4A audio file
# audio = AudioSegment.from_file("audio.m4a", format="m4a")
#
# # Export the audio file as MP3
# audio.export("audio.mp3", format="mp3")
from datetime import datetime

# Get the current local date and time
current_datetime = datetime.now()

# Format the date as "month_day_year"
formatted_date = current_datetime.strftime("%B_%d_%Y")

# Format the time with AM/PM format
formatted_time = current_datetime.strftime("%I_%M_%p")

# Combine the formatted date and time
formatted_datetime = f"{formatted_date}_{formatted_time}"

# Print the formatted date and time
print(formatted_datetime)
