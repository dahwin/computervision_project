import time

while True:
    current_time = time.localtime() # get the current time as a time struct
    current_second = time.strftime("%S", current_time) # extract the second component as a string
    print(current_second)
    time.sleep(1) # wait for 1 second before printing the next second
