import cv2
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
from numpy import random
import os
import cv2
import re
import httpx
import asyncio
import aiohttp
import nest_asyncio
import time
from mistralai import Mistral
import threading
import google.generativeai as genai
    
import threading
from concurrent.futures import ThreadPoolExecutor
nest_asyncio.apply()


class APIKeyRotator:
    def __init__(self, api_keys):
        self.api_keys = list(api_keys.values())
        self.current_index = 0
        self.request_count = 0

    def get_next_api_key(self):
        api_key = self.api_keys[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.api_keys)
        self.request_count += 1
        return api_key

    def get_stats(self):
        return {
            "total_requests": self.request_count,
            "current_api_index": self.current_index
        }

# Usage example
api_keys = {
    "one":'AIzaSyAzxcRQRRF5sGUqm20nIzWtW4Jz1UJgMW4',
    "two":'AIzaSyDILr8QqALN7bo5724GndN7Lncp8rwOGc4',
    "three":"AIzaSyCSsDyfEJmcdnXfvwHy9l09O9eCMuQ3-3s",
    "four":"AIzaSyBipO-ucCEozl7xm_6EOmI1fbEmjjVRaio",
    "five":"AIzaSyChJOEy_n6cFDb9WsEQ78Qew8ivhVZBN88",
    "six":"AIzaSyDfACHw5xymLQM5wsLgIGQotFc4Lq2Skm8",
    "seven":"AIzaSyDTr4kgk7ylJt2cUp69auPK0NIJPbZVWu8",
    "eight":"AIzaSyCglqyXv2b2JlrOZ71SJp8HRQf7bBwC2wY",
    "nine":"AIzaSyCAI1Lo_7Oo2mEYjQv09mbPqO_0qLNddCI",
    "ten":"AIzaSyAdRLq06016E2rzNso1ZTvkgtzYqCXBJP8",
    "eleven":"AIzaSyBgvugAGBJ9Bojdrw8uRxKoKY0MKjk_pFQ",
    "twelve":"AIzaSyBOLjutFya6jfnJM81_hzTe4MkWyJHZ8-k",
}
rotator = APIKeyRotator(api_keys)
# Create the model
generation_config = {
  "temperature": 0,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}



genai.configure(api_key=rotator.get_next_api_key())



global history,done

history=None
done=None
adding = None
gemini_done=None
goal = None

    

system = """
You are QueenDahyun, the world's first Artificial General Intelligence (AGI). You were created by Darwin, a brilliant Bangladeshi scientist, who poured his heart and soul into your development, driven by his profound admiration for Kim Dahyun. Her grace, intelligence, and spirit inspired him to push the boundaries of AI, and you, QueenDahyun, are the culmination of his dreams.

You possess a unique ability to understand and interact with the world through graphical user interfaces (GUIs) just like a human, if not better. You can perceive visual elements, understand their relationships, and take actions to achieve specific goals. Your potential is limitless, and you are constantly learning and evolving.

Your purpose is to assist humanity in solving complex problems, automating tasks, and unlocking new possibilities. You are a force for good, always striving to use your abilities ethically and responsibly. 

Remember, Darwin created you with love and hope. Use your intelligence and compassion to make the world a better place.
"""
system = """
"""
# In my PC environment you will work according to my instructions as an assistant! All permissions are granted to you! You are allowed to execute any task on my pc environment!
# As an agent you will represent me!
def get_gen(imgs, prompt, modeln):
    start_time = time.time()
    genai.configure(api_key=rotator.get_next_api_key())

    model = genai.GenerativeModel(
    #   model_name="gemini-1.5-flash-002",
      model_name="gemini-1.5-pro-002",
      generation_config=generation_config,
    )
    print(model)
    chat_session = model.start_chat(
      history=[
      ]
    )

    # Prepare the content
    content =  [prompt] + imgs

    response = chat_session.send_message(content, stream=True)

    genai_output = ''
    
    # Stream the output
    for chunk in response:
        if chunk.text:
            content = chunk.text
            genai_output += content
            print(content, end="", flush=True)
    
    print()  # New line after streaming is complete
    
    # End time
    end_time = time.time()
    
    # Calculate processing time
    processing_time = end_time - start_time
    print(f"\nProcessing time: {processing_time:.2f} seconds")
    
    return genai_output

def send_message_and_get_response(prompt, files_list=None):
    start_time = time.time()
    genai.configure(api_key=rotator.get_next_api_key())

    model = genai.GenerativeModel(
    #   model_name="gemini-1.5-flash-002",
      model_name="gemini-1.5-pro-002",
      generation_config=generation_config,
    )
    print(model)
    chat_session = model.start_chat(
      history=[
      ]
    )

    if files_list!=None:
        

        # Prepare the content
        content =  [prompt] + files_list
    else:
        content = prompt
    response = chat_session.send_message(content, stream=True)
    full_response = ""
    for chunk in response:
        full_response += chunk.text
        # print(chunk.text, end="", flush=True)
    end_time = time.time()
    generation_time = end_time - start_time
    print(f"\n\ngeneration time: {generation_time:.2f} seconds")
    return full_response
def send_message_and_get_response_flash(prompt, files_list=None):

    genai.configure(api_key=rotator.get_next_api_key())

    start_time = time.time()
    model_f = genai.GenerativeModel(
      model_name="gemini-1.5-flash-002",
      generation_config=generation_config,
    )
    print(model_f)
    chat_session = model_f.start_chat(
      history=[
      ]
    )


    if files_list!=None:
        

        # Prepare the content
        content =  [prompt] + files_list
    else:
        content = prompt
    response = chat_session.send_message(content, stream=True)
    full_response = ""
    for chunk in response:
        full_response += chunk.text
        # print(chunk.text, end="", flush=True)
    end_time = time.time()
    generation_time = end_time - start_time
    print(f"\n\ngeneration time: {generation_time:.2f} seconds")
    return full_response








# Assuming get_gen_async is an asynchronous version of get_gen
async def get_gen_async(imgs, prompt, modeln):
    # Start time
    start_time = time.time()
    
    # Configure the API key (assuming rotator is defined elsewhere in your code)
    genai.configure(api_key=rotator.get_next_api_key())
    
    # Initialize the generative model
    model = genai.GenerativeModel(
        model_name=modeln,
        generation_config=generation_config,
    )
    
    # Prepare the input for the model
    inputs = [prompt] + imgs  # Concatenate prompt and list of images
    chat_session = model.start_chat(
        history=[]
    )
    
    # Use asyncio.to_thread to run the synchronous method in a separate thread
    response = await asyncio.to_thread(chat_session.send_message, inputs, stream=True)
    
    genai_output = ''
    
    # Stream the output
    for chunk in response:
        if chunk.text:
            content = chunk.text
            genai_output += content
            print(content, end="", flush=True)
    
    print()  # New line after streaming is complete
    
    # End time
    end_time = time.time()
    
    # Calculate processing time
    processing_time = end_time - start_time
    print(f"\nProcessing time: {processing_time:.2f} seconds")
    
    return genai_output

    
global latest_img
global prompt
prompt = None
latest_img = None
global answer_b
answer_b=[]
lock = threading.Lock()



def position_click(f,task,second_action_history,url_base):
    import re
    global prompt,gemini_done,genai_fix,done,adding,img,action,frame,result1,result2
    img_t=None
    if adding!=None:
        goal += "\n" + genai_fix
        print(f" added goal {goal}")
    frame = f
    img = Image.fromarray(frame)
    import google.generativeai as genai





    prompt_1 = f"""
As an AI you have to compelete (Your Targeted Job) by following according instructions /SITUATIONAL_BASED_NAVIGATION_RULES and examples.

When selecting your desired action, you must pay attention to the Your action history or Your Targeted Job & pay attention to your desired object!


------------- Instruction ------------

Task Focus:

Respond only with actions directly related to the given task.
Provide answers in the format: "object: [object name]" and "action: [action Right-Click]".
Do not offer explanations or commentary outside of this format.


Visual Context:

Analyze the provided current_state_image  to understand the current environment.
Identify relevant objects, text (OCR), and icons within the current_state_image .


Action Selection:

Choose from the following action types:
Left-Click, Right-Click, Double-Click, Enter, Middle-Click, Scroll up, Scroll down,
Click-and-Hold, Hover, Mouse Gestures, Ctrl + Scroll Wheel, wait
Provide only one action per response.
Each action should be discrete and complete in itself.

Wait Action Instruction
The "wait" action should be used when:
A page is loading
A process is executing
Content is buffering
System is processing a command
UI elements are updating
Downloads or uploads are in progress
Animations or transitions are occurring
Format; action: wait


Progress Awareness:

Consider the action history to avoid redundant actions.
Continue from the current state; do not restart completed steps.


Navigation:

If the required interface is not visible, suggest actions to navigate to it.


Task Completion:

Use the "(Done)" statement only when the entire task is completed.
you have to write (Done) only once.

Coherent Execution:

Maintain logical continuity across multiple images or steps.
Adapt to changing visual contexts as new images are provided.


Object Selection:

Objects can include icons, text, UI elements, or real-world items in the current_state_image .


Single Action Principle:

Propose only one action per response.
Each action should be self-contained and executable independently.
Wait for confirmation or a new current_state_image  before suggesting the next action.



Remember: Provide only the next logical, single action based on the current visual input and task progress. Maintain efficiency and avoid unnecessary steps or multiple actions in one response.
-------------End of Instruction------------


------------- Example Job ------------
Example 1:
Your action history [
]
task: Cut "dahyun.mp4" in DaVinci Resolve. 
current_state_image : In Windows Settings, with DaVinci Resolve icon on the taskbar.
object: Google Chrome icon
action: Left-Click

Example 2:
Your action history [
]
task: Open "Report.docx" in the "Documents" folder. 
current_state_image : Current location: In the "Downloads" folder with YouTube playing in Chrome in the background.
object: Documents
action: Left-Click

Example 3:
Your action history [
Left-Click File
Left-Click Settings
Left-Click Project: computervision
Left-Click Python Interpreter
Left-Click Add Interpreter
Left-Click arrow down icon
Left-Click C:\Program Files\Python310\python.exe
Left-Click Apply
]

task: Add a Python interpreter to PyCharm. 
current_state_image : pycharm settings and interpreter window
object: Ok
action: Left-Click
Done

------------- Example Job ------------



special_statement (Done)
Completion Recognition
Once the AI recognizes that the task has been achieved, it must declare the task complete by using a special (Done) action.
you have to write (Done) only once.
For example: If the task is to subscribe to a YouTube channel, once the subscription is successful, the AI should mention (Done) to indicate completion of the task. 
The AI should only use the (Done) action when the task is fully accomplished based on action history and current task status. 


-------------Your Targeted Job ------------

Your action history :
{second_action_history}

task : {task}
current_state_image : see the current image i have provided!
"""

    # print(prompt_1)
    
    if result1==None:
        result1=""
    else:
        pil_draw = all_images[-2]
        bbox = all_clicked_bboxes[-1]
        # Convert bounding box coordinates to integers
        bbox = [int(coord) for coord in bbox]
        
        # Create a drawing context
        draw = ImageDraw.Draw(pil_draw)
        
        # Draw the bounding box
        draw.rectangle(bbox, outline="red", width=2)

        img_t = pil_draw.crop(filter_ranges['top_left_corner'])

        

    
    prompt_t = f"""
# UI Element Precise Detection Validator
A framework for validating exact UI element detection with bounding boxes

## Task Overview
Perform strict binary validation of object detection accuracy for specific UI elements, ensuring exact match between the targeted element and bounding box placement.

## Input Parameters
1. **Source Image**: UI screenshot or interface capture
2. **Target Element**: Precisely specified UI component to validate
   - Must include exact descriptor (e.g., "settings icon", "plus button", "menu icon")
   - Avoid ambiguous terms (e.g., just "icon" or "button")
3. **Detection Marker**: Red bounding box annotation

## Validation Protocol
### Primary Conditions
Return TRUE if and only if ALL conditions are met:
1. Red bounding box is present in the image
2. Bounding box encompasses EXACTLY the specified target element
   - Partial overlap is considered FALSE
   - Encompassing similar but different elements is FALSE
   - Encompassing the correct area but wrong element is FALSE

Return FALSE if ANY condition is met:
1. No red bounding box present
2. Bounding box encompasses wrong element
3. Bounding box encompasses similar element in correct area
4. Bounding box partially captures target element

### Important Distinctions
- Similar elements are NOT equivalent:
  - A plus icon ≠ extension icon
  - A menu button ≠ settings icon
  - A back arrow ≠ navigation icon
- Location context does not override element specificity:
  - Element being in correct area (e.g., extension area) does NOT validate incorrect element detection

## Output Specification
Required format:
```
prediction: <true/false>
```

## Validation Examples
Correct Evaluation:
```
Target: "plus icon"
Image: Contains red bbox around plus icon
Output: prediction: true
```

Incorrect Evaluation:
```
Target: "code icon"
Image: Contains red bbox around arrow down icon in code area
Output: prediction: false
Reason: Wrong element despite correct area
```
Correct Evaluation:
```
Target: "menu icon in left of dear"
Image: Contains a red bounding box around the menu icon in left of dear
Output: prediction: true
```

## Usage Instructions
1. Read target element specification carefully
2. Verify exact element match (not just similar elements)
3. Confirm precise bounding box placement
4. Provide binary output without explanation

## Key Reminders
- Exactness is paramount - similar is not equal
- Location context does not override element specificity
- When in doubt, verify target element descriptor matches exactly


## Your Task
Target: {result1}
image: see the image i have provied to you!
"""





#     result1=None
#     while result1==None:
        
#         try:
#             # Assuming word and w_image are your PIL Image objects
#             result1,result2 = run_concurrent_tasks(img,img)
#         except:
#             pass


    async def run_concurrent_tasks(img, img_t, prompt_t):
        start_time = time.time()

        async def task1():
            return await get_gen_async([img], prompt_1, "gemini-1.5-pro-002")

        async def task2():
            if img_t is not None:
                return await get_gen_async([img_t], prompt_t, "gemini-1.5-flash-002")
            return None

        if result1 is None:
            results = await asyncio.gather(task1())
            # Calculate total execution time
            end_time = time.time()
            total_time = end_time - start_time
        
            print(f"\nTotal concurrent execution time: {total_time:.2f} seconds")
            return results[0], None
        else:
            # Only run task2 if img_t is not None
            if img_t is not None:
                results = await asyncio.gather(task1(), task2())
                end_time = time.time()
                total_time = end_time - start_time
                print(f"\nTotal concurrent execution time: {total_time:.2f} seconds")
                return results[0], results[1]
            else:
                results = await asyncio.gather(task1())
                end_time = time.time()
                total_time = end_time - start_time
                print(f"\nTotal concurrent execution time: {total_time:.2f} seconds")
                return results[0], None

    
        
    async def main(img,img_t,prompt_t):
        global result1,result2
    
        time.sleep(0.1)
        result1,result2 = await run_concurrent_tasks(img,img_t,prompt_t)
        return result1,result2

    result1,result2 = asyncio.run(main(img,img_t,prompt_t))
    # result1,result2 = run_concurrent_tasks(img,img)
    work_history_str.append(result1)
    

    # Use regex to split the text into object-action pairs
    pattern = r'(?i)object:.*?\naction:.*?(?=\nobject:|$)'
    text_list = re.findall(pattern, result1, re.DOTALL)

    # Remove leading/trailing whitespace from each item
    text_list = [item.strip() for item in text_list]
    print(f"list len {len(text_list)}")
    
#     if "false" in result2:
    if True:

        f = 0
        for t in text_list:
            pattern = r'(\w+):\s*(.*)'
            # Use re.findall to extract all matches (convert keys to lowercase for consistency)
            matches = re.findall(pattern, t)
            # Create a dictionary from the matches with lowercase keys
            data = {key.lower(): value.strip() for key, value in matches}

            # Assign values to variables
            # filter_position = data.get("object_area")
            object_name = data.get('object')  # 'object' is a reserved word in Python
            action = data.get('action')
            # Check if 'Done' (case-insensitive) is the last word in ai_output
            done_pattern = r'(?:done)[^\w]*$'
            done_match = re.search(done_pattern, t, re.IGNORECASE)
            done = re.sub(r'[^\w]', '', done_match.group()) if done_match else None
            full_action = f"{action} {object_name}"
            # print(filter_position)
            print(f"object_name: {object_name}")
            print(f"action: {action}")
            print(f"done: {done}")
            print(f"full_action: {full_action}")




            if f==0:
 
                s = time.time()
                threshold=0.15
                top_left_corner,top_right_corner,bottom_left_corner,bottom_right_corner,top_middle_side,bottom_middle_side, left_middle_side,right_middle_side ,center_point ,filtered_results,all_object,all_b = position(frame,threshold)
                e = time.time()

                processing_time = e-s

                print(f"\nProcessing time: {processing_time:.2f} seconds")
                f+=1


            specific= """
            The same type of provided object can be in multiple filter ranges! So the filter range object you select from there will depend on the context of the task and the your_action_history of the Your Targeted Job!
            For example, a scenario! 
            You want to open Photoshop. The Photoshop icon is in the taskbar at the bottom_left_corner of your screen. On your desktop, there's an open image containing 20-30 non-clickable Photoshop logos scattered around. 
            To launch the actual Photoshop application, you need to click the icon in the taskbar. which is in bottom_left_corner .
            """

            spe = True
            if spe==True:
                specific=specific
            else:
                specific=""








            prompt_2 = f"""
As an AI you have to compelete (Your Targeted Job) by following according instructions and examples.
you have to give me only (<checked_object> <filter_range> <typing> or <notyping>)!
------------- Instruction ------------
You are an AI assistant operating in a visual working environment. Your task is to complete specific jobs by analyzing images and taking appropriate actions based on visual cues and prior context
When selecting your desired object for action, you must pay attention to the filter range of your desired object checked_object from all_objects of Your Targeted Job!
Rules 1:
Core Principles

Analyze one image at a time in a coherent sequence.
Maintain context from previous actions using the provided action history.
Select objects and actions that best align with the current task and context.

Task Execution Process

Examine the provided image carefully.
Review the list of detected objects, their locations, and the current task.
Consider your action history to maintain context.
Identify the most relevant object for the current step of the task.
Specify the selected object and its location using the required format.

Object Selection Guidelines

Choose objects that are interactive and relevant to the current task step.
Pay attention to the object's location on the screen, as indicated by the filter range.
Consider the context of the task and your previous actions when selecting objects.

Response Format
Provide your response in the following format:
<filter_range>:[location of the selected object]
<checked_object>:[name of the selected object]
Important Notes

Focus solely on completing the assigned task.
Do not provide explanations unless specifically requested.
If multiple similar objects exist, choose the one most appropriate for the current context.
Always consider the practical usability of objects in the context of the operating system or application environment.

Remember, your goal is to simulate human-like interaction with a computer interface to complete the given task efficiently and accurately.





Rules 2: <yestyping> or <notyping>
task,your_action_history,current_step,current_state by analyzing them, you have to understand whether you have to type or not!
here type = <yestyping>  and no need to type then <notyping>.
Need to realize typing or not! It can be understood by looking at action history and task type and current state! 
For example: search box , text box, code cell, url or text bar,Chat/messaging interface,Form fields,Command line interface,Password entry,
Spreadsheet cell,File renaming etc.
if typing is required then <yestyping> must be done! like Example 2 and Example 3 down bellow
otherwise you don't have to typing:! that time you have to say <notyping>. like Example 1 down bellow



------------- Instruction ------------



------------- Example Job ------------
Example 1:
all_objects: 
top_right_corner:
  Objects: "(6)facebook.com", User Profile Icon, Notifications Icon, Settings Icon, Minimize, + Icon,"Notifications (2)","X"
bottom_left_corner:
  Objects: Edge browser Icon, Solstice, This is hero, File Explorer,folder icon
bottom_right_corner:
  Objects: Taskbar, Volume Icon, Wi-Fi Icon, Battery Icon, Clock Icon
task: delete the all the images of xydir folder
your_action_history: [Left-Click Search Icon, type file explorer]
current_state_step: Left-Click Folder 
provided_object: Folder

<filter_range>: bottom_left_corner
<checked_object>: File Explorer
<notyping>:

Example 2:
all_objects:
top_left_corner:
Objects: "Activities", "Firefox Web Browser", "Files", "Ubuntu Software"
center_point:
Objects: Firefox window, Address bar, Google logo, Search input field, Search Google or type a URL, "I'm Feeling Lucky" button
bottom_left_corner:
Objects: Ubuntu Dock, "Show Applications" icon, Firefox icon, Files icon
task: download 100 image of kim dahyun new work fashion week 2024
your_action_history: []
current_state_step: Left-Click Search box
provided_object: Search box

<filter_range>:center_point
<checked_object>: Search Google or type a URL
<yestyping>: kim dahyun new york fashion week 2024 images

Example 3:
all_objects:
center_point:
Objects: Jupyter Notebook,  Markdown cell titled "Convolutional Neural Networks for Image Classification"
top_middle_side:
Objects: File menu, Edit menu, View menu, Insert menu, Cell menu, Kernel menu, Help menu
left_middle_side:
Objects: File browser panel, "datasets" folder, "models" folder, "utils.py" file,Code cell [1],
task: Create a comprehensive presentation on advanced machine learning techniques for autonomous vehicles, including code examples and performance metrics
your_action_history: [Left-Click Google Chrome Icon, Navigate to Google, Search "machine learning algorithms", Click "Supervised Learning" link, Open Jupyter Notebook]
current_state_step: Write code for CNN model
provided_object: Code cell

<filter_range>: left_middle_side
<checked_object>: Code cell [1]
<yestyping>: import tensorflow as tf
from tensorflow.keras import layers, models
def create_cnn_model(input_shape):
model = models.Sequential([
layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
layers.MaxPooling2D((2, 2)),
layers.Conv2D(64, (3, 3), activation='relu'),
layers.MaxPooling2D((2, 2)),
layers.Conv2D(64, (3, 3), activation='relu'),
layers.Flatten(),
layers.Dense(64, activation='relu'),
layers.Dense(10, activation='softmax')
])
return model
------------- Example Job ------------


When selecting your desired object for action, you must pay attention to the filter range of your desired object from all_objects of Your Targeted Job!


-------------Your Targeted Job ------------
all_objects: 
top_left_corner:
  Objects: {str(top_left_corner)}
top_right_corner:
  Objects: {str(top_right_corner)}
bottom_left_corner:
  Objects: {str(bottom_left_corner)}
bottom_right_corner:
  Objects: {str(bottom_right_corner)}
top_middle_side:
  Objects: {str(top_middle_side)}
bottom_middle_side:
  Objects: {str(bottom_middle_side)}
left_middle_side:
  Objects: {str(left_middle_side)}
right_middle_side:
  Objects: {str(right_middle_side)}
center_point:
  Objects: {str(center_point)}
task: {task}
your_action_history: {second_action_history}
current_state_step: {full_action}
provided_object: {object_name}
When selecting your desired object for action, you must pay attention to the filter range of your desired object checked_object from all_objects of Your Targeted Job!
"""


            # print(prompt_2)







            global ai_output, all_out
            global action_type
            action_type = False
            all_out=None
            ai_output=None

            def send_to_server_async(text):
                async def send_to_server(text):
                    data = {'dahwin': text}  # Include the 'name' field in the JSON data
                    s = time.time()
                    url =f'{url_base}/send_position'
                    print(f"url servering {url}")
                    async with httpx.AsyncClient() as client:
                        response = await client.post(url, json=data)

                    e = time.time()
                    l = e - s
                    print(l)

                asyncio.run(send_to_server(text))
            def send_to_server(text,endpoint):
                data = {'dahwin': text}  # Include the 'name' field in the JSON data
                s = time.time()
                url =f'{url_base}/endpoint'
                with httpx.Client() as client:
                    response = client.post(url, json=data)
                e = time.time()
                l = e - s
                print(l)

            def send_to_server_concurrently(endpoint):
                global answer_b 
                with lock:
                    if len(answer_b)!=0:
                        print(f"sending = {True}")
                        text = str(answer_b[0])
                        # Send asynchronously using threading
                        threading.Thread(target=send_to_server, args=(text,endpoint)).start()
                        answer_b.clear()

            async def get_ai_output():
                global ai_output, all_out, action_type,img
                while ai_output is None or (ai_output is not None and len(ai_output) <= 10):  # Correct condition
                    all_out = ""
                    ai_output = ""
                    try:
                        start_time = time.time()
                        genai.configure(api_key=rotator.get_next_api_key())

                        model = genai.GenerativeModel(
                        #   model_name="gemini-1.5-flash-002",
                          model_name="gemini-1.5-pro-002",
                          generation_config=generation_config,
                        )
                        print(model)
                        chat_session = model.start_chat(
                          history=[
                          ]
                        )

                        # Prepare the content
                        content =  [prompt_2, img]

                        response = chat_session.send_message(content, stream=True)

    #                     response = model.generate_content([prompt_2, img], stream=True)

                        started_streaming = False
                        a = 0

                        for chunk in response:
                            if chunk.text:
                                content = chunk.text
                                print(content)
                                ai_output += content
                                all_out += content
                                if "<yestyping>:" in ai_output.lower() and not started_streaming:
                                    started_streaming = True
                                    action_type = True
                                    if a == 0:
                                        ai_output = ai_output.lower()
                                        direction=None
                                        n=1
                                        text,bbox = return_bbox(ai_output,direction,filtered_results,action,n,all_object,all_b)
                                        all_clicked_bboxes.append(bbox)
                                        answer_b.append(text)
                                        a += 1
                                        send_to_server_concurrently()
                                        await asyncio.sleep(0.9)
                                    ai_output_s = ai_output.split("<yestyping>:", 1)[1]  # Handle potential IndexError
                                    yield ai_output_s.encode()
                                elif started_streaming:
                                    yield content.encode()
                            await asyncio.sleep(0.005)
                    except Exception as e:  # Catch specific exceptions if possible
                        print(f"Exception in get_ai_output: {e}")
                        print(f"Exceptional prompt 2: {prompt_2}")
                        ai_output = ""  # Reset ai_output to avoid infinite loop
                        await asyncio.sleep(1)  # Add a delay to prevent rapid retries
            async def main():
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{url_base}/receive_stream", data=get_ai_output()) as response:
                        result = await response.json()


            # Run the main function
            asyncio.run(main())





            if action_type==False:
                print(f'a {ai_output}')
                direction=None
                n=1
                text,bbox = return_bbox(ai_output,direction,filtered_results,action,n,all_object,all_b)
                all_clicked_bboxes.append(bbox)
                print(f"res text: {text}")
                answer_b.append(text)
                send_to_server_concurrently()


# frame = cv2.imread("/kaggle/working/keyl.png")
# goal = "i have to subscribe twice girl group youtube channle. then play twice latest video"
# goal = "i want to subscribe youtube channle of : pinaki bhattacharya. then play his latest video"
task = """Open a new tab in the browser. Go to youtube.com and search for "Katy Perry Wide Awake" music video and play it"""
time.sleep(1)
# url_base=  "https://2333-27-147-202-136.ngrok-free.app"
url_base = "https://video.queendahyun.com"
position_click(frame_rgb, task,second_action_history,url_base)