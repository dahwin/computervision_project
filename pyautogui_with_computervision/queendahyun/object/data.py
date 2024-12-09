from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests
import torch
# load the processor
processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map='auto'
)

# load the model
model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map='auto'
)


prompt = f"""
# UI Element Detection and Bounding Box Assignment Task

## Objective
Identify and match specific UI elements with their corresponding bounding box IDs in user interface screenshots.

## Input Format
- **Image:** [UI screenshot with numbered bounding boxes]
- **Target UI Element:** [Detailed element specification]

## Required Output
- Single integer representing the correct bounding box ID
- Format: [ID]

## Element Specification Guidelines
1. **Location Details:**
   - Precise spatial position (top, bottom, left, right)
   - Relative positioning to other UI elements
   
2. **Visual Characteristics:**
   - Shape description
   - Icon/symbol details
   - Size information
   - Color properties (if relevant)
   
3. **Functional Description:**
   - Element type (button, icon, menu, etc.)
   - Purpose/functionality
4 . **Don't give any explaination**
   - think deeply
   - just the the output don't give any explaination
   
## Validation Rules
1. The selected bounding box must FULLY contain the target element
2. In case of overlapping boxes, select the most precise fit
3. Consider pixel-perfect alignment
4. Verify element boundaries match bounding box edges

## Example Query
"Identify the bounding box ID containing the extension icon, which appears as a small dark square symbol in the upper portion of the interface. The icon should be completely enclosed within the bounding box boundaries."

## Required Analysis Steps
1. Locate the target element using provided specifications
2. Identify ALL bounding boxes intersecting with the element
3. Verify complete containment
4. Select the most precise containing box
5. Double-check alignment and boundaries
6. Return only the box ID number

## Quality Checks
- Verify full element containment
- Confirm precise boundary alignment
- Validate against overlapping alternatives
- Ensure unique identification

## Output Format
Single integer only:
[Bounding Box ID]

-------------------
**Example:**

**Image:** [Image of a UI with numbered bounding boxes]
**Target Element:** "The dropdown menu located below the search bar"
**Output:** 17  (Assuming bounding box 17 corresponds to the described dropdown)


**Evaluation Criteria:**

The output will be evaluated based on accuracy. The correct bounding box ID must be returned for the given target element description.

-------------Your Targeted Job ------------
**Target Element:** arrow down icon right of Calibri (Body)
**Output:**"""



# ... existing code ...

# Process inputs
image = Image.open("/kaggle/input/sexxxx/image_with_filtered_bboxes (1).png")
inputs = processor.process(images=[image], text=prompt)

# Convert tensors to appropriate types and move to device
processed_inputs = {}
for k, v in inputs.items():
    if k == 'input_ids':
        # Keep input_ids as long integers
        processed_inputs[k] = v.long().to(model.device).unsqueeze(0)
    elif k == 'image_input_idx':
        # Ensure image_input_idx is long type
        processed_inputs[k] = v.long().to(model.device).unsqueeze(0)
    else:
        # Convert other tensors to float16
        processed_inputs[k] = v.to(dtype=torch.float16).to(model.device).unsqueeze(0)

# Use processed inputs for generation
output = model.generate_from_batch(
    processed_inputs,
    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
    tokenizer=processor.tokenizer
)

# only get generated tokens; decode them to text
generated_tokens = output[0,processed_inputs['input_ids'].size(1):]
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

# print the generated text
print(generated_text)
