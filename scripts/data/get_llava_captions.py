import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

# Load existing captions if they exist
captions_file = "captions.json"
if os.path.exists(captions_file):
    with open(captions_file, "r") as f:
        results = json.load(f)
    print(f"Loaded existing captions for {len(results)} images")
else:
    results = {}
    print("Starting with empty captions dictionary")

# Load model and processor
model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(0)
processor = AutoProcessor.from_pretrained(model_id)

# Load image paths
image_dir = "/projects/bdbk/cherd/rendersynth/data/sat_images"
image_files = sorted([
    os.path.join(image_dir, fname)
    for fname in os.listdir(image_dir)
    if fname.endswith(".jpeg") or fname.endswith(".jpg") or fname.endswith(".png")
])

# Filter out images that already have captions
images_to_process = []
for image_path in image_files:
    frame_id = os.path.basename(image_path).replace(".jpeg", "").replace(".jpg", "").replace(".png", "")
    if frame_id not in results:
        images_to_process.append(image_path)

print(f"Found {len(image_files)} total images")
print(f"Need to process {len(images_to_process)} new images")

if not images_to_process:
    print("All images already have captions!")
else:
    prompt = "Briefly describe the main features visible in this satellite image."
    
    for image_path in tqdm(images_to_process, desc="Processing images"):
        frame_id = os.path.basename(image_path).replace(".jpeg", "").replace(".jpg", "").replace(".png", "")
        
        # Prepare the conversation for image captioning
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            }
        ]
        
        prompt_tokens = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )
        
        image = Image.open(image_path).convert("RGB")
        inputs = processor(
            text=prompt_tokens,
            images=image,
            return_tensors="pt"
        ).to(model.device, torch.float16)
        
        output = model.generate(**inputs, max_new_tokens=76)
        decoded = processor.decode(output[0][2:], skip_special_tokens=True)
        
        # Clean up the output
        caption = decoded.strip()
        prefix = prompt + "assistant"
        if caption.startswith(prefix):
            caption = caption[len(prefix):].strip()
        
        results[frame_id] = caption
        
        # Save incrementally every 10 images to avoid losing progress
        if len([k for k in results.keys() if k == frame_id or frame_id in images_to_process[:images_to_process.index(image_path)+1]]) % 10 == 0:
            with open(captions_file, "w") as f:
                json.dump(results, f, indent=2)

# Final save
with open(captions_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"Finished! Total captions: {len(results)}")