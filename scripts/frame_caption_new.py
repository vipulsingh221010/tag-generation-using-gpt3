from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import sys
import time
from tqdm import tqdm
import json
import os
import re
import numpy as np
import urllib.parse
data = {}
st = int(sys.argv[1])
en = int(sys.argv[2])
print(st)
print(en)
#torch.cuda.set_per_process_memory_fraction(0.5)
device ='cuda'

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

# device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float32
)
model.to(device)
x=0
path = '/home/intern/interndata/vipul/video-verbalization-main/scripts/frame_extraction/ucf_frames/'

with open('/home/intern/interndata/vipul/video-verbalization-main/scripts/frame_extraction/ucf_labels.json') as file:
    labels=json.load(file)

for video_file in os.listdir(path):
    start_time = time.time()

    
    x+=1
    print(x)
    dir=video_file
    print(dir)
    caption = []
    objects = []
    colors = []
    summary = []
    file_path = os.path.join(path, dir)
    if not os.path.isdir(file_path):
        continue
    data[dir] = {}
    try :
        files = list(sorted_alphanumeric(os.listdir(file_path)))
    #print(files)
    #torch.cuda.empty_cache()
        for i in range(0, len(files)):
            imgs = files[i]
            image = Image.open(path+dir+'/'+imgs).convert('RGB')
                #print(np.mean(image))
            if(np.mean(image)>=10):
                with torch.no_grad():
                    prompt = "Can you caption this image?"
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float32)
                    generated_ids = model.generate(**inputs)
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                    caption.append(generated_text)
                    torch.cuda.empty_cache()
        print(caption)

        data[dir]['caption'] = caption
        data[dir]['truth_label']=labels[dir]['truth_label']
        with open('caption_ucf'+str(st)+'.json', 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(e)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken for iteration {x}: {elapsed_time} seconds")

with open('caption_ucf'+str(st)+'.json', 'w') as f:
    json.dump(data, f)
