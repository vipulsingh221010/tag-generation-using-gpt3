from dotenv import load_dotenv
from adobellm.util.azureopenai_api_call import OpenAIAPI
import os
import json
import traceback

load_dotenv('config_gpt.sh')
openapi = OpenAIAPI(loglevel="info")
engine = os.environ.get("OPENAI_AZURE_DEPLOYMENT_GPT35_TURBO", "")

with open("/home/intern/interndata/vipul/video-verbalization-main/scripts/caption_ucf1.json") as f:
    data = json.load(f)

with open("/home/intern/interndata/vipul/video-verbalization-main/scripts/frame_extraction/ucf_labels.json") as f:
    label=json.load(f)

header = ""

results={}
cnt = 0
for name in data:
    try:
        d={}
    
        captions=data[name]['caption']

        prompt = "Generate tags which represent the whole video based on the video captions provided in sequence:\n\n"
        for i, caption in enumerate(captions, start=1):
            prompt += f" {i}: {caption}\n"
        prompt += "\nNote: The captions are arranged in sequential order representing the temporal component of the video.\n"
        #print(prompt)
        rst_gpt35 = openapi.make_chat_call(header, prompt, engine, max_tokens=1000)
        if rst_gpt35 != "":
            
            print(label[name])
            tag_list = rst_gpt35.split(", ")
            print(tag_list)
            d['tags']=tag_list
            d['truth_label']=label[name]
            #rows.append({"id": name, "url": url, "prompt": prompt, "story": rst_gpt35})
            cnt += 1
            print(cnt)
            results[name]=d
            
    except:
        traceback.print_exc()



with open("tags_gpt35_ucf.json", "w") as f:
    json.dump(results, f)
