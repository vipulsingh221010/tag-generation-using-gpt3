from test_p import find_flow_batch
import inspect
import cv2
import os
import numpy as np
import torch
import traceback
from gmflow.gmflow.gmflow import GMFlow
import argparse
from tqdm import tqdm
import functools
import torch
import torchvision
import json
import torchvision.models as models
import torchvision.transforms as transforms

argparser = argparse.ArgumentParser()
#argparser.add_argument("--input_path", type=str, default="", help="path to videos")
argparser.add_argument("--output_path", type=str, default="/home/intern/interndata/vipul/video-verbalization-main/scripts/frame_extraction/ucf_frames", help="path to outputs")
argparser.add_argument("--jumper", type=int, default=20, help="jumper for optical flow")
argparser.add_argument("--intensityThresh", type=int, default=30, help="intensity threshold for optical flow")
argparser.add_argument("--device", type=str, default="cuda", help="device to use")
argparser.add_argument("--ckpt", type=str, default="gmflow_sintel-0c07dcb3.pth", help="checkpoint to use")
argparser.add_argument("--heuristic", type=str, default="max", help="Take max or min velocity frame in a clip")
argparser.add_argument("--vggThresh", type=float, default=0.3, help="threshold for vgg similarity")

args = argparser.parse_args()
device=args.device
model = GMFlow(
    feature_channels=128,
    num_scales=1,
    upsample_factor=8,
    num_head=1,
    attention_type="swin",
    ffn_dim_expansion=4,
    num_transformer_layers=6,
).to(device)

vgg16 = models.vgg16(pretrained=True).to(device)
vgg16.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

checkpoint = torch.load(args.ckpt)  # map_location=device

weights = checkpoint["model"] if "model" in checkpoint else checkpoint

model.load_state_dict(weights, strict=False)
model.eval()
heuristic = np.argmax if args.heuristic == "max" else np.argmin


def get_video_frame_skipped_from_path(video, jumper):
    '''
    Given a video path and a jumper, return the frames and the frame indexes skipped by the jumper value (e.g. jumper=4 means fps//4 frames are skipped)
    '''
    fps = video.get(cv2.CAP_PROP_FPS)
    jump = fps // jumper  # take 4 frames at 0.25 second interval for optical flow calculation
    frames = []
    frames_idxs = []
    # print(f"jump is {jump}")
    len_video = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    jump=max(jump,len_video//75)
    print(f"jump is {jump}")
    for frame_idx in range(0, int(len_video), int(jump)):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video.read()
        if ret is False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (512, 256), cv2.INTER_CUBIC)
        frames.append(frame)
        frames_idxs.append(frame_idx)
    return np.array(frames_idxs), np.array(frames)


def get_vgg_sims(frames):
    '''
    Given the frames and frame indexes, return the vgg embeddings for each frame
    '''
    # Convert the numpy array to a tensor
    batch = torch.from_numpy(frames).float().to(device)
    batch = batch.permute(0, 3, 1, 2)

    # Apply the transformations to the batch of images
    batch_transformed = torch.stack([transform(img) for img in batch])

    # Pass the batch of images through the vgg16 model to get the embeddings
    with torch.no_grad():
        embeddings = vgg16(batch_transformed.to(device))

    # Flatten the embeddings to have shape [batch_size, num_features]
    embeddings = embeddings.view(embeddings.size(0), -1)
    sims = torch.nn.functional.cosine_similarity(embeddings[:-1], embeddings[1:], dim=1).to(device)
    return sims.cpu().numpy()


def get_intensity_velocities(frames_idxs, frames):
    '''
    Given a video path and a jumper, return the intensity and velocity changes between frames, for all frames skipped by the jumper value
    '''

    sims = get_vgg_sims(frames)
    ints = np.mean(frames, axis=(1, 2))
    rInt, gInt, bInt = ints[:, 0], ints[:, 1], ints[:, 2]
    rInt1, rInt2 = rInt[:-1], rInt[1:]
    gInt1, gInt2 = gInt[:-1], gInt[1:]
    bInt1, bInt2 = bInt[:-1], bInt[1:]
    intChange = np.sqrt(
        (rInt1 - rInt2) ** 2 + (gInt1 - gInt2) ** 2 + (bInt1 - bInt2) ** 2
    )
    frames = np.array([frames[:-1], frames[1:]]).transpose(1, 0, 2, 3, 4)
    flows = find_flow_batch(model, frames, device=device)
    function_file = inspect.getfile(find_flow_batch)
    #print(function_file)
    flows_uv = flows.transpose(0, 2, 3, 1)
    velocities = np.mean(np.abs(flows_uv), axis=(1, 2, 3))
    #print(len(velocities))
    return {
        "idxs": frames_idxs.tolist(),
        "intensities": intChange.tolist(),
        "vgg_changes": (1 - sims).tolist(),
        "velocities": velocities.tolist()
    }


def save_frames(frames_idxs, intChange, vgg_changes, velocities, video, video_path):
    '''
    Given the intensity, velocity, idxs all with the same index as flow_out. Break the frames into clips wherever intensity > thresh. And for each clip find the frame_idx with the highest velocity. save that frame from the video
    '''
    f_idxs = []
    temp_frames, temp_vel = [], []
    #print(len(frames_idxs))
    for idx, intc, vggc, vel in zip(frames_idxs, intChange, vgg_changes, velocities):
        temp_frames.append(idx), temp_vel.append(vel)
        if intc > args.intensityThresh or vggc > args.vggThresh:
            f_idxs.append(temp_frames[heuristic(temp_vel)])
            temp_frames, temp_vel = [], []

    if len(temp_frames) > 0:
        f_idxs.append(temp_frames[heuristic(temp_vel)] + 1)
    print(len(f_idxs))
    for idx in f_idxs:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        _, frame = video.read()
        #print(video_path)

        # cv2.imwrite(os.path.join(args.output_path, video_path.split("_")[0] + "_%d.jpg" % idx), frame)
        cv2.imwrite(os.path.join(args.output_path + '/' + video_path, video_path + "_%d.jpg" % idx), frame)
    

# if __name__ == "__main__":
#     videos = [vid for vid in os.listdir(args.input_path)]
#     for video_path in tqdm(videos):
#         try :
#             if video_path.split(".")[-1] != "mp4":
#                 continue

#             video = cv2.VideoCapture(os.path.join(args.input_path, video_path))

#             if os.path.exists(os.path.join(args.output_path, video_path.split("_")[0] + ".json")):
#                 flow_out = json.load(open(os.path.join(args.output_path, video_path.split("_")[0] + ".json"), "r"))
#             else:

#                 frames_idxs, frames = get_video_frame_skipped_from_path(video, args.jumper)
#                 flow_out = get_intensity_velocities(frames_idxs, frames)
#                 json.dump(flow_out, open(os.path.join(args.output_path, video_path.split("_")[0] + ".json"), "w"))
#                 save_frames(flow_out["idxs"], flow_out["intensities"], flow_out["vgg_changes"], flow_out["velocities"], video, video_path)

#         except RuntimeError as e:
#             if 'CUDA out of memory' in str(e):
#                 torch.cuda.empty_cache()
#                 with open("log_cuda_oom.txt", "a") as f:
#                     f.write(video_path+"\n")
#             else:
#                 with open("log_other.txt", "a") as f:
#                     f.write(video_path+str(e)+"\n")

#         except Exception as e:
#             with open("log_other.txt", "a") as f:
#                 f.write(video_path+str(e)+"\n")
num=0
labels={}
if __name__ == "__main__":
    test_set_dir="/home/intern/interndata/vipul/UCF-101"
    for class_folder in os.listdir(test_set_dir):
        class_folder_path = os.path.join(test_set_dir, class_folder)
        x=0
        for video_file in os.listdir(class_folder_path):

            video_name = os.path.splitext(video_file)[0]
            #ground_truth_labels[video_name] = class_folder
            video_path = os.path.join(class_folder_path, video_file)
            torch.cuda.empty_cache()
            try:
                
            
                if x==50:
                    break
                x+=1 
                print(f"count is {num}")
                num+=1
                video = cv2.VideoCapture(video_path)
                video_new_path = video_path[:video_path.rindex('.')]
                #print(video_new_path)
                os.makedirs(os.path.join(args.output_path, video_name), exist_ok=True)
                #print(video_new_path)
                if os.path.exists(os.path.join(args.output_path, video_name + ".json")):
                    print(args.output_path)
                    flow_out = json.load(open(os.path.join(args.output_path, video_name + ".json"), "r"))
                else:
                    with torch.no_grad():

                        #print(f"count is {num}")
                        frames_idxs, frames = get_video_frame_skipped_from_path(video, args.jumper)
                        print(f"number of frames {len(frames)}")
                        flow_out = get_intensity_velocities(frames_idxs, frames)
                        #print(len(flow_out))

                        json.dump(flow_out, open(os.path.join(args.output_path, video_name + "fr.json"), "w"))

                        save_frames(flow_out["idxs"], flow_out["intensities"], flow_out["vgg_changes"],
                                 flow_out["velocities"],
                                video, video_name)
                        labels[video_name]=class_folder
                    # save_frames(flow_out["idxs"], flow_out["intensities"], flow_out["velocities"], video, video_new_path)

            except Exception as e:
                traceback.print_exc()
                if 'CUDA out of memory' in str(e):
                    torch.cuda.empty_cache()
                    with open("log_cuda_oom.txt", "a") as f:
                        f.write(video_file + device + "\n")
                continue
    
    with open("ucf_labels.json", "w") as file:
            json.dump(labels, file)
            #if num==100:
               # break
       # if num==100:
            #break


