import torch
import numpy as np
from PIL import Image
import os
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
import matplotlib.pyplot as plt
from imgutils.detect import detect_person
import glob
from moviepy.editor import ImageSequenceClip
import argparse
import hashlib
import pickle
import cv2
from scenedetect import detect, ContentDetector
from tqdm import tqdm

def hash_text(text):
    encoded_text = text.encode("utf-8")
    hash_object = hashlib.sha256(encoded_text)
    hash_hex = hash_object.hexdigest()
    return str(hash_hex)


checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
device = "cuda"
predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)

def sam2inference(images,image_paths,boxes):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = predictor.init_state(video_path=images)
        predictor.reset_state(inference_state)

        
        for idx,box in enumerate(boxes):
            _, _, _ = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=idx+1,
                box=box,
            )

        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        predictor.reset_state(inference_state)
        del inference_state
        video_frames = []
        pre_frame,pre_frame_alpha = None,None
        for i, image in enumerate(image_paths[len(images)-len(video_segments):]):
            current_frame = image
            # ビデオフレームごとのマスクを取得
            if i in video_segments:
                frame_segments = video_segments[i]
                combined_mask = None
                for obj_id, segment_mask in frame_segments.items():
                    if combined_mask is None:
                        combined_mask = segment_mask[0]
                    else:
                        combined_mask |= segment_mask[0]
                original_img = image
                
                if combined_mask.shape != original_img.shape[:2]:
                    from skimage.transform import resize
                    combined_mask = resize(combined_mask, original_img.shape[:2], order=0, preserve_range=True).astype(bool)
                output_img = np.concatenate([original_img,combined_mask.reshape(*combined_mask.shape,1)*255],axis=-1).astype(np.uint8)
            else:
                original_img = image
                combined_mask = np.zeros(original_img.shape[:2])
                output_img = np.concatenate([original_img,combined_mask.reshape(*combined_mask.shape,1)*255],axis=-1).astype(np.uint8)
            if i>1:
                channel_mse = np.max((current_frame-pre_frame)**2,axis=-1)>100
                alpha = (combined_mask+pre_frame_alpha)>0
                mse = channel_mse[alpha].sum()/alpha.sum()

            if i==1 or (i>1 and mse>1e-4):
                video_frames.append(output_img)
                # Image.fromarray(output_img).save(f"{output_path}/{i:05d}.png")
            pre_frame = original_img
            pre_frame_alpha = combined_mask
    return video_frames

parser = argparse.ArgumentParser()

# 引数の設定
parser.add_argument("--video_path", type=str)
parser.add_argument("--output_dir", type=str)

# 引数をパース
args = parser.parse_args()

video_path = args.video_path
output_dir = args.output_dir
hash_dir = "hash_scene_list"

# 出力フォルダを作成
os.makedirs(output_dir, exist_ok=True)
os.makedirs(hash_dir, exist_ok=True)

scene_list_path = os.path.join(hash_dir, hash_text(video_path) + ".pkl")
if os.path.exists(scene_list_path):
    scene_list = pickle.load(open(scene_list_path,"rb"))
else:
    scene_list = detect(video_path, ContentDetector(), show_progress=True)
    pickle.dump(scene_list,open(scene_list_path,"wb"))

# 動画を読み込む
cap = cv2.VideoCapture(video_path)

# 動画情報を取得
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"動画のフレーム数: {frame_count}, FPS: {fps}")

os.makedirs(output_dir,exist_ok=True)
previous_mp4 = sorted(os.listdir(output_dir))
if len(previous_mp4)==0:
    last_generated_data = None
else:
    last_generated_data = max(int(i.replace(".mp4","").replace("scene_","")) for i in previous_mp4)

# ==== 各シーンごとに全フレームを保存 ====
for scene_index, scene in enumerate(scene_list):
    if scene_index <= last_generated_data:
        continue
    scene_start_frame = scene[0].get_frames()
    scene_end_frame = scene[1].get_frames()-1
    print(f"シーン{scene_index+1}: 開始フレーム {scene_start_frame}, 終了フレーム {scene_end_frame}")

    # シーン内の全フレームを保存
    frames = []
    for frame_number in tqdm(range(scene_start_frame, scene_end_frame + 1),"frame"):
        # 動画の現在のフレームを設定
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # フレームを取得
        ret, frame = cap.read()
        if not ret:
            print(f"フレーム {frame_number} を取得できませんでした。スキップします。")
            continue
        frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

    if last_generated_data is not None:
        if scene_index<=last_generated_data:
            continue
            
    try:
        video_filepath = os.path.join(output_dir,f'{scene_index:05d}.mp4')

        images_raw = frames
        images = [images_raw[0],images_raw[0]] + images_raw
        images_pil = [Image.fromarray(i) for i in images]
        if os.path.exists(video_filepath) or len(images_raw)<16:
            continue
        for ann_frame_idx in range(len(images_pil)):
            persons = detect_person(images_pil[ann_frame_idx])
            if len(persons)!=0:
                break
        if len(persons)==0:
            continue
        boxes = np.stack([i[0] for i in persons])
        video_frames = sam2inference(images_pil,images,boxes)

        if len(video_frames)>16:
            video = ImageSequenceClip(video_frames, fps=10)
            video.write_videofile(video_filepath, codec='libx264', bitrate="8000k")
    except KeyboardInterrupt:
        break