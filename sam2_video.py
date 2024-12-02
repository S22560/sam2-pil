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

image_path = "../images3"
output_path = "../output3"
fps = 5

os.makedirs(output_path,exist_ok=True)

images = sorted(glob.glob(f"{image_path}/*.jpg"))
output_video_path = f"{output_path}/raw_video.mp4"
clip = ImageSequenceClip(images, fps=fps)
clip.write_videofile(output_video_path, codec="libx264")
boxes = np.stack([i[0] for i in detect_person(images[0])])

checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
device = "cuda"
predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)

pil_images = [Image.open(i) for i in images]

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    inference_state = predictor.init_state(pil_images)
    predictor.reset_state(inference_state)

    ann_frame_idx = 0 
    for idx,box in enumerate(boxes):
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=idx+1,
            box=box
        )

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    for i, image in enumerate(images):
        # ビデオフレームごとのマスクを取得
        if i in video_segments:
            frame_segments = video_segments[i]
            # 複数のオブジェクトIDがある場合を考慮してマスクを結合
            combined_mask = None
            for obj_id, segment_mask in frame_segments.items():
                if combined_mask is None:
                    combined_mask = segment_mask[0]
                else:
                    combined_mask |= segment_mask[0]
            # 元の画像を読み込む
            original_img = np.array(Image.open(image).convert("RGB"))
            
            # マスクの形状を確認し、一致させる（必要であればリサイズ）
            if combined_mask.shape != original_img.shape[:2]:
                from skimage.transform import resize
                combined_mask = resize(combined_mask, original_img.shape[:2], order=0, preserve_range=True).astype(bool)
            # マスクを元画像に適用 (マスク部分は色を変更)
            output_img = original_img.copy()
            output_img[~combined_mask] = [0, 255, 0]  # マスク部分を赤色にする例
            
            # 出力するためのコード（保存する場合）
            Image.fromarray(output_img).save(f"{output_path}/masked_frame_{i:04d}.png")
    

    # 保存された画像を動画として結合する
    output_video_path = f"{output_path}/masked_video.mp4"
    output_images = sorted(glob.glob(os.path.join(output_path, "masked_frame_*.png")))

    clip = ImageSequenceClip(output_images, fps=fps)
    clip.write_videofile(output_video_path, codec="libx264")