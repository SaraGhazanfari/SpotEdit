import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
import os, json
import shutil

pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")

root_path = '/home/colligo/SpotFrame/spotframe_real'
ann_file = 'spotframe_real.jsonl'
spotframe_real = list()

with open(os.path.join(root_path, ann_file)) as file:
    for line in file.readlines():
        spotframe_real.append(json.loads(line))

for item_idx, item in enumerate(spotframe_real[39:]):
    print(item_idx, item['video_id'])
    if item['level'] != -1:
        continue
    frame_name_list = sorted(os.listdir(os.path.join(root_path, item['video_id'].split('_')[0])))
    frame_name_list.remove('.DS_Store')

    for frame_idx in item['final_frame_list']:
        frame_path = os.path.join(root_path, item['video_id'].split('_')[0], frame_name_list[frame_idx])
        frame = load_image(frame_path)
        out_frame_path = os.path.join("/home/colligo/SpotFrame/imgs", item['video_id'], frame_name_list[frame_idx])
        if os.path.exists(out_frame_path):
            continue
        image = pipe(
              image=frame,
              prompt=item['prompt'],
              guidance_scale=2.5
            ).images[0]

        os.makedirs(os.path.dirname(out_frame_path), exist_ok=True)
        image.save(out_frame_path)
        
        

        

    


