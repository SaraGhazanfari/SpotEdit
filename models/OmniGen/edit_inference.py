# Prediction interface for Cog ⚙️
# https://cog.run/python

import os, json
import subprocess
import time
import sys, argparse
from typing import List
from utils import read_ann_file
from cog import BasePredictor, Input, Path
from PIL import Image

sys.path.insert(0, "OmniGen")
from OmniGen import OmniGenPipeline


MODEL_CACHE = "model_cache"
MODEL_URL = (
    f"c"
)

os.environ.update(
    {
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": MODEL_CACHE,
        "TORCH_HOME": MODEL_CACHE,
        "HF_DATASETS_CACHE": MODEL_CACHE,
        "TRANSFORMERS_CACHE": MODEL_CACHE,
        "HUGGINGFACE_HUB_CACHE": MODEL_CACHE,
    }
)


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def __init__(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        self.pipe = OmniGenPipeline.from_pretrained("/scratch/sg7457/code/SpotEdit/saved_models/OmniGen1")

    def predict(
        self, prompt:str, image_list:List,
        width=1024, height=1024, 
        inference_steps=50,
        guidance_scale=2.5,
        img_guidance_scale=1.6,
        seed=None,
        max_input_image_size=1024,
        separate_cfg_infer=True,
        offload_model=False,
        use_input_image_size_as_output=False,
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        input_images = [str(img) for img in image_list if img is not None]
        output = self.pipe(
            prompt=prompt,
            input_images=None if len(input_images) == 0 else input_images,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            img_guidance_scale=img_guidance_scale,
            num_inference_steps=inference_steps,
            separate_cfg_infer=separate_cfg_infer,
            use_kv_cache=True,
            offload_kv_cache=True,
            offload_model=offload_model,
            use_input_image_size_as_output=use_input_image_size_as_output,
            seed=seed,
            max_input_image_size=max_input_image_size,
        )
        return output[0]

def main(args):
    
    root_out_image_path = f'/scratch/sg7457/dataset/spotedit/generated_images/{args.mode}/omnigen'
        
    spotedit_list = read_ann_file(args.mode)
    inferencer = Predictor()
    start_idx = 0
    for item_idx, item in enumerate(spotedit_list[start_idx:]):
                        
        output_image_path = os.path.join(root_out_image_path, str(item['id']), item['image_list'][-1].split('/')[-1])

        print(f'{item_idx+start_idx}/{len(spotedit_list)}', output_image_path)

        if os.path.exists(output_image_path):
            continue

        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        prompt = item['prompt'].replace('first image', '<img><|image_1|></img>').replace('second image', '<img><|image_2|></img>')
    
        ret = inferencer.predict(prompt=prompt, image_list=item['image_list'][:2])
        ret.save(output_image_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Define arguments
    parser.add_argument(
        "--mode",
        type=str,
        choices=["real", "syn", "dreamedit"],  # restrict allowed values
        required=True
    )
    args = parser.parse_args()
    main(args)
