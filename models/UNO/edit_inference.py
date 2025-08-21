# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from PIL import Image
import json

import os

from uno.flux.pipeline import UNOPipeline
from utils import read_ann_file

def edit_image(spotedit_list, root_out_image_path, pipeline):

    start_idx = 0
    for item_idx, item in enumerate(spotedit_list[start_idx:]):
        
        input_images = [
                        Image.open(item['image_list'][0]).convert("RGB"), # Ref image
                        Image.open(item['image_list'][1]).convert("RGB"), # Input image  
                    ]
                        
        output_image_path = os.path.join(root_out_image_path, str(item['id']), item['image_list'][-1].split('/')[-1])

        print(f'{item_idx+start_idx}/{len(spotedit_list)}', output_image_path, flush=True)

        if os.path.exists(output_image_path):
            continue

        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        ret = pipeline(prompt=item['prompt'], ref_imgs=input_images)
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
    root_out_image_path = f'/scratch/sg7457/dataset/spotedit/generated_images/{args.mode}/uno'
        
    # args_tuple = parser.parse_args_into_dataclasses() 
    # args = args_tuple[0]

    pipeline = UNOPipeline(model_type='flux-dev', device='cuda:0', offload=False, only_lora=True, lora_rank=512)
    
    edit_image(read_ann_file(args.mode), root_out_image_path=root_out_image_path, pipeline=pipeline)






    
