import cv2, os, json
from diffusers import DiffusionPipeline
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re, argparse


def swap_words(sentence, word1, word2):
    pattern = rf'\b({word1}|{word2})\b'

    def replacer(match):
        return word2 if match.group() == word1 else word1

    return re.sub(pattern, replacer, sentence)



# For the first time of using,
# you need to download the huggingface repo "BAAI/Emu2-GEN" to local first
path = "/scratch/sg7457/code/SpotEdit/saved_models/Emu2-Gen"

multimodal_encoder = AutoModelForCausalLM.from_pretrained(
    f"{path}/multimodal_encoder",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    variant="bf16"
)
tokenizer = AutoTokenizer.from_pretrained(f"{path}/tokenizer")

pipe = DiffusionPipeline.from_pretrained(
    path,
    custom_pipeline="pipeline_emu2_gen",
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    variant="bf16",
    multimodal_encoder=multimodal_encoder,
    tokenizer=tokenizer,
)

# # For the non-first time of using, you can init the pipeline directly
# pipe = DiffusionPipeline.from_pretrained(
#     path,
#     custom_pipeline="pipeline_emu2_gen",
#     torch_dtype=torch.bfloat16,
#     use_safetensors=True,
#     variant="bf16",
# )

pipe.to("cuda")

def read_ann_file(ann_path):
    spotedit_list = list()
    with open(ann_path) as file:
        for line in file.readlines():
            spotedit_list.append(json.loads(line))
    return spotedit_list



def edit_image(spotedit_list, root_out_image_path):

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

        input_images.append(item['prompt'])

        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        ret = pipe(input_images)
        ret.image.save(output_image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Define arguments
    parser.add_argument(
        "--mode",
        type=str,
        choices=["real", "syn"],  # restrict allowed values
        required=True
    )
    args = parser.parse_args()
    
    root_out_image_path = f'/scratch/sg7457/dataset/spotedit/generated_images/{args.mode}/emu2'
    ann_file = f'/scratch/sg7457/code/SpotEdit/spotframe_benchmark_{args.mode}_withgt.jsonl'
        
    edit_image(read_ann_file(ann_file), root_out_image_path=root_out_image_path)
