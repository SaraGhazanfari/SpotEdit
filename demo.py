import os, json, argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image



def write_jsonl_file(jsonl_file, data_list):
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for item in data_list:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")
            
            
benchmark_list = list()
with open('train_data_step_3.jsonl') as f:
    for line in f.readlines():
        benchmark_list.append(json.loads(line))
# count = 0        
# for item_idx, item in enumerate(benchmark_list):
#     if 'status' in item:
#         count += len(item['status'])
           
# print(count)  
        
 

TRAIN_DATA = '/vast/sg7457/spotedit/train_data'
IMG_ROOT = '/vast/sg7457/spotedit/syn_videos'

with PdfPages('train_part.pdf') as pdf:
    for item_idx, item in enumerate(benchmark_list):
        if item_idx % 20 != 0:
            continue
        status_idx = -1
        if 'status' not in item or len(item['status']) != len(item['images']) * len(item['objects']):
            continue
        for obj_idx, obj in enumerate(item['objects']):
            for img_idx, img in enumerate(item['images']):
                status_idx += 1
                
                if not item['status'][status_idx]:
                    continue
                input_img_path=os.path.join(TRAIN_DATA, f'{item_idx}_{obj_idx}', img.split('/')[-1])
                out_img_path = os.path.join(IMG_ROOT, img)
                ref_img_path = os.path.join(IMG_ROOT, item['images'][(img_idx+1) % 2])

                fig, axes = plt.subplots(1, 3, figsize=(20, 4))
                fig.suptitle(f"Instruction: Add {obj} from the first image to the second image.", fontsize=16)
                axes[0].imshow(Image.open(ref_img_path))
                axes[1].imshow(Image.open(input_img_path))
                axes[2].imshow(Image.open(out_img_path))
                plt.show()
                plt.tight_layout()
                pdf.savefig(fig)  # Save current figure to PDF
                plt.close(fig) 
                break 
            


