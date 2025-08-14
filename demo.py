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
            
def show_training_samples():            
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
                if obj[-1] == '.':
                    obj = obj[:-1]
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
            

def show_benchmark_samples(category):
    benchmark_list = list()
    with open(f'spotframe_benchmark_{category}_withgt.jsonl') as f:
        for line in f.readlines():
            benchmark_list.append(json.loads(line))
            
    root_model_image_path = f'/vast/sg7457/spotedit/generated_images/{category}/'
    root_image_path = '/vast/sg7457/spotedit/syn_videos/'
    root_ref_image_path = f'/vast/sg7457/spotedit/gpt_generated_images/{category}/'
    
    model_titles_and_path = [
                  ('Ref. Image', root_ref_image_path), 
                  ('Input Image', root_image_path),
                  ('Emu2', os.path.join(root_model_image_path, 'emu2')),
                  ('OmniGen', os.path.join(root_model_image_path, 'omnigen')),
                  ('UNO', os.path.join(root_model_image_path, 'uno')),
                  ('BAGEL', os.path.join(root_model_image_path, 'bagel')),
                  ('OmniGen2', os.path.join(root_model_image_path, 'omnigen2')),
                  ('GPT-4o', root_ref_image_path)
                ]

    # Create a multi-page PDF
    with PdfPages(f'{category}_spotframe_samples.pdf') as pdf:
        for prompt_idx, item in enumerate(benchmark_list[:108]):
            has_break=False
            source_image_paths = [path for _, path in model_titles_and_path]
            source_image_paths[:2] = item['image_list'][:2]
            source_image_paths[2:-1] = [os.path.join(path, str(item['id']), 
                                                     item['image_list'][-1].split('/')[-1]) for path in source_image_paths[2:-1]]

            source_image_paths[-1] = os.path.join(source_image_paths[-1], str(item['id']), item['image_list'][-1].split('/')[-1])
            
            # for temp_idx, path in enumerate(source_image_paths):
            #     if not os.path.exists(path):
            #         print(model_titles_and_path[temp_idx][0])
            #         has_break=True
            #         break
            
            # if has_break:
            #     continue

            fig, axes = plt.subplots(1, 8, figsize=(20, 4))
            fig.suptitle(f"Instruction: {item['prompt']}", fontsize=16)

            first_img = Image.open(source_image_paths[0])
            target_size = first_img.size  # (width, height)

            for ax_idx, (ax, img_path) in enumerate(zip(axes, source_image_paths)):
                
                img = Image.open(img_path).resize(target_size)
                ax.imshow(img)
                ax.set_title(model_titles_and_path[ax_idx][0], fontweight='bold', fontsize=20)
                ax.axis('off')

            plt.tight_layout()
            pdf.savefig(fig)  # Save current figure to PDF
            plt.close(fig)  
            break 
    
show_benchmark_samples(category='syn')