import os, json, time
from models import InternVLModel, FluxModel
from PIL import Image
import re, argparse


def write_jsonl_file(jsonl_file, data_list):
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for item in data_list:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")

def read_jsonl_file(jsonl_file):
    data_list = list()
    with open(jsonl_file) as f:
        for line in f.readlines():
            data_list.append(json.loads(line))

    return data_list
        
def read_initial_ann(root='', num_imgs=5):
    data_list = read_jsonl_file(os.path.join(root, 'train.jsonl'))
   
    selected_data_list = list()
    id_ = 1

    for item_idx, item in enumerate(data_list):
        print(item_idx)
        for idx in range(0, len(item['captions']), num_imgs):
            not_save = False
            for image_path in item['images'][idx:idx+num_imgs]:
                if not os.path.exists(os.path.join(IMG_ROOT, image_path)):
                    not_save = True
                    
            if not not_save:
                selected_data_list.append({ 'orig_id': item['id'],
                                            'id': id_,
                                            'captions': item['captions'][idx:idx+num_imgs],
                                            'images':  item['images'][idx:idx+num_imgs]})
                id_ += 1
    write_jsonl_file('train_data_step_0.jsonl', selected_data_list)
    print(len(data_list), len(selected_data_list))
    
    return selected_data_list

def get_objects_in_image(item, mm_model):

    instruction = "List each three distinct object in the image and that can be removed while preserving the image's overall context or purpose. Please don't mention a part of an object"
    
    item['objects'] = list()
    for image_path in item['images']:
        image_path_list=os.path.join(IMG_ROOT, image_path)
        # Image.open(image_path).save(f'{img_idx}.jpg')
        item['objects'].append(mm_model.get_response(image_path_list,  prompt=instruction))
    return item
        
        
def finalize_objects_in_image(item, mm_model):
    obj_set = set()
    
    for obj in item['objects']:
        for obj in re.sub(r"^\d+\.\s*", "", obj, flags=re.MULTILINE).split('\n'):
            obj_set.add(obj) 
    
    list_of_sets = list()   
    for img_idx, image_path in enumerate(item['images']):
        list_of_sets.append(set())
        for obj in obj_set:
            instruction = f"Is {obj} fully observable in this image?\nAnswer with yes or no"
            image_path_list=os.path.join(IMG_ROOT, image_path)
            if 'yes' in mm_model.get_response(image_path_list,  prompt=instruction).lower():
                list_of_sets[-1].add(obj)
    return list(list_of_sets[0].intersection(list_of_sets[1]))
  
def run_step_1():    
    return read_initial_ann(root='/vast/sg7457/spotedit/syn_annotations')
    
def run_step_2():
    data_list = read_jsonl_file('train_data_step_0.jsonl')

    if os.path.exists('train_data_step_1.jsonl'):
        temp_data_list = read_jsonl_file('train_data_step_1.jsonl')

    data_list[:len(temp_data_list)] = temp_data_list
    mm_model = InternVLModel()

    for item_idx in range(len(temp_data_list), len(data_list)):
        item = data_list[item_idx]
        print(item_idx, flush=True)
        item = get_objects_in_image(item, mm_model)
        if item_idx % 50 == 0:
            write_jsonl_file('train_data_step_1.jsonl', data_list[:item_idx+1])
            
def run_step_3():
    mm_model = InternVLModel()
    data_list = read_jsonl_file('train_data_step_1.jsonl')
    selected_data_list = read_jsonl_file('train_data_step_2.jsonl')
    start_time = time.time()
    for item_idx, item in enumerate(data_list):
        print(f'{item_idx}/{len(data_list)}/{round((time.time()-start_time)/60, 2)}', flush=True)
        if item['id'] <= selected_data_list[-1]['id']:
            continue
        del item['captions']
        try:
            for start_idx in [0,2,3]:
                selected_data_list.append({'orig_id': item['orig_id'],
                    'id':item['id'],
                    'images':item['images'][start_idx:start_idx+2],
                    'objects':item['objects'][start_idx:start_idx+2],
                    })
                selected_data_list[-1]['objects'] = finalize_objects_in_image(selected_data_list[-1], mm_model)
        except:
            print('=====================', item_idx)
        if item_idx % 50 == 0:
            write_jsonl_file('train_data_step_2.jsonl', selected_data_list)
  
def run_step_4(start_idx):
    flux_model = FluxModel()
    data_list = read_jsonl_file('train_data_step_2.jsonl') 
    start_time = time.time()
    for item_idx, item in enumerate(data_list):
        if item_idx < start_idx:
            continue
        print(f'{item_idx}/{len(data_list)}/{round((time.time()-start_time)/60, 2)}', flush=True)
        for obj_idx, obj in enumerate(item['objects']):
            for img in item['images']:
                dist_img_path=os.path.join(TRAIN_DATA, f'{item_idx}_{obj_idx}', img.split('/')[-1])
                if os.path.exists(dist_img_path):
                    continue
                os.makedirs(os.path.dirname(dist_img_path), exist_ok=True)
                img_path = os.path.join(IMG_ROOT, img)
                edited_img = flux_model.get_response(img_path, obj)
                edited_img.save(dist_img_path)
                
  
def all_images_exist(item, item_idx):
    for obj_idx, obj in enumerate(item['objects']):
        if obj[-1] == '.':
            obj = obj[:-1]
        for img_idx, img in enumerate(item['images']):
            edited_img_path=os.path.join(TRAIN_DATA, f'{item_idx}_{obj_idx}', img.split('/')[-1])
            if not os.path.exists(edited_img_path):
                return False
        return True
                               
def run_step_5():
    data_list = read_jsonl_file('train_data_step_2.jsonl') 
    temp_data = read_jsonl_file('train_data_step_3.jsonl')
    data_list[:len(temp_data)] = temp_data
    
    mm_model = InternVLModel()
    start_time = time.time()
    while True:
        for item_idx, item in enumerate(data_list):
            print(f'{item_idx}/{len(data_list)}/{round((time.time()-start_time)/60, 2)}', flush=True)
            if len(item['images'])< 2 or 'status' in item or not all_images_exist(item, item_idx):
                continue
            item['status'] = list()
            
            for obj_idx, obj in enumerate(item['objects']):
                if obj[-1] == '.':
                    obj = obj[:-1]
                instruction = f"Is {obj} in Image-1 correctly removed in Image-2?\nAnswer with yes or no"
                for img_idx, img in enumerate(item['images']):
                    edited_img_path=os.path.join(TRAIN_DATA, f'{item_idx}_{obj_idx}', img.split('/')[-1])
                    img_path = os.path.join(IMG_ROOT, img)
                    if not os.path.exists(edited_img_path):
                        continue
                    if 'yes' in mm_model.get_response([img_path, edited_img_path], instruction).lower():
                        item['status'].append(True)
                        ref_img_path = os.path.join(IMG_ROOT, item['images'][(img_idx+1) % 2]) 
                        instruction1 = f'Do Image-1 and Image-2 have the similar {obj}?\nAnswer with yes or no'
                        if 'no' in mm_model.get_response([ref_img_path, img_path], instruction1).lower():
                            item['status'][-1]=False
                    else:
                        item['status'].append(False)
    
        write_jsonl_file('train_data_step_3.jsonl', data_list)
        print('Saved!!', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Define arguments
    parser.add_argument(
        "--step",
        type=int,
        choices=list(range(0,6)),  # restrict allowed values
        required=True
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        required=False
    )
    
    args = parser.parse_args()          
                   
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    TRAIN_DATA = '/vast/sg7457/spotedit/train_data' 
    IMG_ROOT='/vast/sg7457/spotedit/syn_videos' 
    if args.step == 1:
        run_step_1()
    if args.step == 2:
        run_step_2()   
    if args.step == 3:
        run_step_3() 
    if args.step == 4:
        run_step_4(args.start_idx)
    if args.step == 5:
        run_step_5()

    data_list = read_jsonl_file(os.path.join('/vast/sg7457/spotedit/syn_annotations', 'train.jsonl'))
    item = data_list[0]
    
    import matplotlib.pyplot as plt
    from PIL import Image
    
    rows, cols = 6, 5

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    # Loop through images and plot them
    for i, ax in enumerate(axes.flat):
        ax.imshow(Image.open(os.path.join(IMG_ROOT, item['images'][i])))
        ax.axis("off")  # Hide axes

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("output_grid.png", dpi=300)
    plt.close()