import os, json
import pandas as pd


def read_spotedit(ann_path):
    spotedit_list = list()
    with open(ann_path) as file:
        for line in file.readlines():
            spotedit_list.append(json.loads(line))
    return spotedit_list

def read_dreamedit():
    import pandas as pd
    root = '/vast/sg7457/DreamEditBench'
    df = pd.read_csv(os.path.join(root, 'data/relacement_subset.csv'))
    pairs = []

    current_subject = None
    reference_image = None
    seen_subjects = set()
    for row_id, row in df.iterrows():
        if pd.notna(row["subject"]):  # New subject block
            current_subject = row["subject"]
            reference_image = row["file_name"]
            img_path=os.path.join(root, 'data/replacement_source', current_subject.strip())
        else:  # Input image row
            first_input_image = row["file_name"]
            # if current_subject in seen_subjects:
            #     continue
            pairs.append({
                "id": row_id,
                "prompt": f'Replace the {current_subject} in the second image with {current_subject} in the first image',
                "image_list": [os.path.join(img_path, reference_image),
                               os.path.join(img_path, first_input_image),
                               os.path.join(img_path, first_input_image)],
                "target_obj": current_subject,
                "obj": current_subject
            })
            # seen_subjects.add(current_subject)
    print(len(pairs))
    return pairs

def read_ann_file(mode):
    if mode == 'dreamedit':
        ann_list = read_dreamedit()
    else:
        ann_list= read_spotedit(f'{mode}.jsonl') 
    return ann_list
