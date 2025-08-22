import json


def read_spotedit(ann_path):
    spotedit_list = list()
    with open(ann_path) as file:
        for line in file.readlines():
            spotedit_list.append(json.loads(line))
    return spotedit_list

def read_dreamedit():
    import os
    import csv

    root = '/vast/sg7457/DreamEditBench'
    csv_path = os.path.join(root, 'data/relacement_subset.csv')

    pairs = []
    current_subject = None
    reference_image = None
    seen_subjects = set()
    img_path = None

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row_id, row in enumerate(reader):
            subject = (row.get("subject") or "").strip()
            file_name = (row.get("file_name") or "").strip()

            if subject:  # New subject block
                current_subject = subject
                reference_image = file_name
                img_path = os.path.join(root, 'data/replacement_source', current_subject)
            else:  # Input image row
                first_input_image = file_name
                pairs.append({
                    "id": row_id,
                    "prompt": f"Replace the {current_subject} in the second image with {current_subject} in the first image",
                    "image_list": [
                        os.path.join(img_path, reference_image),
                        os.path.join(img_path, first_input_image),
                        os.path.join(img_path, first_input_image),
                    ],
                    "target_obj": current_subject,
                    "obj": current_subject,
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
