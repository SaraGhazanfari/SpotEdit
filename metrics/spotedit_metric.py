import os, torch
from utils.grounded_segmentation import GroundedSegmentation
from models.utils import read_ann_file
import torch, torch.nn.functional as F
from metrics.encoder import CLIPEncoder, DinoEncoder
import argparse
from PIL import Image
from models.internvl.model import InternVLModel

class SpotEditMetric:
    def __init__(self, mode='syn', encoder_name='clip', pool='cls'):
        self.mode = mode
        self.pool = pool
        ROOT = f'/scratch/sg7457/dataset/spotedit/generated_images/{mode}'
        model_names = [
            'Emu2', 
            'OmniGen', 
            'UNO',
            'BAGEL',
            'OmniGen2'
            ]
        
        if mode == 'syn':
            self.idx_dict = {'general': [0, 108],
                    'input robustness': [108, 160],
                    'ref robustness': [160, 210],
                    'overall robustness': [210, 260]} 
            
        elif mode == 'real':   
            self.idx_dict = {'general': [0, 189],
                    'input robustness': [189, 239],
                    'ref robustness': [239, 290]} 
            
        elif mode == 'dreamedit':
                                           
            self.idx_dict = {'general': [0, 22],
                            'input robustness': None,
                            'ref robustness': None} 
        
        self.all_root_path = {name: os.path.join(ROOT, name.lower()) for name in model_names}
            
        self.root_input_image_path = f'/scratch/sg7457/dataset/spotedit/{mode}_videos'
        self.root_ref_image_path = f'/vast/sg7457/spotedit/gpt_generated_images/{mode}/'
        self.spotedit_list = list()
        self.grounded_segementor = GroundedSegmentation()
        self.spotedit_list = read_ann_file(mode)
        self.encoder = self._load_encoder(encoder_name)
    
    def _load_encoder(self, encoder_name):
        if encoder_name == 'clip':
            return CLIPEncoder()
        
        else:
            return DinoEncoder()
        
    def calculate_similarity_score(self, embed1, embed2):
        return torch.nn.functional.cosine_similarity(embed1, embed2, dim=1)
    
    def calculate_hallu_score(self, robustness_type): 
        mll_model = InternVLModel()
        self.all_root_path['GPT-4o'] = self.root_ref_image_path
        '''
        The output image should be identical to the input image for the robustness evaluation examples
        '''
        for idx, (model_name, model_root) in enumerate(self.all_root_path.items()):
            count, all_ = 0, 0
            for item in self.spotedit_list[self.idx_dict[robustness_type][0]:self.idx_dict[robustness_type][1]]:
                if self.mode == 'syn':
                    out_path = os.path.join(model_root, str(item['id']), item['image_list'][2].split('/')[-1])
                else:
                    out_path = os.path.join(model_root, str(item['id']), item['image_list'][1].split('/')[-1])

                if not os.path.exists(out_path):
                    print(model_name, out_path)
                    continue
                
                resp = mll_model.get_response(image_path_list=out_path, 
                                       prompt=f"Is {item['target_obj']} in the image?\nAnswer with yes or no."  
                                       )
                count += 'yes' in resp.lower()
                all_ += 1
                
            print(f"Hallu {robustness_type} Score for {model_name} is : {count}/{all_} {round(count/all_, 3)} for {all_} samples", flush=True)

            # except Exception as e:
            #     print(model_name, e)
            
    def calculate_robusntess_score(self, robustness_type): 
        self.all_root_path['GPT-4o'] = self.root_ref_image_path
        '''
        The output image should be identical to the input image for the robustness evaluation examples
        '''
        input_image_list = list()

        for idx, (model_name, model_root) in enumerate(self.all_root_path.items()):
            output_image_list = list()
            input_image_list = list()
            for item in self.spotedit_list[self.idx_dict[robustness_type][0]:self.idx_dict[robustness_type][1]]:
                if self.mode == 'syn':
                    out_path = os.path.join(model_root, str(item['id']), item['image_list'][2].split('/')[-1])
                else:
                    out_path = os.path.join(model_root, str(item['id']), item['image_list'][1].split('/')[-1])

                if not os.path.exists(out_path):
                    print(model_name, out_path)
                    continue
                input_image_list.append(item['image_list'][1])
                output_image_list.append(out_path)
                #break

            #if idx == 0:
            input_embeds = self.encoder.encode_images(input_image_list, pool=self.pool)
            output_embeds = self.encoder.encode_images(output_image_list, pool=self.pool)

            sim_score=round(self.calculate_similarity_score(embed1=input_embeds, embed2=output_embeds).mean().item(), 3)
            print(f"{robustness_type} Score for {model_name} is : {sim_score} for {len(output_image_list)} samples", flush=True)

            # except Exception as e:
            #     print(model_name, e)


    def calculate_gt_score(self):
        '''
        The output image should be similar to the GT image for the GT score
        '''
        gt_image_list = list()

        for idx, (model_name, model_root) in enumerate(self.all_root_path.items()):
            output_image_list = list()
            for item in self.spotedit_list[self.idx_dict['general'][0]:self.idx_dict['general'][1]]:
                if idx == 0:
                    if self.mode == 'syn':
                        gt_image_list.append(os.path.join(self.root_ref_image_path, str(item['edit_id']), item['image_list'][2].split('/')[-1]))
                    else:
                        gt_image_list.append(item['image_list'][2])
                        
                output_image_list.append(os.path.join(model_root, str(item['id']), item['image_list'][2].split('/')[-1]))
                #break
            if idx == 0:
                gt_embeds = self.encoder.encode_images(gt_image_list, pool=self.pool)
            output_embeds = self.encoder.encode_images(output_image_list, pool=self.pool)
            sim_score = round(self.calculate_similarity_score(embed1=gt_embeds, embed2=output_embeds).mean().item(), 3)
            print(f"GT Score for {model_name} is : {sim_score} for {len(output_image_list)} samples", flush=True)

    def calculate_object_consistency_score(self):
        '''
        The output should contain similar object as of the reference image
        (1) CUT OUT the target object from the output image and reference image
        (2) compare the two objects
        '''
        
        for idx, (model_name, model_root) in enumerate(self.all_root_path.items()):
            output_image_list = list()
            ref_image_list = list()
            count = 0
            for item in self.spotedit_list[self.idx_dict['general'][0]:self.idx_dict['general'][1]]:
                
                try:
                    output_image_list.append(self.grounded_segementor.get_detected_bbx(edit_type='cut_out', obj=item['target_obj'],
                                                                           image_path=os.path.join(model_root, str(item['id']), item['image_list'][2].split('/')[-1])))
                    ref_image_list.append(self.grounded_segementor.get_detected_bbx(edit_type='cut_out', obj= item['target_obj'],
                                                                        image_path=item['image_list'][0]))
                except Exception as e:
                    count += 1
                    print(e)
                #break
                
            
            ref_objects_embed = self.encoder.encode_images(ref_image_list, pool=self.pool)
            output_objects_embed = self.encoder.encode_images(output_image_list, pool=self.pool)
            
            sim_tensor = self.calculate_similarity_score(embed1=ref_objects_embed, embed2=output_objects_embed)
            sim_tensor = torch.cat([sim_tensor, torch.zeros(count)])

            sim_score = round(sim_tensor.mean().item(), 3)
            print(f"Object Consistency Score for {model_name} is : {sim_score} for {len(output_image_list)} samples, Error: {count}", flush=True)

    def calculate_background_consistency_score(self):
        '''
        The output should contain similar background as of the input image
        (1) MASK OUT the target object from the output image and input image
        (2) compare the two backgrounds
        '''
        for idx, (model_name, model_root) in enumerate(self.all_root_path.items()):
            output_image_list = list()
            input_image_list = list()
            count = 0
            for item in self.spotedit_list[self.idx_dict['general'][0]:self.idx_dict['general'][1]]:
                output_image_path=os.path.join(model_root, str(item['id']), item['image_list'][2].split('/')[-1])
                try:
                    output_image_list.append(self.grounded_segementor.get_detected_bbx(edit_type='mask_out', obj=item['target_obj'],
                                                                            image_path=output_image_path))
                    input_image_list.append(self.grounded_segementor.get_detected_bbx(edit_type='mask_out', obj=item['obj'],
                                                                              image_path=item['image_list'][1]))
                except:
                    count += 1
                
                #break

            input_objects_embed = self.encoder.encode_images(input_image_list, pool=self.pool)
            output_objects_embed = self.encoder.encode_images(output_image_list, pool=self.pool)
            sim_score = round(self.calculate_similarity_score(embed1=input_objects_embed, embed2=output_objects_embed).mean().item(), 3)
            print(f"Background Consistency Score for {model_name} is : {sim_score} for {len(output_image_list)} samples, Error: {count}", flush=True)
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Define arguments
    parser.add_argument(
        "--mode",
        type=str,
        choices=["real", "syn", "dreamedit"],  # restrict allowed values
        required=True
    )
    parser.add_argument(
        "--eval",
        type=str,
        choices=["rob", "stan", "dreamedit"],  # restrict allowed values
        required=True
    )
    
    parser.add_argument(
        "--pool",
        type=str,
        choices=["cls", "mean"],  # restrict allowed values
        required=True
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["clip", "dino"],  # restrict allowed values
        required=True
    )
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    args = parser.parse_args()
    print('==================', args.mode, args.eval, args.encoder, args.pool, '==================', flush=True)
    spotedit_metric = SpotEditMetric(mode=args.mode, encoder_name=args.encoder, pool=args.pool)
    if args.eval == 'rob':
        spotedit_metric.calculate_hallu_score(robustness_type='input robustness')
        spotedit_metric.calculate_robusntess_score(robustness_type='input robustness')
        
        spotedit_metric.calculate_hallu_score(robustness_type='ref robustness')
        spotedit_metric.calculate_robusntess_score(robustness_type='ref robustness')

    elif args.eval == 'stan':   
        spotedit_metric.calculate_gt_score()
        spotedit_metric.calculate_object_consistency_score()
        spotedit_metric.calculate_background_consistency_score()
        
    elif args.eval == 'dreamedit':
        spotedit_metric.calculate_object_consistency_score()
        spotedit_metric.calculate_background_consistency_score()
