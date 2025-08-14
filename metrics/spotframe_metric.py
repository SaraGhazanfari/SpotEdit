import os, torch
from PIL import Image
from utils.grounded_segmentation import GroundedSegmentation
from models.utils import read_ann_file


class SpotFrameMetric:
    def __init__(self, mode='syn'):
        ROOT = f'/scratch/sg7457/dataset/spotedit/generated_images/{mode}'
        model_names = [
            #'Emu2', 'OmniGen', 'UNO',
            'BAGEL',
            #'OmniGen2'
            ]
        self.mode = mode
        if mode == 'syn':
            self.idx_dict = {'general': [0, 108],
                    'input robustness': [108, 160],
                    'ref robustness': [160, 210],
                    'overall robustness': [210, 260]} 
            
        elif mode == 'real':   
            self.idx_dict = {'general': [0, 180],
                    'input robustness': [180, 230],
                    'ref robustness': [230, 281]} 
            
        elif mode == 'dreamedit':
                                           
            self.idx_dict = {'general': [0, 22],
                            'input robustness': None,
                            'ref robustness': None} 
        
        self.all_root_path = {name: os.path.join(ROOT, name.lower()) for name in model_names}
            
        self.root_input_image_path = f'/scratch/sg7457/dataset/spotedit/{mode}_videos'
        self.root_ref_image_path = '/mnt/localssd/spot_edit/george_story/edited_videos/'
        self.spotframe_list = list()
        self.grounded_segementor = GroundedSegmentation()
        self.spotframe_list = read_ann_file(mode)
        self.similarity_model, self.preprocess = self._load_similarity_model()
    
    def _load_similarity_model(self):
        import open_clip
        model_name='hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_name)
        tokenizer = open_clip.get_tokenizer(model_name)
        return model, preprocess_val

    def _encode_images(self, image_list):
        if isinstance(image_list[0], str):
            image_list = [Image.open(img_path) for img_path in image_list]
        images = torch.concat([self.preprocess(img_path).unsqueeze(0) for img_path in image_list])
        with torch.no_grad(), torch.autocast("cuda"):
            return self.similarity_model.encode_image(images)

    def calculate_similarity_score(self, embed1, embed2):
        return torch.nn.functional.cosine_similarity(embed1, embed2, dim=1)
    
    def calculate_robusntess_score(self, robustness_type): 
        '''
        The output image should be identical to the input image for the robustness evaluation examples
        '''
        input_image_list = list()

        for idx, (model_name, model_root) in enumerate(self.all_root_path.items()):
            output_image_list = list()
            input_image_list = list()
            for item in self.spotframe_list[self.idx_dict[robustness_type][0]:self.idx_dict[robustness_type][1]]:
                if self.mode == 'syn':
                    out_path = os.path.join(model_root, str(item['id']), item['image_list'][2].split('/')[-1])
                else:
                    out_path = os.path.join(model_root, str(item['id']), item['image_list'][1].split('/')[-1])

                if not os.path.exists(out_path):
                    print(model_name, out_path)
                    continue
                input_image_list.append(item['image_list'][1])
                output_image_list.append(out_path)

            #if idx == 0:
            input_embeds = self._encode_images(input_image_list)
            output_embeds = self._encode_images(output_image_list)

            sim_score=round(self.calculate_similarity_score(embed1=input_embeds, embed2=output_embeds).mean().item(), 4)
            num_samples=self.idx_dict[robustness_type][1]-self.idx_dict[robustness_type][0]
            print(f"{robustness_type} Score for {model_name} is : {sim_score} for {num_samples} samples")

            # except Exception as e:
            #     print(model_name, e)


    def calculate_gt_score(self):
        '''
        The output image should be similar to the GT image for the GT score
        '''
        gt_image_list = list()

        for idx, (model_name, model_root) in enumerate(self.all_root_path.items()):
            output_image_list = list()
            for item in self.spotframe_list[self.idx_dict['general'][0]:self.idx_dict['general'][1]]:
                if idx == 0:
                    if self.mode == 'syn':
                        gt_image_list.append(os.path.join(self.root_ref_image_path, str(item['edit_id']), item['image_list'][2].split('/')[-1]))
                    else:
                        gt_image_list.append(item['image_list'][2])
                        
                output_image_list.append(os.path.join(model_root, str(item['id']), item['image_list'][2].split('/')[-1]))

            if idx == 0:
                gt_embeds = self._encode_images(gt_image_list)
            output_embeds = self._encode_images(output_image_list)
            sim_score = round(self.calculate_similarity_score(embed1=gt_embeds, embed2=output_embeds).mean().item(), 4)
            print(f"GT Score for {model_name} is : {sim_score} for {self.idx_dict['general'][1]-self.idx_dict['general'][0]} samples")

    def calculate_object_consistency_score(self):
        '''
        The output should contain similar object as of the reference image
        (1) CUT OUT the target object from the output image and reference image
        (2) compare the two objects
        '''
        ref_image_list = list()

        for idx, (model_name, model_root) in enumerate(self.all_root_path.items()):
            output_image_list = list()
            for item in self.spotframe_list[self.idx_dict['general'][0]:self.idx_dict['general'][1]]:
                if idx == 0:
                    ref_image_list.append(self.grounded_segementor.get_detected_bbx(edit_type='cut_out', obj= item['target_obj'],
                                                                            image_path=item['image_list'][0]))
                    # os.makedirs(os.path.join('masks', str(item['id'])), exist_ok=True)
                    # ref_image_list[-1].save(os.path.join('masks', str(item['id']), item['image_list'][0].split('/')[-1]))
                output_image_list.append(self.grounded_segementor.get_detected_bbx(edit_type='cut_out', obj=item['target_obj'],
                                                                           image_path=os.path.join(model_root, str(item['id']), item['image_list'][2].split('/')[-1])))
                
                # output_image_list[-1].save(os.path.join('masks', str(item['id']), item['image_list'][2].split('/')[-1]))

            if idx == 0:
                ref_objects_embed = self._encode_images(ref_image_list)

            output_objects_embde = self._encode_images(output_image_list)
            sim_score = round(self.calculate_similarity_score(embed1=ref_objects_embed, embed2=output_objects_embde).mean().item(), 4)
            print(f"Object Consistency Score for {model_name} is : {sim_score} for {self.idx_dict['general'][1]-self.idx_dict['general'][0]} samples")

    def calculate_background_consistency_score(self):
        '''
        The output should contain similar background as of the input image
        (1) MASK OUT the target object from the output image and input image
        (2) compare the two backgrounds
        '''
        input_image_list = list()

        for idx, (model_name, model_root) in enumerate(self.all_root_path.items()):
            output_image_list = list()
            for item in self.spotframe_list[self.idx_dict['general'][0]:self.idx_dict['general'][1]]:
                if idx == 0:
                    input_image_list.append(self.grounded_segementor.get_detected_bbx(edit_type='mask_out', obj=item['obj'],
                                                                              image_path=item['image_list'][1]))

                output_image_path=os.path.join(model_root, str(item['id']), item['image_list'][2].split('/')[-1])  

                output_image_list.append(self.grounded_segementor.get_detected_bbx(edit_type='mask_out', obj=item['target_obj'],
                                                                           image_path=output_image_path))
            if idx == 0:
                input_objects_embed = self._encode_images(input_image_list)

            output_objects_embde = self._encode_images(output_image_list)
            sim_score = round(self.calculate_similarity_score(embed1=input_objects_embed, embed2=output_objects_embde).mean().item(), 4)
            print(f"Background Consistency Score for {model_name} is : {sim_score} for {self.idx_dict['general'][1]-self.idx_dict['general'][0]} samples")
            


spotframe_metric = SpotFrameMetric(mode='dreamedit')

#spotframe_metric.calculate_robusntess_score(robustness_type='input robustness')
#spotframe_metric.calculate_robusntess_score(robustness_type='ref robustness')
# spotframe_metric.calculate_robusntess_score(robustness_type='overall robustness')


# spotframe_metric.calculate_gt_score()
spotframe_metric.calculate_object_consistency_score()
spotframe_metric.calculate_background_consistency_score()