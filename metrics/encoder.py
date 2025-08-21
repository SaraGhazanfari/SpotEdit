import os, torch
from PIL import Image
from utils.grounded_segmentation import GroundedSegmentation
from models.utils import read_ann_file
import torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def dino_load_image(path):
    transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    if isinstance(path, str):
        img = Image.open(path).convert("RGB")
    else:
        img = path
    return transform(img).unsqueeze(0)  # (1,3,H,W)

class DinoEncoder:
    
    def __init__(self):
        
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')#'dinov2_vits14')
        self.model.eval()

    @torch.no_grad()
    def encode_images(self, img_list, pool='cls'):  # pool: 'mean' or 'cls'
        img_list = [dino_load_image(img_path) for img_path in img_list]
        feats = self.model.forward_features(torch.concat(img_list))
        if pool == 'mean':
            z = feats['x_norm_patchtokens'].mean(dim=1)   # (B, D)
        else:
            z = feats['x_norm_clstoken']                  # (B, D)
        # L2-normalize for cosine similarity
        z = F.normalize(z, dim=-1)
        return z
    
class CLIPEncoder:
    def __init__(self):
        import open_clip
        model_name='hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
        self.model, preprocess_train, self.processor = open_clip.create_model_and_transforms(model_name)
        # from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

        # self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        # self.model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-large-patch14")
        
    @torch.no_grad()
    def encode_images(self, image_list, pool):
        if isinstance(image_list[0], str):
            image_list = [Image.open(img_path) for img_path in image_list]
        images = torch.stack([self.processor(img) for img in image_list], dim=0)
        return F.normalize(self.model.encode_image(images), dim=-1)
