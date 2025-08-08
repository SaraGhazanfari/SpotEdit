class FluxModel:
    def __init__(self):
        self.flux_pipe = self._load_model()

    def _load_model(self):
        import torch
        from diffusers import FluxKontextPipeline
        pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
        pipe.to("cuda")
        return pipe

    def get_response(self, image_path, obj):

        from diffusers.utils import load_image
        prompt = f'Remove {obj} from the images'
        image = self.flux_pipe(
                image=load_image(image_path),
                prompt=prompt,
                guidance_scale=2.5
                ).images[0]

        return image