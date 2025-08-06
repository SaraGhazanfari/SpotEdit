# -*- coding: utf-8 -*-

from PIL import Image
from typing import List, Optional

from safetensors.torch import load_file
import torch
import torch.nn as nn
from torchvision import transforms as TF

from .emu import EmuModel
from .conf.emu_conf import CLIPVisionCfg, TextDecoderCfg
from .constants import EVA_IMAGE_SIZE, OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .constants import DEFAULT_IMG_PLACEHOLDER, DEFAULT_VID_PLACEHOLDER
from .constants import DEFAULT_VIDEO_TOKEN, FAKE_VIDEO_END_TOKEN
from .constants import SYSTEM_MESSAGE, GROUND_SYSTEM_MESSAGE, USER_TOKEN, ASSISTANT_TOKEN, GRD_SYMBOL, DEFAULT_EOS_TOKEN
from .mixin import ModelParallelMixin


class EmuChatGeneration(nn.Module, ModelParallelMixin):

    def __init__(
        self,
        emu_model: EmuModel,
        eva_size=EVA_IMAGE_SIZE,
        eva_mean=OPENAI_DATASET_MEAN,
        eva_std=OPENAI_DATASET_STD,
        **kwargs,
    ):
        super().__init__()

        self.emu_model = emu_model
        self.emu_model.eval()

        self.transform = TF.Compose([
            TF.Resize((eva_size, eva_size), interpolation=TF.InterpolationMode.BICUBIC),
            TF.ToTensor(),
            TF.Normalize(mean=eva_mean, std=eva_std),
        ])

    @torch.no_grad()
    def forward(
        self,
        inputs: List[Image.Image | str] | List[List[Image.Image | str]],
        is_grounding: bool = False,
        num_beams: int = 5,
        max_new_tokens: int = 10,
        min_len: int = 1,
        do_sample: bool = False,
        penalty_alpha: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        length_penalty: float = -1,
        repetition_penalty: float = 1.0,
        synced_gpus: bool = False,
        skip_special_tokens: bool = True,
        **kwargs,
    ):
        """
            For chat generation, inputs must be List[List[str | Image.Image]]
            Otherwise, inputs must be List[str | Image.Image]
        """
        assert isinstance(inputs, list), "inputs must be a list"
        device = self.emu_model.device()
        dtype = self.emu_model.dtype()

        # for chat generation
        if isinstance(inputs[0], list):
            assert len(inputs) % 2 == 1, "last message must be user input"
            (
                text_prompt,
                image_prompt,
                video_prompt,
                image_placeholder,
                video_placeholder,
            ) = self._prepare_chat_inputs(
                inputs,
                is_grounding,
                device,
                dtype,
            )
        else:
            if isinstance(inputs, list):
                assert all([isinstance(i, str | Image.Image) for i in inputs]), "input can't be list of list for normal generation"
            (
                text_prompt,
                image_prompt,
                video_prompt,
                image_placeholder,
                video_placeholder,
            ) = self._prepare_inputs(
                inputs,
                device,
                dtype,
            )

        output = self.emu_model.generate(
            text=text_prompt,
            image=image_prompt,
            video=video_prompt,
            image_placeholder=image_placeholder,
            video_placeholder=video_placeholder,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            min_len=min_len,
            do_sample=do_sample,
            penalty_alpha=penalty_alpha,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            synced_gpus=synced_gpus,
            skip_special_tokens=skip_special_tokens,
            **kwargs,
        )

        return output[0]

    def _prepare_inputs(
        self,
        inputs: List[Image.Image | str],
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        image_placeholder: str = DEFAULT_IMG_PLACEHOLDER,
        video_placeholder: str = DEFAULT_VID_PLACEHOLDER,
    ):
        is_video = False
        text_prompt, image_prompt, video_prompt = "", [], []
        for x in inputs:
            if x == FAKE_VIDEO_END_TOKEN:
                is_video = False
            elif isinstance(x, str):
                if x == DEFAULT_VIDEO_TOKEN:
                    is_video = True
                text_prompt += x
            elif is_video:
                text_prompt += video_placeholder
                video_prompt.append(self.transform(x))
            else:
                text_prompt += image_placeholder
                image_prompt.append(self.transform(x))

        if len(image_prompt) == 0:
            image_prompt = None
        else:
            image_prompt = torch.stack(image_prompt)
            image_prompt = image_prompt.to(device=device, dtype=dtype)

        if len(video_prompt) == 0:
            video_prompt = None
        else:
            video_prompt = torch.stack(video_prompt)
            video_prompt = video_prompt.to(device=device, dtype=dtype)

        return [text_prompt], image_prompt, video_prompt, image_placeholder, video_placeholder

    def _prepare_chat_inputs(
        self,
        inputs: List[List[Image.Image | str]],
        is_grounding: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        image_placeholder: str = DEFAULT_IMG_PLACEHOLDER,
        video_placeholder: str = DEFAULT_VID_PLACEHOLDER,
    ):
        text_prompt = GROUND_SYSTEM_MESSAGE if is_grounding else SYSTEM_MESSAGE
        image_prompt, video_prompt = None, None

        prev_r = None
        for msg in inputs:
            if prev_r == ASSISTANT_TOKEN:
                text_prompt += f"{DEFAULT_EOS_TOKEN}{USER_TOKEN}: "
                prev_r = USER_TOKEN
            elif prev_r is None:
                text_prompt += f" {USER_TOKEN}: "
                prev_r = USER_TOKEN
            else:
                text_prompt += f" {ASSISTANT_TOKEN}: "
                prev_r = ASSISTANT_TOKEN

            text, image, video, _, _ = self._prepare_inputs(msg, device, dtype, image_placeholder, video_placeholder)

            text_prompt += text[0]
            if image is not None:
                image_prompt = image if image_prompt is None else torch.cat([image_prompt, image])
            if video is not None:
                video_prompt = video if video_prompt is None else torch.cat([video_prompt, video])

        text_prompt += f" {ASSISTANT_TOKEN}:"
        if is_grounding:
            text_prompt += GRD_SYMBOL

        return [text_prompt], image_prompt, video_prompt, image_placeholder, video_placeholder

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        instruct: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        use_safetensors: bool = False,
        **kwargs,
    ):
        ins = cls.from_config(instruct=instruct, **kwargs).to(dtype)
        if use_safetensors:
            state_dict = load_file(path)
        else:
            state_dict = torch.load(path)

        ins.emu_model.load_state_dict(state_dict, strict=True)
        return ins

    @classmethod
    def from_config(
        cls,
        instruct: bool = False,
        **kwargs,
    ):
        if instruct:
            vision_cfg = CLIPVisionCfg(n_query=256, v_query=64)
            text_decoder_cfg = TextDecoderCfg(instruct=True)
        else:
            vision_cfg = CLIPVisionCfg()
            text_decoder_cfg = TextDecoderCfg()

        emu_model = EmuModel(vision_cfg=vision_cfg, text_decoder_cfg=text_decoder_cfg)
        return cls(
            emu_model=emu_model,
            **kwargs,
        )


    def multicuda(
        self,
        device_list: List[str | torch.device],
    ):
        """
            A simple multi device strategy, which distribute blocks in large language modles averagely
            into multi devices while keeping rest layers in LLM on the first device
            emu_model.visual:                           4B [cuda:0]
            emu_model.project_down:                   omit [cuda:0]
            emu_model.project_up:                     omit [cuda:0]
            emu_model.decoder.lm.model.embed_tokens:  omit [cuda:0]
            emu_model.decoder.lm.model.norm:          omit [cuda:0]
            emu_model.decoder.lm.lm_head:             omit [cuda:0]
            emu_model.decoder.lm.model.layers.[0..59]: 33B (0.55B/layer) [cuda:0 ~ cuda:x]
        """
        mp_rule = {
            "emu_model.visual": device_list[0],
            "emu_model.project_down": device_list[0],
            "emu_model.project_up": device_list[0],
            "emu_model.decoder.lm.model.embed_tokens": device_list[0],
            "emu_model.decoder.lm.model.norm": device_list[0],
            "emu_model.decoder.lm.lm_head": device_list[0],
        }

        other_params = self.params_num(self.emu_model.visual) + \
                       self.params_num(self.emu_model.project_down) + \
                       self.params_num(self.emu_model.project_up) + \
                       self.params_num(self.emu_model.decoder.lm.model.embed_tokens) + \
                       self.params_num(self.emu_model.decoder.lm.model.norm) + \
                       self.params_num(self.emu_model.decoder.lm.lm_head)

        layer_params = self.params_num(self.emu_model.decoder.lm.model.layers[0])
        layer_num = len(self.emu_model.decoder.lm.model.layers)

        total_params = other_params + layer_params * layer_num
        params_per_device = [total_params / len(device_list) for _ in device_list]
        params_per_device[0] -= other_params

        accumulate_params, device_idx = 0, 0
        for idx in range(layer_num):
            if accumulate_params + layer_params > params_per_device[device_idx] and device_idx < len(device_list) - 1:
                accumulate_params = 0
                device_idx += 1

            mp_rule[f"emu_model.decoder.lm.model.layers.{idx}"] = device_list[device_idx]
            accumulate_params += layer_params

        self.parallel(mp_rule)
        return self

    def multito(self, device_list: List[str | torch.device]):
        return self.multicuda(device_list)
