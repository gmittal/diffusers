# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

# limitations under the License.


import warnings

import torch

from ...pipeline_utils import DiffusionPipeline


class DDPMPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(self, batch_size=1, generator=None, encoder_hidden_states=None, attention_mask=None, output_type="pil", **kwargs):
        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            # Set device as before (to be removed in 0.3.0)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

        # Sample gaussian noise to begin loop
        #import pdb; pdb.set_trace()
        image = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.sample_size[0], self.unet.sample_size[1]),
            generator=generator,
        )
        image = image.to(self.device)

        # set step values
        self.scheduler.set_timesteps(1000)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image, t, encoder_hidden_states, attention_mask)["sample"]

            # 2. compute previous image: x_t -> t_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator)["prev_sample"]

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return {"sample": image}

    @torch.no_grad()
    def run(self, batch_size=1, generator=None, encoder_hidden_states=None, attention_mask=None, output_type="pil", **kwargs):
        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            # Set device as before (to be removed in 0.3.0)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

        # Sample gaussian noise to begin loop
        #import pdb; pdb.set_trace()
        image = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.sample_size[0], self.unet.sample_size[1]),
            generator=generator,
        )
        image = image.to(self.device)

        # set step values
        self.scheduler.set_timesteps(1000)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image, t, encoder_hidden_states, attention_mask)["sample"]

            # 2. compute previous image: x_t -> t_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator)["prev_sample"]


            image_yield = (image / 2 + 0.5).clamp(0, 1)
            image_yield = image_yield.cpu().permute(0, 2, 3, 1).numpy()
            yield image_yield


        ###image = (image / 2 + 0.5).clamp(0, 1)
        ###image = image.cpu().permute(0, 2, 3, 1).numpy()
        ###yield image
        #if output_type == "pil":
        #    image = self.numpy_to_pil(image)

        #return {"sample": image}

    @torch.no_grad()
    def run_clean(self, batch_size=1, generator=None, encoder_hidden_states=None, attention_mask=None, output_type="pil", **kwargs):
        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            # Set device as before (to be removed in 0.3.0)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

        # Sample gaussian noise to begin loop
        #import pdb; pdb.set_trace()
        image = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.sample_size[0], self.unet.sample_size[1]),
            generator=generator,
        )
        image = image.to(self.device)

        # set step values
        self.scheduler.set_timesteps(1000)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image, t, encoder_hidden_states, attention_mask)["sample"]

            # 2. compute previous image: x_t -> t_t-1
            result = self.scheduler.step_clean(model_output, t, image, generator=generator)#["prev_sample"]
            image_clean = result['orig_sample']
            image = result['prev_sample']


            image_yield = (image / 2 + 0.5).clamp(0, 1)
            image_yield = image_yield.cpu().permute(0, 2, 3, 1).numpy()

            image_clean_yield = (image_clean / 2 + 0.5).clamp(0, 1)
            image_clean_yield = image_clean_yield.cpu().permute(0, 2, 3, 1).numpy()
            yield image_yield, image_clean_yield

    @torch.no_grad()
    def swap(self, batch_size=1, generator=None, encoder_hidden_states=None, attention_mask=None, output_type="pil", swap_step=-1, **kwargs):
        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            # Set device as before (to be removed in 0.3.0)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

        # Sample gaussian noise to begin loop
        #import pdb; pdb.set_trace()
        assert batch_size % 2 == 0, batch_size
        half_batch_size = batch_size // 2
        image = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.sample_size[0], self.unet.sample_size[1]),
            generator=generator,
        )
        image = image.to(self.device)

        # set step values
        self.scheduler.set_timesteps(1000)
        import pdb; pdb.set_trace()

        for t in self.progress_bar(self.scheduler.timesteps):
            if swap_step == t:
                print ('-'*10)
                print ('swap')
                image[half_batch_size:] = image[:half_batch_size]
            # 1. predict noise model_output
            model_output = self.unet(image, t, encoder_hidden_states, attention_mask)["sample"]

            # 2. compute previous image: x_t -> t_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator)["prev_sample"]

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return {"sample": image}
