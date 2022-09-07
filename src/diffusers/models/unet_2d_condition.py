from typing import Dict, Union

import torch
import torch.nn as nn

from ..configuration_utils import ConfigMixin, register_to_config
from ..modeling_utils import ModelMixin
from .embeddings import TimestepEmbedding, Timesteps
from .unet_blocks import UNetMidBlock2DCrossAttn, get_down_block, get_up_block, CrossAttnDownBlock2D, CrossAttnUpBlock2D, CrossAttnDecoderPositionDownBlock2D, CrossAttnDecoderPositionUpBlock2D, UNetMidBlock2DCrossAttnDecoderPosition, CrossAttnDecoderPositionEncoderPositionDownBlock2D, CrossAttnDecoderPositionEncoderPositionUpBlock2D, UNetMidBlock2DCrossAttnDecoderPositionEncoderPosition, CrossAttnEncoderPositionDownBlock2D, CrossAttnEncoderPositionUpBlock2D, UNetMidBlock2DCrossAttnEncoderPosition
from .unet_blocks import UNetMidBlock2DCrossAttnLSTM, CrossAttnLSTMDownBlock2D, CrossAttnLSTMUpBlock2D
import torch

class UNet2DConditionModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        gradient_checkpointing=False,
        sample_size=None,
        in_channels=4,
        out_channels=4,
        center_input_sample=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        block_out_channels=(320, 640, 1280, 1280),
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-5,
        cross_attention_dim=1280,
        attention_head_dim=8,
        mid_block_type='UNetMidBlock2DCrossAttn',
    ):
        super().__init__()
        self.mid_block_type = mid_block_type
        self.gradient_checkpointing = gradient_checkpointing

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim,
                downsample_padding=downsample_padding,
            )
            self.down_blocks.append(down_block)

        # mid
        #import pdb; pdb.set_trace()
        if self.mid_block_type == 'UNetMidBlock2DCrossAttn':
            self.mid_block = UNetMidBlock2DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift="default",
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim,
                resnet_groups=norm_num_groups,
            )
        elif self.mid_block_type == 'UNetMidBlock2DCrossAttnLSTM':
            self.mid_block = UNetMidBlock2DCrossAttnLSTM(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift="default",
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim,
                resnet_groups=norm_num_groups,
            )
        elif self.mid_block_type == 'UNetMidBlock2DCrossAttnDecoderPosition':
            self.mid_block = UNetMidBlock2DCrossAttnDecoderPosition(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift="default",
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim,
                resnet_groups=norm_num_groups,
            )
        elif self.mid_block_type == 'UNetMidBlock2DCrossAttnDecoderPositionEncoderPosition':
            self.mid_block = UNetMidBlock2DCrossAttnDecoderPositionEncoderPosition(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift="default",
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim,
                resnet_groups=norm_num_groups,
            )
        elif self.mid_block_type == 'UNetMidBlock2DCrossAttnEncoderPosition':
            self.mid_block = UNetMidBlock2DCrossAttnEncoderPosition(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift="default",
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim,
                resnet_groups=norm_num_groups,
            )
        else:
            assert False, self.mid_block_type

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def forward(
        self,
        sample,#: torch.FloatTensor,
        timestep,#: Union[torch.Tensor, float, int],
        encoder_hidden_states,#: torch.Tensor,
        attention_mask=None,
        ): #-> Dict[str, torch.FloatTensor]:

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension
        timesteps = timesteps.broadcast_to(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        emb = self.time_embedding(t_emb)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:

            if hasattr(downsample_block, "attentions") and downsample_block.attentions is not None and (isinstance(downsample_block, CrossAttnDownBlock2D) or isinstance(downsample_block, CrossAttnDecoderPositionDownBlock2D) or isinstance(downsample_block, CrossAttnDecoderPositionEncoderPositionDownBlock2D) or isinstance(downsample_block, CrossAttnEncoderPositionDownBlock2D) or isinstance(downsample_block, CrossAttnLSTMDownBlock2D)):
                #import pdb; pdb.set_trace()
                if not self.gradient_checkpointing:
                    sample, res_samples = downsample_block(
                        hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
                    )
                else:
                    #X = torch.utils.checkpoint.checkpoint(self.cnn_block_2, X)
                    sample, res_samples = torch.utils.checkpoint.checkpoint(downsample_block, 
                        (sample, emb, encoder_hidden_states, attention_mask))
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if not self.gradient_checkpointing:
            sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask)
        else:
            sample = torch.utils.checkpoint.checkpoint(self.mid_block, (sample, emb, encoder_hidden_states, attention_mask))

        # 5. up
        for upsample_block in self.up_blocks:

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "attentions") and upsample_block.attentions is not None and (isinstance(upsample_block, CrossAttnUpBlock2D) or isinstance(upsample_block, CrossAttnDecoderPositionUpBlock2D) or isinstance(upsample_block, CrossAttnDecoderPositionEncoderPositionUpBlock2D) or isinstance(upsample_block, CrossAttnEncoderPositionUpBlock2D) or isinstance(upsample_block, CrossAttnLSTMUpBlock2D)):
                if not self.gradient_checkpointing:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                    )
                else:
                    sample = torch.utils.checkpoint.checkpoint(upsample_block, (sample, res_samples, emb, encoder_hidden_states, attention_mask))
            else:
                sample = upsample_block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples)

        # 6. post-process
        # make sure hidden states is in float32
        # when running in half-precision
        sample = self.conv_norm_out(sample.float()).type(sample.dtype)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        output = {"sample": sample}

        return output
