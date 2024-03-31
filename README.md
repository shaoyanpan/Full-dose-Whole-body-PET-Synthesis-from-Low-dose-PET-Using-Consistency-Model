
# 2D-Medical-Denoising-Diffusion-Probabilistic-Model
**This is the repository for the paper published in Medical Physics: "[Full-dose Whole-body PET Synthesis from Low-dose PET Using High-efficiency Denoising Diffusion Probabilistic Model: PET Consistency Model](https://iopscience.iop.org/article/10.1088/1361-6560/acca5c/meta)".**
Consistency Model is one of the super fast Diffusion Denoising Probability Models (DDPMs), which only use 2-timestep to generate the target image, while the DDPMs usually require 50- to 1000-timesteps. This is particular useful for: 1) Three-dimensional Medical image synthesis, 2) Image translation instead image creation like traditional DDPMs do.

The codes were created based on [image-guided diffusion](https://github.com/openai/guided-diffusion), [SwinUnet](https://github.com/HuCaoFighting/Swin-Unet), and [Monai](https://monai.io/)

# Required packages

The requires packages are in test_env.yaml.

Create an environment using Anaconda:
```
conda env create -f \your directory\test_env.yaml
```

# Usage

The usage is in the jupyter notebook Consistency_Low_Dose_Denoising_main.ipynb. Including how to build the consistency-diffusion forward process, how to build a network, and how to call the whole Consistency process to train, and sample new synthetic images. However, we give simple example below:

**Create diffusion**
```
from cm.resample import UniformSampler
from cm.karras_diffusion import KarrasDenoiser,karras_sample

consistency = KarrasDenoiser(        
        sigma_data=0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        distillation=False,
        loss_norm="l1")

schedule_sampler = UniformSampler(consistency)
```

**Create network for input image with size of 64x64 (Notice this is because we apply the 64x64 patch-based training and inference for our 96x196 low-dose PET images**
```
from Diffusion_model_transformer import *

num_channels=128
attention_resolutions="16,8"
channel_mult = (1, 2, 3, 4)
num_heads=[4,4,8,16]
window_size = [[4,4],[4,4],[4,4],[4,4]]
num_res_blocks = [2,2,2,2]
sample_kernel=([2,2],[2,2],[2,2]),

attention_ds = []
for res in attention_resolutions.split(","):
    # Careful for the image_size//int(res), only use for CNN
    attention_ds.append(image_size//int(res))
class_cond = False
use_scale_shift_norm = True

Consistency_network = SwinVITModel(
        image_size=img_size,
        in_channels=2,
        model_channels=num_channels,
        out_channels=1,
        dims=2,
        sample_kernel = sample_kernel,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=0,
        channel_mult=channel_mult,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=num_heads,
        window_size = window_size,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=False,
        use_new_attention_order=False,
    ).to(device)
```

**Train the consistency model (you don't have to use the ema as in our .ipynb**
```
all_loss = consistency.consistency_losses(Consistency_network,
            target,
            condition,
            num_scales,
            target_model=A_to_B_model_ema)
loss = (all_loss["loss"] * weights).mean()
```

**generate new synthetic images**
```
num_sample = 10
image_size = 256
x = diffusion.p_sample_loop(model,(num_sample, 1, image_size, image_size),clip_denoised=True)
```


# Visual examples

![image_1](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/3a814bd3-1107-4d23-b295-9088530754d8)
![image_2](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/cfb2d2c8-f611-497c-93ff-99b7f1ad27a7)
![image_3](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/e183a0fd-dcd0-4b1a-8c5f-b861c05b4b9f)
![image_27](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/6c43ef4a-6903-4a72-9363-421fd5c264b4)

![image_4](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/877cfa01-d1b9-4728-ad14-58ac41a3ef9d)
![image_402](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/8c44d75c-7a9b-4de6-ba01-bae18b5dfe2c)
![image_6](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/955b5c65-e4a6-4e08-a870-bd59ad0682bd)
![image_69](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/48f9413e-e630-41e3-9edf-57ad3887822c)

![image_1](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/e19f614d-3441-407c-bbbb-e76d2cda6fa3)
![image_5](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/959e8a26-4925-4799-a2b7-a4f8f2e15e43)
![image_7](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/1b4dffb9-a324-4e4b-b76a-1f18648bdb37)
![image_6](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/e1300ad7-2a5a-42ea-8980-8f37427ca7b1)

![image_8](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/0ac4a0f3-ce65-4280-8442-ac8f2e000c4d)
![image_6](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/32a0d462-ebbe-465e-9ac2-e8c5d8f75e07)
![image_4](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/f64e4cc0-155d-4b17-b6aa-68d2362be7ec)
![image_46](https://github.com/shaoyanpan/2D-Medical-Denoising-Diffusion-Probabilistic-Model-/assets/89927506/43a3b4ce-7469-4f18-8dd7-87689df410b7)

