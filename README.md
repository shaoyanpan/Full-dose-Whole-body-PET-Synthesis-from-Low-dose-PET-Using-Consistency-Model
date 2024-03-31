
# 2D-Medical-Consistency-Model
**This is the repository for the paper published in Medical Physics: "[Full-dose Whole-body PET Synthesis from Low-dose PET Using High-efficiency Denoising Diffusion Probabilistic Model: PET Consistency Model](https://iopscience.iop.org/article/10.1088/1361-6560/acca5c/meta)".**

Consistency Model is one of the super fast Denoising Diffusion Probability Models (DDPMs), which only use 2-timestep to generate the target image, while the DDPMs usually require 50- to 1000-timesteps. This is particular useful for: 1) Three-dimensional Medical image synthesis, 2) Image translation instead image creation like traditional DDPMs do.

The codes were created based on [image-guided diffusion](https://github.com/openai/guided-diffusion), [SwinUnet](https://github.com/HuCaoFighting/Swin-Unet), and [Monai](https://monai.io/)

# Required packages

The requires packages are in test_env.yaml.

Create an environment using Anaconda:
```
conda env create -f \your directory\test_env.yaml
```

# Usage

The usage is in the jupyter notebook Consistency_Low_Dose_Denoising_main.ipynb. Including how to build the consistency-diffusion forward process, how to build a network, and how to call the whole Consistency process to train, and sample new synthetic images. However, we give simple example below:

**Create Consistency-diffusion**
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

# Don't forget the ema model. You must have this to run the code no matter you use ema or not.
Consistency_network_ema = copy.deepcopy(Consistency_network)
```

**Train the consistency model (you don't have to use the ema as in our .ipynb**
```
# Create fake examples, just for you to run the code
img_size = (96,192) # Adjust this for the size of your image input
condition = torch.randn([1,1,96,192]) #batch, channel, height, width
target = torch.randn([1,1,96,192]) #batch, channel, height, width

all_loss = consistency.consistency_losses(Consistency_network,
            target,
            condition,
            num_scales,
            target_model=Consistency_network_ema)
loss = (all_loss["loss"] * weights).mean()
```

**generate new synthetic images**
```
# Create fake examples
Low_dose = torch.randn([1,1,96,192]) #batch, channel, height, width
img_size = (96,192) # Adjust this for the size of your image input

# Set up the step# for your inference
consistency_num = 3
steps = np.round(np.linspace(1.0, 150.0, num=consistency_num))
def diffusion_sampling(condition,A_to_B_model):
    sampled_images = karras_sample(
                        consistency,
                        A_to_B_model,
                        shape=Low_dose.shape,
                        condition=Low_dose,
                        sampler="multistep",
                        steps = 151,
                        ts = steps,
                        device = device)
    return sampled_images

# Patch-based inference parameter
overlap = 0.75
mode ='constant'
back_ground_intensity = -1
Inference_patch_number_each_time = 40
from monai.inferers import SlidingWindowInferer
inferer = SlidingWindowInferer(img_size, Inference_patch_number_each_time, overlap=overlap,
                               mode =mode ,cval = back_ground_intensity, sw_device=device,device = device)

# 
High_dose_samples = inferer(Low_dose,diffusion_sampling,Consistency_network)  
```


# Visual examples
![Picture1](https://github.com/shaoyanpan/Full-dose-Whole-body-PET-Synthesis-from-Low-dose-PET-Using-Consistency-Model/assets/89927506/15e56941-d7c6-4eab-994a-04e2d1d4d1df)

