# MAT: Mask-Aware Transformer for Large Hole Image Inpainting (CVPR 2022 Best Paper Finalist, Oral)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mat-mask-aware-transformer-for-large-hole/image-inpainting-on-places2-1)](https://paperswithcode.com/sota/image-inpainting-on-places2-1?p=mat-mask-aware-transformer-for-large-hole)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mat-mask-aware-transformer-for-large-hole/image-inpainting-on-celeba-hq)](https://paperswithcode.com/sota/image-inpainting-on-celeba-hq?p=mat-mask-aware-transformer-for-large-hole)

#### Wenbo Li, Zhe Lin, Kun Zhou, Lu Qi, Yi Wang, Jiaya Jia

#### [\[Paper\]](https://arxiv.org/abs/2203.15270)
---

## :rocket:  :rocket:  :rocket: **News**

- **\[2022.10.03\]** Model for FFHQ-512 is available. ([Link](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155137927_link_cuhk_edu_hk/ESwt5gvPs4JOvC76WAEDfb4BSJZNy-qsfJSUZz2kTxYyWw?e=71nHCJ))

- **\[2022.09.10\]** We could provide all testing images of Places and CelebA inpainted by our MAT and other methods. Since there are too many images, please send an email to wenboli@cse.cuhk.edu.hk and explain your needs.

- **\[2022.06.21\]** We provide a SOTA Places-512 model ([Places\_512\_FullData.pkl](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155137927_link_cuhk_edu_hk/EuY30ziF-G5BvwziuHNFzDkBVC6KBPRg69kCeHIu-BXORA?e=7OwJyE)) trained with full Places data (8M images). It achieves significant improvements on all metrics.

    <table>
    <thead>
      <tr>
        <th rowspan="2">Model</th>
        <th rowspan="2">Data</th>
        <th colspan="3">Small Mask</th>
        <th colspan="3">Large Mask</th>
      </tr>
      <tr>
        <th>FID&darr;</th>
        <th>P-IDS&uarr;</th>
        <th>U-IDS&uarr;</th>
        <th>FID&darr;</th>
        <th>P-IDS&uarr;</th>
        <th>U-IDS&uarr;</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>MAT (Ours)</td>
        <td>8M</td>
        <td><b>0.78</b></td>
        <td><b>31.72</b></td>
        <td><b>43.71</b></td>
        <td><b>1.96</b></td>
        <td><b>23.42</b></td>
        <td><b>38.34</b></td>
      </tr>
      <tr>
        <td>MAT (Ours)</td>
        <td>1.8M</td>
        <td>1.07</td>
        <td>27.42</td>
        <td>41.93</td>
        <td>2.90</td>
        <td>19.03</td>
        <td>35.36</td>
      </tr>
      <tr>
        <td>CoModGAN</td>
        <td>8M</td>
        <td>1.10</td>
        <td>26.95</td>
        <td>41.88</td>
        <td>2.92</td>
        <td>19.64</td>
        <td>35.78</td>
      </tr>
      <tr>
        <td>LaMa-Big</td>
        <td>4.5M</td>
        <td>0.99</td>
        <td>22.79</td>
        <td>40.58</td>
        <td>2.97</td>
        <td>13.09</td>
        <td>32.39</td>
      </tr>
    </tbody>
    </table>

- **\[2022.06.19\]** We have uploaded the CelebA-HQ-256 model and masks. Because the original model was lost, we retrained the model so that the results may slightly differ from the reported ones.

---

## Web Demo

Thank [Replicate](https://replicate.com/home) for providing a [web demo](https://replicate.com/fenglinglwb/large-hole-image-inpainting) for our MAT. But I didn't check if this demo is correct. You are recommended to use our models as following.

---

## Visualization

We present a transformer-based model (MAT) for large hole inpainting with high fidelity and diversity.

![large hole inpainting with pluralistic generation](/figures/teasing.png)

Compared to other methods, the proposed MAT restores more photo-realistic images with fewer artifacts.

![comparison with sotas](/figures/sota.png)

## Usage

It is highly recommanded to adopt Conda/MiniConda to manage the environment to avoid some compilation errors.

1. Clone the repository.
    ```shell
    git clone https://github.com/fenglinglwb/MAT.git 
    ```
2. Install the dependencies.
    - Python 3.7
    - PyTorch 1.7.1
    - Cuda 11.0
    - Other packages
    ```shell
    pip install -r requirements.txt
    ```

## Quick Test

1. We provide models trained on CelebA-HQ, FFHQ and Places365-Standard at 512x512 resolution. Download models from [One Drive](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155137927_link_cuhk_edu_hk/EuY30ziF-G5BvwziuHNFzDkBVC6KBPRg69kCeHIu-BXORA?e=7OwJyE) and put them into the 'pretrained' directory. The released models are retrained, and hence the visualization results may slightly differ from the paper.

2. Obtain inpainted results by running
    ```shell
    python generate_image.py --network model_path --dpath data_path --outdir out_path [--mpath mask_path]
    ```
    where the mask path is optional. If not assigned, random 512x512 masks will be generated. Note that 0 and 1 values in a mask refer to masked and remained pixels.

    For example, run
    ```shell
    python generate_image.py --network pretrained/CelebA-HQ.pkl --dpath test_sets/CelebA-HQ/images --mpath test_sets/CelebA-HQ/masks --outdir samples
    ```

    Note. 
    - Our implementation only supports generating an image whose size is a multiple of 512. You need to pad or resize the image to make its size a multiple of 512. Please pad the mask with 0 values.
    - If you want to use the CelebA-HQ-256 model, please specify the parameter 'resolution' as 256 in generate\_image.py.

## Train

For example, if you want to train a model on Places, run a bash script with
```shell
python train.py \
    --outdir=output_path \
    --gpus=8 \
    --batch=32 \
    --metrics=fid36k5_full \
    --data=training_data_path \
    --data_val=val_data_path \
    --dataloader=datasets.dataset_512.ImageFolderMaskDataset \
    --mirror=True \
    --cond=False \
    --cfg=places512 \
    --aug=noaug \
    --generator=networks.mat.Generator \
    --discriminator=networks.mat.Discriminator \
    --loss=losses.loss.TwoStageLoss \
    --pr=0.1 \
    --pl=False \
    --truncation=0.5 \
    --style_mix=0.5 \
    --ema=10 \
    --lr=0.001
```

Description of arguments:
- outdir: output path for saving logs and models
- gpus: number of used gpus
- batch: number of images in all gpus
- metrics: find more metrics in 'metrics/metric\_main.py'
- data: training data
- data\_val: validation data
- dataloader: you can define your own dataloader
- mirror: use flip augmentation or not 
- cond: use class info, default: false
- cfg: configuration, find more details in 'train.py'
- aug: use augmentation of style-gan-ada or not, default: false
- generator: you can define your own generator
- discriminator: you can define your own discriminator
- loss: you can define your own loss
- pr: ratio of perceptual loss
- pl: use path length regularization or not, default: false
- truncation: truncation ratio proposed in stylegan
- style\_mix: style mixing ratio proposed in stylegan
- ema: exponoential moving averate, ~K samples
- lr: learning rate

## Evaluation

We provide evaluation scrtips for FID/U-IDS/P-IDS/LPIPS/PSNR/SSIM/L1 metrics in the 'evaluation' directory. Only need to give paths of your results and GTs.

We also provide our masks for CelebA-HQ-val and Places-val [here](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155137927_link_cuhk_edu_hk/EuY30ziF-G5BvwziuHNFzDkBVC6KBPRg69kCeHIu-BXORA?e=7OwJyE).


## Citation

    @inproceedings{li2022mat,
        title={MAT: Mask-Aware Transformer for Large Hole Image Inpainting},
        author={Li, Wenbo and Lin, Zhe and Zhou, Kun and Qi, Lu and Wang, Yi and Jia, Jiaya},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        year={2022}
    }

## License and Acknowledgement
The code and models in this repo are for research purposes only. Our code is bulit upon [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch).
