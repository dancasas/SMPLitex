# SMPLitex: A Generative Model and Dataset for 3D Human Texture Estimation from Single Image

<img src="img/casas_BMVC23.png" alt="Teaser image" width="80%"/>

[[Project website](https://dancasas.github.io/projects/SMPLitex/index.html)]

## Abstract

> We propose SMPLitex, a method for estimating and manipulating the complete 3D appearance of humans captured from a single image. SMPLitex builds upon the recently proposed generative models for 2D images, and extends their use to the 3D domain through pixel-to-surface correspondences computed on the input image. To this end, we first train a generative model for complete 3D human appearance, and then fit it into the input image by conditioning the generative model to the visible parts of subject. Furthermore, we propose a new dataset of high-quality human textures built by sampling SMPLitex conditioned on subject descriptions and images. We quantitatively and qualitatively evaluate our method in 3 publicly available datasets, demonstrating that SMPLitex significantly outperforms existing methods for human texture estimation while allowing for a wider variety of tasks such as editing, synthesis, and manipulation.

## SMPLitex dataset
[Dataset](./textures)

## Install instructions
Coming soon...

## Citation

```
@inproceedings{casas2023smplitex,
    title = {{SMPLitex: A Generative Model and Dataset for 3D Human Texture Estimation from Single Image}},
    author = {Casas, Dan and Comino-Trinidad, Marc},
    booktitle = {British Machine Vision Conference (BMVC)},
    year = {2023}
}
```