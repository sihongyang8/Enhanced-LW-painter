# Enhanced LW-painter

# Environment
    Python
    pytorch
    opencv
    PIL
    colorama
or see the requirements.txt


# Run
    1.train the model
    python train.py
    2.test the model
    python test.py

# Download Datasets
We use [FFHQ](https://github.com/NVlabs/ffhq-dataset), [LFW](http://vis-www.cs.umass.edu/lfw/index.html), [Dunhuang Mogao Grottoes Mural](https://github.com/qinnzou/mural-image-inpainting) and [Paris StreetView](https://github.com/pathak22/context-encoder) datasets. Liu et al. provides 12k [irregular masks](https://nv-adlr.github.io/publication/partialconv-inpainting).

# Acknowledgement
[MSCSWT-Net](https://github.com/bobo0303/MSCSWT-Net)
[LSKA](https://github.com/StevenLauHKHK/Large-Separable-Kernel-Attention)
[EAN](https://github.com/Lihahaah/EAN)
[CTSDG](https://github.com/Xiefan-Guo/CTSDG)
[ELAN](https://github.com/xindongzhang/ELAN)
[BSConv](https://github.com/zeiss-microscopy/BSConv)

# Citation
If you found this code helpful, please consider citing:
    

    @article{article,
    author = {Yang, Sihong and Zhang, Qian and Yang, Yucheng and Shi, Jiliang and Bai, Wuer and Liu, Shuang},
    year = {2025},
    month = {09},
    pages = {},
    title = {Enhanced LW-painter: Lightweight Image Inpainting via Large Receptive Field and Feature Fusion Optimization},
    volume = {19},
    journal = {Signal, Image and Video Processing},
    doi = {10.1007/s11760-025-04685-5}
    }
