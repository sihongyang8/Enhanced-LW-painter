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

#Citation
If you found this code helpful, please consider citing:
    
     @article{Enhanced LW-painter,
      title={Enhanced LW-painter: Lightweight Image Inpainting via Large Receptive Field and Feature Fusion Optimization},
      author={Sihong Yang, Qian Zhang, Yucheng Yang, Jiliang Shi, Wuer Bai , Shuang Liu},
      journal={Signal, Image and Video Processing},
      year={2025}
    }
