### SSC
This is a PyTorch implementation for the method [Stain Standardization Capsule](https://doi.org/10.1109/JBHI.2020.2983206) presented in the paper:

Yushan Zheng, Zhiguo Jiang*, Haopeng Zhang, Fengying Xie, Dingyi Hu, Shujiao Sun, Jun Shi, and Chenghai Xue, Stain standardization capsule (SSC) for application-driven histopathological image normalization, IEEE Journal of Biomedical and Health Informatics, 2021, 25(2):337-347.


## Run
```
python main.py [image-folder with train and val lists] \
    --use-ssc --weighted-sample
```
use multiple GPUs:
```
python main.py [image-folder with train and val lists] \
    --use-ssc --weighted-sample \
    --dist-url 'tcp://localhost:10001' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0
```

## Cite
```
@article{zheng2020stain,
  author    = {Zheng, Yushan and Jiang, Zhiguo and Zhang, Haopeng and Xie, Fengying and Hu, Dingyi 
               and Sun, Shujiao and Shi, Jun and Xue, Chenghai},
  title     = {Stain standardization capsule for application-driven histopathological image normalization},
  journal   = {IEEE Journal of Biomedical and Health Informatics},
  volume    = {25}, 
  number    = {2}, 
  pages     = {337--347}
  year      = {2021},
}    
```