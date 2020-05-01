# Adaptive Channel-Wise EM Attention

The code is based on [Latent Filter Scaling for Multimodal Unsupervised Image-to-Image Translation](https://arxiv.org/abs/1812.09877) (**LFS**)and [Expectation-Maximization Attention Networks for Semantic Segmentation](https://arxiv.org/abs/1907.13426)(EMANET).



**layers.py**

This file contains the implementation of  the module of Adaptive Channel-Wise EM Attention.

Here is an explanation:

```python
 b, c, h, w = x.size()               # get feature map x shape
 x = x.view(b, c, h*w)               # b * c * n   , shape of x
 mu = self.mu.repeat(b, 1, 1)        # b * k * n   , shape of base mu
 with torch.no_grad():
     for i in range(self.stage_num):
         x_t = x.permute(0, 2, 1)    # b * n * c
         z = torch.bmm(mu, x_t)      # b * k * c   , E-step
         z = F.softmax(z, dim=1)     # b * k * c
         z_ = z / (1e-6 + z.sum(dim=2, keepdim=True))
         mu = torch.bmm(z_, x)       # b * k * n   , m-step
         mu = self._l2norm(mu, dim=2)

 z_t = z.permute(0, 2, 1)            # b * c * k
 self.scale_weight = self.scale_weight.view(-1, self.k,1) #scale_weight is style code
 mu = mu * self.scale_weight         #base mu scaling with style code
 x = z_t.matmul(mu)                  # b * c * n  , R-step
 x = x.view(b, c, h, w)              # b * c * h * w
 x = F.relu(x, inplace=True)
```




# Usage

You should download the dataset and split images into "./data/trainA" and "./data/trainB" folders.

To train a model, use 

```
python train.py
```

To test, use

```
python test.py
```

# Evaluation Methods

[**Learned Perceptual Image Patch Similarity (LPIPS) metric**](https://github.com/richzhang/PerceptualSimilarity).

It measures the diversity of images.

**Note**: To reproduce our scores, you should use the weights of version "v0.0" they provided.



[**Naturalness Image Quality Evaluator (NIQE) no-reference image quality score**](https://github.com/WenlongZhang0724/RankSRGAN)

It measures the quality of images.

**Note**: We use the code of NIQE in RankSRGAN, and you should install complete matlab first if you want to run the code.



# Experiments

|      |           NIQE            | LPIPS |
| :--: | :-----------------------: | :---: |
| LFS  | 10.36(our implementation) | 0.109 |
| ours |           10.51           | 0.114 |



**2020.5.1 update**：

Add "moving average" for optimizing base mu.

|      |           NIQE            | LPIPS |
| :--: | :-----------------------: | :---: |
| LFS  | 10.36(our implementation) | 0.109 |
| ours |           10.74           | 0.142 |