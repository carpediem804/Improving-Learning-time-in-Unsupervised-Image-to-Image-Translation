# Improving Learning time in Unsupervised Image-to-Image Translation


## Abstract
 Unsupervised image-to-image translation can map
local textures between two domains, but typically fails when the
domain requires big shape changes. It is difficult to learn how to
make such big change using the basic convolution layer, and
furthermore it takes much time to learn. For faster learning and
high-quality image generation, we propose to use Cycle GAN that is
combined with Resnet in a network that is connected with the
residual block for upsampling to make big shape change and
construct faster image-to-image translation.


## our approach 
In unsupervised image-to-image translation, when the
shape of data images is very different, the encoderdecoder
format is used for learning. However, there is a
problem that it takes some time to learn because there is
no influence between layers. The main idea of Resnet's
idea is to use a residual block that
creates a skip connection so that the gradient can flow
well. This is similar to the Long Term Short Memory (LSTM),
which introduces a forget gate and so on to better flow the
gradient of the previous step. In this study, by using these
residual blocks in the generator upsampling, it is possible
to flow the big feature well, but it does not have big shape
transformation, and it is easy to learn because the image
quality is improved, the network is structured with high
speed. and a generator with residual blocks is constructed.
The architecture of this study is based on DiscoGAN and
CycleGAN. The existing DiscoGAN has the basic encoderdecoder
structure as shown in Figure 5, and it is easy to
change the format. However, since it has the basic decoder
structure of the network, it is difficult to learn upsampling
through the narrow bottleneck layer and output image
size. And it has a low resolution limit due to the low
capacity of the network. Therefore, the generator in this
study includes residual bocks in the upsampling layer, thus
making it easier to learn and be capable of producing
images of high quality.

## Generator model 
![image](https://user-images.githubusercontent.com/33194900/70114532-246e2c80-16a0-11ea-86be-1cf506c8fa7b.png)

## result 
### Total learning time and Morphological transformation error
| |Our|DiscoGAN|
|------|---|---|
|500 Epoch Time|41:54:60|61:41:44|
|Classification Error|8.92%|8.77%|

### result image example
![image](https://user-images.githubusercontent.com/33194900/70114603-5aabac00-16a0-11ea-90d1-601d98d26337.png)

### Resolution Result
![image](https://user-images.githubusercontent.com/33194900/70114633-6ac38b80-16a0-11ea-8a81-b12aa456a43a.png)


##Conclusion
In this study, we used a residual block in the upsampling
structure in the generator to maintain the shape
transformation and construct a network that takes much
less time to generate high quality images. One of the
problems in existing unsupervised image-to-image
translation, which takes a long learning time, is solved by
using a residual block, and the image can be converted at a
higher image quality than the existing research. In future
work, we plan to construct a more efficient network by
combining new ideas from generator or loss function as
well as generator structure.
