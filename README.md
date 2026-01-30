Project in progress...

Steps followed:
1. As in paper started to code ResNet50 from scratch.
2. In ResNet paper, the ResNet 50 uses bottleneck block instead of basic block, stem block as the entey preprocessing, also implement shortcut when the channel size differs due to increse in channel size and then finally the classification head.
3. The ConvNexT model has following difference from the ResNet:
    - **Training Techniques**, the model is trainied similar to how a DeiT or Swin Transformers are trained. Adam optimizers, Mixup, Cutmix and Label Smoothing.
    -  **Changing the Stage compute ratio**, The standard ResNet50 uses these stage ratios (3, 4, 6, 3) and instead we now use these ratio (3, 3, 9, 3), this aligns with FLOPs with Swin-T
    - **Changing Stem to "Patchify"**, instead of the stem 7x7, stride 2 the paper suggest using patchify layer which is 4x4 stride 4.
    - **ResNeXt-ify**, the paper suggest on using the Depthwise convolution which turn the calculation similar to transformers, mixing information only in the specific channel.
    - **Inverted Bottlenecks**, this is directly from transformers models this change reduces flop nad increses accuracy
    - **Large Kernel Sizes**, this is from swin-T architecture. Moving the depthwise convolution layer and then increasing the kernel size to 7x7.
    - **Micro Designs**, these include changing ResNet internal compenents to moderen standards. Like changing ReLU to GeLU, fewer activation functions, Fewer normalitation layers, using LN instaed of BN and downsampling layers.
4. Following above changes to ResNet is dubbed as ConvNexT as quoted in paper.