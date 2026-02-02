# ConvNeXt from Scratch

> What I cannot create, I do not understand.

Following the coding models from scratch series...

This project is a PyTorch implementation of the **ConvNeXt** architecture (ConvNeXt-Tiny transformed from ResNet-50), built from the scratch. The project follows the paper ["A ConvNet for the 2020s"](https://arxiv.org/pdf/2201.03545), demonstrating how to modernize a standard ResNet-50 into a state-of-the-art ConvNet.

The model is trained and tested on the **Food-101** dataset from scratch. Acheived around **74% top 1 accuracy**.

## Model Architecture
Steps followed for coding the model:

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
5. We use the model for training.

## Training Details

The training procedure of model, all these are sourced from the paper:
- **Optimizer**: AdamW optimizer.
- **LR Scheduling**: Linear Warmup for the first 20 epochs, followed by Cosine Annealing.
- **Augmentation**: MixUp & CutMix and Random Erasing to make model learn effectively.
- **Mixed Precision**: to speed up training in A100s.
- **Logging**: Integration with TensorBoard for tracking loss and accuracy.

## Project Structure

- **`model.py`**: Complete implementation of the `ConvNext` model.
- **`train.py`**: Implements the training loop
- **`dataset.py`**: Handles the Food-101 dataset loading. Loads the dataloader.
- **`config.py`**: loads the config
- **`inference.ipynb`**: notebook to load trained weights and perform inference on images
- **`collab_run.ipynb`**: training setup for training the model in a Google Colab,

## Usage

### 1. Requirements
Ensure you have Python installed along with the necessary dependencies:
- PyTorch
- Torchvision
- Tensorboard
- Tqdm

### 2. Training
To train the model locally:
1. Adjust settings in `config.py` as needed (e.g., `batch_size`, `model_folder`).
2. Run the script:
   ```bash
   python train.py
   ```

### 3. Inference
Open `inference.ipynb`, the notebook demonstrates how to:
- Instantiate the model and load weights.
- Preprocess input images.
- Visualize top-5 predictions for any given food image.

## Credits
- Architecture based on the paper *A ConvNet for the 2020s* 
- Dataset: [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).

---

**A ConvNet for the 2020s**