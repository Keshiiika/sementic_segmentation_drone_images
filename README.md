# Semantic Drone Dataset Segmentation using U-Net

This repository hosts an implementation of the U-Net architecture tailored for semantic segmentation of drone images sourced from the Semantic Drone Dataset. The project's objective is to accurately delineate various classes such as buildings, roads, trees, and vehicles from aerial images.

Semantic image segmentation endeavors to assign a label to every pixel within an image, indicating the class it represents. This task, often termed as dense prediction, entails predicting for each pixel in the image.

In semantic segmentation, the anticipated output transcends mere labels and bounding box parameters. Instead, the output manifests as a high-resolution image, typically matching the input image's dimensions. In this output, every pixel is assigned to a specific class, constituting a pixel-level image classification.

## Dataset

The dataset can be obtained from the following link:

[Semantic Drone Dataset](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset)

The Semantic Drone Dataset focuses on enhancing the safety of autonomous drone navigation and landing procedures by emphasizing semantic comprehension of urban environments. The dataset comprises imagery showcasing over 20 houses captured from a bird's eye view at altitudes ranging from 5 to 30 meters above ground level. Images are obtained using a high-resolution camera, yielding a size of 6000x4000 pixels (24 megapixels). The training dataset encompasses 400 publicly accessible images, while the test dataset comprises 200 private images.

For person detection, the dataset includes bounding box annotations for both the training and test sets. Additionally, pixel-accurate annotations are provided for semantic segmentation tasks for the same sets. The dataset's complexity is streamlined to 20 distinct classes, as outlined in Table 1 below.

### Table 1: Semantic Classes of the Drone Dataset

1. Tree
2. Grass
3. Other vegetation
4. Dirt
5. Gravel
6. Rocks
7. Water
8. Paved area
9. Pool
10. Person
11. Dog
12. Car
13. Bicycle
14. Roof
15. Wall
16. Fence
17. Fence pole
18. Window
19. Door
20. Obstacle

## Requirements

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- Albumentations
- Scikit-learn
- Pandas
- Numpy
- Matplotlib

## Usage

1. Clone the repository:

bash
git clone https://github.com/Keshiiika/sementic_segmentation_drone_images.git
cd semantic-segmentation-drone-images


2. Download the Semantic Drone Dataset and place it in the appropriate directory.

3. Run the Jupyter Notebook semantic_segmentation.ipynb to preprocess the data, train the U-Net model, and generate segmentation masks.

## Data Preprocessing

To prepare the data for training, the following steps were performed:

1. *Data Augmentation*: The original images were resized to 1536x1024 pixels while maintaining the aspect ratio. Data augmentation techniques, such as random cropping, horizontal flipping, and vertical flipping, were applied to increase the diversity of the training data and improve model generalization.

2. *Dataset Split*: The augmented dataset was split into train, validation, and test sets, with an 80-10-10 ratio, respectively.

3. *Dataset Pipeline*: A TensorFlow dataset pipeline was created to efficiently load and preprocess the images and masks during training. This included operations such as shuffling, mapping, batching, and prefetching.

## Model Architecture

The U-Net architecture was used for this project, which is a popular choice for image segmentation tasks. The model consists of an encoder-decoder structure with skip connections between the encoder and decoder paths. The encoder path captures the context and spatial information from the input image, while the decoder path generates the segmentation masks.

The U-Net model was built using TensorFlow and Keras, with the following key components:

- *Encoder*: The encoder path consists of a series of convolutional blocks, each followed by batch normalization, ReLU activation, and max pooling layers. The number of filters increases as the depth increases, allowing the model to capture more complex features.

- *Bridge*: The bridge connects the encoder and decoder paths without downsampling.

- *Decoder*: The decoder path consists of upsampling layers followed by concatenation with the corresponding encoder features (skip connections). This helps the model retain spatial information and localization details.

- *Output Layer*: The final layer uses a 1x1 convolution with softmax activation to produce the segmentation mask with the desired number of classes.

The model architecture can be customized by modifying the number of filters and layers in the build_unet function.

## Training

The U-Net model was trained using the TensorFlow dataset pipeline, with the following hyperparameters:

- Optimizer: Adam
- Learning Rate: 1e-4
- Batch Size: 4
- Epochs: 30
- Loss Function: Categorical Cross-Entropy
- Metrics: Accuracy

During training, callbacks were used to save the best model, reduce the learning rate on plateau, and implement early stopping to prevent overfitting. The training process can be monitored using TensorFlow callbacks such as ModelCheckpoint, ReduceLROnPlateau, and EarlyStopping.

## Results

The trained model generates segmentation masks for the test images, which are saved in the ./results directory. The segmentation masks can be visualized and evaluated using appropriate metrics.

Here are some example results:

<p align="center">
  <img src="https://github.com/Keshiiika/sementic_segmentation_drone_images/blob/main/results.png" alt="Original Image">
  <img src="https://github.com/Keshiiika/sementic_segmentation_drone_images/blob/main/result%202.png" alt="Original Image">
  <img src="https://github.com/Keshiiika/sementic_segmentation_drone_images/blob/main/result%203.png" alt="Original Image">
  <img src="https://github.com/Keshiiika/sementic_segmentation_drone_images/blob/main/result%204.png" alt="Original Image">
  <img src="https://github.com/Keshiiika/sementic_segmentation_drone_images/blob/main/result%205.png" alt="Original Image">
  <img src="https://github.com/Keshiiika/sementic_segmentation_drone_images/blob/main/result%206.png" alt="Original Image">
  <img src="https://github.com/Keshiiika/sementic_segmentation_drone_images/blob/main/result%207.png" alt="Original Image">
</p>

All the above images shows the predicted segmentation mask generated by the U-Net model.

## Group Members

- Solanki Honey (B21EE068)
- Keshika Patwari (B21CS039)
