# Semantic Drone Dataset Segmentation using U-Net

This repository contains an implementation of the U-Net architecture for semantic segmentation of drone images from the Semantic Drone Dataset. The project aims to segment various classes such as buildings, roads, trees, and vehicles from aerial images.

## Dataset

The Semantic Drone Dataset consists of high-resolution aerial images captured by drones, along with their corresponding semantic segmentation masks. The dataset can be obtained from the following link:

[Semantic Drone Dataset](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset)

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
git clone https://github.com/your-username/semantic-drone-segmentation.git
cd semantic-drone-segmentation


2. Download the Semantic Drone Dataset and place it in the appropriate directory.

3. Run the Jupyter Notebook semantic_segmentation.ipynb to preprocess the data, train the U-Net model, and generate segmentation masks.

## Model Architecture

The U-Net architecture consists of an encoder and a decoder, with skip connections between the encoder and decoder layers. The encoder downsamples the input image, while the decoder upsamples the feature maps to produce the segmentation masks.

The model architecture can be customized by modifying the number of filters and layers in the build_unet function.

## Data Preprocessing

The dataset is preprocessed using the augment_data function, which performs data augmentation techniques such as random cropping, horizontal flipping, and vertical flipping. The preprocessed images and masks are stored in the ./new_data directory.

## Training

The model is trained using the tf_dataset function, which creates a TensorFlow dataset pipeline for efficient data loading and preprocessing. The training process can be monitored using TensorFlow callbacks such as ModelCheckpoint, ReduceLROnPlateau, and EarlyStopping.

## Results

The trained model generates segmentation masks for the test images, which are saved in the ./results directory. The segmentation masks can be visualized and evaluated using appropriate metrics.

Here are some example results:

<p align="center">
  <img src="https://github.com/Keshiiika/sementic_segmentation_drone_images/blob/main/results.png" alt="Original Image">
  <img src="https://github.com/Keshiiika/sementic_segmentation_drone_images/blob/main/result%202.png" alt="Original Image" width="250">
  <img src="https://github.com/Keshiiika/sementic_segmentation_drone_images/blob/main/result%203.png" alt="Original Image" width="250">
  <img src="https://github.com/Keshiiika/sementic_segmentation_drone_images/blob/main/result%204.png" alt="Original Image" width="250">
  <img src="https://github.com/Keshiiika/sementic_segmentation_drone_images/blob/main/result%205.png" alt="Original Image" width="250">
  <img src="https://github.com/Keshiiika/sementic_segmentation_drone_images/blob/main/result%206.png" alt="Original Image" width="250">
  <img src="https://github.com/Keshiiika/sementic_segmentation_drone_images/blob/main/result%207.png" alt="Original Image" width="250">
</p>

The left image shows the original input image, the middle image shows the ground truth segmentation mask, and the right image shows the predicted segmentation mask generated by the U-Net model.

## Contributions

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
