# Semantic Segmentation of Cityscapes Dataset

## Project Description
This project aims to perform semantic segmentation on the Cityscapes dataset, which is a large-scale dataset for urban scene understanding. The goal is to accurately classify each pixel in the images into one of the predefined semantic classes, such as road, sidewalk, building, car, pedestrian, etc. By leveraging deep learning techniques, we aim to develop a model that can effectively segment urban scenes and provide valuable insights for various applications.

## Prerequisites
- Python 3
- TensorFlow 
- OpenCV 
- Other dependencies 

## Dataset
The Cityscapes dataset contains high-quality pixel-level annotations for 5,000 images, captured from different cities. The dataset includes various weather conditions, viewpoints, and camera setups. You can download the dataset from the official website: [Cityscapes Dataset](https://www.cityscapes-dataset.com/)

## Model Architecture
For this project, we utilize the U-NET and GAN architecture, which has shown excellent performance on semantic segmentation tasks. 

## Training
To train the semantic segmentation model on the Cityscapes dataset, follow these steps:
1. Download the Cityscapes dataset using the provided link and extract the images and annotations.
2. Preprocess the dataset by resizing the images and generating the corresponding pixel-level labels.
3. Split the dataset into training, validation, and test sets based on your preferred ratio.
4. Initialize the model with the modified U-NET and GAN architecture.
5. Applied Semi-Supervision and Active Learning on these models.
6. Set the hyperparameters, such as learning rate, batch size, and number of epochs.
7. Use the chosen optimizer and loss function to train the model on the training set.
8. Evaluate the model's performance on the validation set and make adjustments if necessary.
9. Test the final trained model on the test set and report the evaluation metrics, such as mean intersection over union (mIoU) and pixel accuracy.

Feel free to modify the hyperparameters, experiment with different model architectures, or apply additional techniques to further enhance the results.

## Results
Mean IOU (intersection over union) exceeds 80%
Pixel Accuracy>90%
Dice Coefficient exceeds 0.8
User Feedback using Active Learning
