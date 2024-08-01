# Project Overview

This document provides essential information on the model, training script, and guidelines for team members involved in the celestial body detection project using Faster R-CNN with ResNet50 backbone.

## Model and Training Script

Our project utilizes a Faster R-CNN model with a ResNet50 backbone for detecting various celestial bodies in images. The training script, `train.py`, is specifically designed for this task.

## Image Dimension Assumption

The `train.py` script assumes all training images are 256 pixels in width and 144 pixels in height, and the object is always located in the near-centre of the image.

## Training Environment

Training this model effectively without a CUDA-enabled GPU might pose challenges. For those lacking suitable hardware, Google Colab is recommended as it is already configured for our training needs.

## Training Duration

For accurate differentiation between closely related classes, such as Mercury and the Moon, it is advisable to increase the number of training epochs. This helps the model in learning finer distinctions.

## Model Access

Due to GitHub's file size restriction of 100MB, the trained model is hosted externally. You can download it using [this link](https://drive.google.com/file/d/1hWsvUoG82yvRbd0EhVfHnu5zrqpsoR9u/view?usp=sharing). Ensure to integrate it with your local setup for further use.

## Model Export and GitHub Limitations

Encountering GitHub's file size limit might require employing model compression techniques or exploring transfer learning with a simpler backbone model for easier sharing and deployment.

## Integration with ROS

The aim is to integrate our model with ROS (Robot Operating System) for inferencing purposes. Code from the `process_image` function within `demo_app/app.py` could be essential for this. Assistance from those familiar with ROS in adapting and testing this functionality is encouraged.

## Next Steps

- Review the model and training script.
- Adjust training parameters and the number of epochs as necessary to enhance class differentiation.
- Explore options for efficient and accurate inferencing by integrating the model into our ROS setup.

