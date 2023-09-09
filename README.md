# image-classification-models

Deep Learning based image classification using popular architectures like VGG16, VGG19, and ResNet50.

## Overview
This project aims to classify images into five categories: cats, dogs, birds, fish, and insects. It utilizes pre-trained models from TensorFlow to extract features and then trains a custom classification head on top of these features.

## Features
- Uses the architectures VGG16, VGG19, and ResNet50.
- Employs transfer learning to reduce training time and prevent overfitting.
- Includes both training and prediction functionalities.
- Utilizes data augmentation to increase the robustness of the model.

## Getting Started
1. Clone the repository:
   ```
   git clone https://github.com/ai-akhmadullin/image-classification-models.git
   cd image-classification-models
   ```
2. (Recommended) Set up a virtual environment

   Create a virtual environment:
   ```python -m venv myvenv```
   
   Activate the virtual environment:
   - macOS and Linux:
     ```source myvenv/bin/activate```
   - Windows:
     ```.\myvenv\Scripts\activate```
5. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Project

- To train the model (valid models are: 'vgg16', 'vgg19', 'resnet'), run:
```
python main.py --train=True --model='vgg16'
```

- To classify an image (choose the model that you have trained, for this case it is 'vgg16'):
```
python main.py --image path_to_image.jpg --model='vgg16'
```

## Results

Results, along with models accuracy, loss graphs, and some prediction samples, can be found in `Summary.pdf`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
