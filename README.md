# WeatherNet-PyTorch

A Convolutional Neural Network (CNN) model for weather prediction, implemented in PyTorch. This model utilizes the `dc-weather-prediction` dataset from Hugging Face to predict weather attributes from satellite images.

## Installation

1. Clone this repository:
```
git clone https://github.com/<your-username>/WeatherNet-PyTorch.git
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

To train the model, run the following command:

```
python train.py
```

## Dataset

The model is trained on the `dc-weather-prediction` dataset from Hugging Face, which contains satellite images and corresponding weather attributes.

## Model

The CNN architecture is designed to extract features from satellite images and predict weather attributes. It includes convolutional layers, batch normalization, dropout layers, and fully connected layers.

## Training

The model is trained with a mean squared error loss function and an Adam optimizer. A learning rate scheduler is employed to adjust the learning rate based on the validation loss.

## Acknowledgments

This project utilizes the `dc-weather-prediction` dataset available on Hugging Face.

## License

This project is open-sourced under the MIT License. See the LICENSE file for more detail
