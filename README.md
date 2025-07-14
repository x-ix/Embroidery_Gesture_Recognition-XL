### Introduction

Transformer-XL-based temporal encoder for binary gesture classification within the context of a real-time intelligent coaching system. Google’s MediaPipe was used to extract frame-wise hand landmarks from two fixed camera angles (top-down and bottom-up) capturing tambour embroidery sequences. These landmarks—21 keypoints per hand per view—were flattened, concatenated, and assembled into temporal matrices of shape (T, 126), where T denotes the sequence length and 126 results from stacking both camera perspectives.

The dataset consists of expert-performed embroidery gestures, split into perfect executions and intentional errors following predefined criteria. These were segmented and labelled to isolate target gestures as positive instances and unrelated motion as negative. The model architecture employs stacked Transformer-XL layers with relative positional encoding and a memory module to preserve long-range temporal dependencies. A feedforward classifier maps sequence encodings to gesture classes.

Inference is structured to simulate dual-camera video streams and can be adapted to live input with minimal modification. Training and evaluation were conducted using CUDA 11.8 and Python 3.11.11.


### Installation

1. Clone this repo:
```bash
git clone https://github.com/x-ix/Embroidery_Gesture_Recognition-XL.git
```
2. Install dependencies (please refer to [requirements.txt](requirements.txt)):
```bash
pip install -r requirements.txt
```

### Usage
```
Usage:
        python inference.py - You can run this straight away to see performance with the included
                              pre-trained model, this file has been configured to assist with
                              integration into a larger application thus has not been optimised
                              for individual execution, that being said if testing model performance
                              on other video data is desired, simply change the input paths and
                              execute.



        Dataset has already been included but if you would like to recreate it anyway follow
        the execution order below:

                python constructor.py - Downloads and upacks required video data (2.3gb)


                python landmarks.py - Extracting Hand Landmarks, also provides annotated output
                                      for visual clarity which can be toggled off in the switches


                python dataset.py - Manipulates data and constructs into final dataset, also
                                    conducts some basic exploration


                python dataset_analysis.py - Optional, exploratory data analysis, go back and
                                             forth between this and altering dataset.py if
                                             applying your own changes to the dataset's constructors
                                             or just to see what the data is like.



        python train.py - Training script, hyperparameters can be changed at the top, remember to
                          reflect hyperparameter changes in inference.py. If seeking to tune
                          hyperparameters set HPTUNING to True which will activate k-fold cross
                          validation and disable model output. Optimiser testing can also be
                          activated by setting both HPTUNING and OTUNING to True. If these are
                          False, regular training and saving will occur.

```


### Important Notes

- During inference the model recieves and analyses matrices of the form (1,126), this has been implemented in inference.py already.

- norm_stats.pt contains the mean and std used during training and has been applied to input matrices just prior to inference, if adapting the algorithm please retain this normalisation for accurate model performance.

- The video files "Top-Down" and "Bottom-up" are used to simulate dual channel streaming and can be swapped out with either a live feed or video files of upward stitches with similar perspectives.

- I haven't tested this using cuda nor python versions outside of 11.8 and 3.11.11 respectively, thus cannot comment on the programs functionality outside of those environments.



### Miscellaneous
Contents of [requirements.txt](requirements.txt):
```
--index-url https://download.pytorch.org/whl/cu118
torch==2.0.1 
torchvision==0.15.2
torchaudio==2.0.2

--extra-index-url https://pypi.org/simple
mediapipe==0.10.21
numpy==1.26.4
opencv-python==4.11.0.86
pandas==2.1.4
matplotlib==3.10.3
seaborn==0.13.2
scikit-learn==1.7.0
scipy==1.15.3
transformers==4.53.1
tqdm==4.67.1
wandb==0.20.1
pytorch-lamb==1.0.0
```


### Closing Notes
Performance is okay but still needs improvement.
