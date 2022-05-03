# Setup

First, clone the GitHub repo as so:
```
git clone https://github.com/gabrielle-ohlson/pollen-project.git
```

or, using SSH:
```
git clone git@github.com:gabrielle-ohlson/pollen-project.git
```


Then, install [anaconda](https://www.anaconda.com/), and create a new virtual environment with the command:
<!-- In the console, run the command: -->
```console
conda create --name pollen_project --file environment.yml
```
<!-- conda create --name pollen_project python=3.8.5 ipython
```

Then, install the necessary dependencies:
```console
```

conda create --name pollen_project --file requirements.txt -->
(this will create a new environment by the name of "pollen_project" and install the necessary dependencies)

Note that this system uses python version 3.8.5.


To activate the enviornment, use the command:
```console
conda activate pollen_project
```

# Background

Due to the highly textural qualities of pollen slides, my approach draws from texture classification methods in computer vision.  In particular, I utilize Local Binary Patterns (LBP) to encode feature detectors that distinguish between various classification categories.

### Pre-processing

To pre-process the data, we run the `pollen.py` file.  The command takes optional arguments such as the path to the input data folder ( `--path`), path to a "target image" used for reference when normalizing the appearance of various pollen grain stains (`--target_path`), the width to resize all images to (so that the size of cropped samples is consistent with the species) (`--resize_width`), the magnification to use (once again, for consistency) (`--magnification`), and the path for saving the output data.

To increase the amount of training data and create uniform images, the pre-processing program computes the bounding boxes of each individual pollen sample in a given image and crops accordingly.  Various precautions are taken to prevent erroneous cropping, though a limited amount of failures do occur.  Additionally, the light conditions are made more consist and the background of images is made a uniform white color by rescaling the exposure intensity using the Python library [scikit-image]({https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.rescale_intensity).  This method is also helpful in ensuring the individual samples are cropped accurately, by removing any debris surrounding the samples on the slides.

After the pre-processing steps are complete, additional steps are taken to remove poorly processed training data, comparing the sample images for each given species with each other to eliminate any outliers that possess significantly deviant image properties.  The final result is a reasonably sizable dataset with individually-cropped pollen grains, all processed to accentuate the textural qualities of samples from a given species and eliminate image variations that are not pertinent to identification, thus reducing the reliance on a high-volume of data.

To compute the LBP representation of each cropped sample image, I use the feature module from scikit-image [skimageLBP](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_local_binary_pattern.html).  With this information, I train a CNN to make predictions for image classification.

Image augmentation can optionally be incorporated to assist in the robustness of the model by supplementing the currently limited data available.  Augmentation techniques include horizontal and vertical flipping and random rotation; special steps are taken to ensure rotation does not crop the samples and instead pads with the white background color, but cropping augmentations can be applied as well (though it is not currently recommended, since the system relies on crop-uniformity due to limited data.  Consider usage when more data is available).  The particular augmentations applied in a given instance are randomly selected for each image.

To apply augmentation transformations, users can assign a value greater than zero to the argument `--aug_count` when running the `pollen.py` (`--aug_count 0` will exclude augmentation from the workflow).  Additionally, the boolean argument `--balance_datasets` can be passed to fix imbalanced species datasets by adding extra augmentations for species with fewer images.

# Usage

### Pre-processing

Pre-processing is achieved with the `pollen.py` executable file.

**Example Usage:** (with default argument values)
```console
python pollen.py 
```

**Arguments:**
```console
usage: pollen.py [-h] [-p PATH] [-s SAVE_PATH] [-t TARGET_PATH] [-r RESIZE_WIDTH] [-m MAGNIFICATION] [-a AUG_COUNT] [-b] [-c] [-d]

optional arguments:
  -h, --help             show this help message and exit
  -p, --path PATH        Absolute or relative path to the folder containing pollen slide images.
  -s, --save_path        Absolute or relative path to the folder for saving the output images.
  -t, --target_path      Path to target image from normalizing sample stains.
  -r, --resize_width     The width to resize all images to (so that the size of cropped samples is consistent with the species).
  -m, --magnification    Magnification for images that will be processed. Input -1 to use any magnification.
  -a, --aug_count        Number of augmentations to generate per-image (input 0 for no augmentation).
  -b, --balance_datasets Option to add extra augmentations for species with less images to fix imbalanced datasets.
  -c, --crop             Use random crop as a data augmentation transform (not currently recommended, since the system relies on crop-uniformity due to limited data. Consider usage when more data is available).
  -d, --debug            Option to include extra verbose that might be helpful for debugging.
```

**Recommended Usage:**
```console
python pollen.py --balance_datasets
```

### Model Training

We use transfer learning on a pre-train [resnet18](https://pytorch.org/hub/pytorch_vision_resnet/) pytorch model.  This is done with the iPython notebook `transfer_learning_pytorch.ipynb`.  This file was adapted from the pytorch ["Transfer Learning for Computer Vision Tutorial"]().

The only variables that the user will likely set are: `data_dir` (path to the pre-processed sample images) and `load_path` (path to trained model from prior use; `None` if training the model).

### Predicting 

After training the model, we can predict the species of a particular image with the `predict.ipynb` notebook.  

The variable `test_img_path` should be set as the path to the input-image the user would like to predict (*without any pre-processing*).  `main_dir` is the name of the folder containing the species sub-directories. \
The variable `load_path` should be set as the path to the saved model from training (as described above).

The program will then apply the pre-processing steps taken for the training images, and predict the species for each sample.  The final output will be the top prediction among all of the cropped samples in the given images.
