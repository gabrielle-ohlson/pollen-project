import argparse
import cv2
import math
import numpy as np
import os
import random

from preprocess import remove_bg, process_samples, generate_normalizer

from helpers import list_images, list_species_names, get_magnification, load_img, resize_img, save_species_img, show_img, unique_colors, cvt_img_format, edge_color

from aug_transforms import horiz_flip, vert_flip, rand_rotate_fill_bg, rand_rotate_strict, rand_rotate_max_area, rand_crop

# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--path", help="Absolute or relative path to the folder containing pollen slide images.", type=str, default="Pollen Slides/") # "Pollen Slides/Calystegia macrostegia"

parser.add_argument("-s", "--save_path", help="Absolute or relative path to the folder for saving the output images.", type=str, default="data")

parser.add_argument("-t", "--target_path", help="Path to target image from normalizing sample stains.", type=str, default="target_imgs/ML102115 40X1C 498_1.JPG")

parser.add_argument("-r", "--resize_width", help="The width to resize all images to (so that the size of cropped samples is consistent with the species).", type=int, default=1024)

parser.add_argument("-m", "--magnification", help="Magnification for images that will be processed.\nInput -1 to use any magnification.", type=int, default=40) #-1 for any magnification


parser.add_argument("-a", "--aug_count", help="Number of augmentations to generate per-image (input 0 for no augmentation).", type=int, default=1)

parser.add_argument("-b", "--balance_datasets", help="Option to add extra augmentations for species with less images to fix imbalanced datasets.", action="store_true")

parser.add_argument("-c", "--crop", help="Use random crop as a data augmentation transform\n(not currently recommended, since the system relies on crop-uniformity due to limited data. Consider usage when more data is available).", action="store_true")


parser.add_argument("-d", "--debug", help="Option to include extra verbose that might be helpful for debugging.", action="store_true")


args = parser.parse_args()

# ------------------------------------------------------------------------------

path = os.path.normpath(args.path)
save_path = os.path.normpath(args.save_path)
target_path = args.target_path
resize_width = args.resize_width
magnification = args.magnification

aug_count = args.aug_count
balance_datasets = args.balance_datasets
crop_aug = args.crop

debug = args.debug

# ------------------------------------------------------------------------------

parent_path, main_dir = os.path.split(path)


if len(target_path): normalizer = generate_normalizer(target_path)
else: normalizer = None

species_names = list_species_names(path)

total_img_ct = 0 #for now

input_imgs = {}
output_imgs = {}

# ------------------------------------------------------------------------------

for species in species_names:
  os.makedirs(f'{save_path}/lbp/{species}', exist_ok=True)
  os.makedirs(f'{save_path}/processed/{species}', exist_ok=True)

  input_imgs[species] = list_images(f'{path}/{species}', magnification=magnification)

  total_img_ct += len(input_imgs[species])

# ----------
transforms = [
  horiz_flip,
  vert_flip,
  rand_rotate_fill_bg
]

if crop_aug:
  transforms.append(rand_rotate_max_area)
  transforms.append(rand_crop)

extra_augs = {}

if balance_datasets:
  max_img_ct = len(max(input_imgs.values(), key=len))
  print(f'max image count for individual species: {max_img_ct}.\n')


def augment(image, transforms, aug_count, bg_color=255, include_original=True):
  imgs = []
  
  if include_original: imgs.append(image)

  img_format = getattr(image, '__module__', 'cv2').split('.')[0]

  PIL_img = image.copy()

  if img_format != 'PIL': PIL_img = cvt_img_format(PIL_img)

  for idx in range(0, aug_count):
    t_img = PIL_img.copy()

    for t in transforms:
      try:
        t_img = t(t_img, bg_color=bg_color)
      except Exception:
        t_img = t(t_img)

    if img_format == 'cv2': t_img = cvt_img_format(t_img)

    imgs.append(t_img)

  return imgs

# ------------------------------------------------------------------------------

img_idx = 1

species_samples = {}

for species, img_paths in input_imgs.items():
  file_idxs = []
  species_imgs = []
  species_imgs_lbp = []

  img_ct = len(img_paths)

  balance_ct = 0

  if balance_datasets:
    balance_ct = (max_img_ct-img_ct)
    print(f'extra augmentations for species {species}: {balance_ct}.')
  balance_per_img = math.ceil(balance_ct/img_ct)

  for i, img_path in enumerate(img_paths):
    print(f'[{img_idx}/{total_img_ct}] species: {species}\n\tpath: {img_path}')

    img = load_img(img_path)
    resized_img = resize_img(img, resize_width=resize_width)
    no_bg_img = remove_bg(resized_img)


    extra_augs = min(balance_per_img, balance_ct)
    balance_ct -= extra_augs

    bg_color = tuple(edge_color(no_bg_img))
    t_ct = random.randint(1, len(transforms)) # select `t_ct` number of transforms from entire list
    img_transforms = random.sample(transforms, t_ct)

    imgs = augment(no_bg_img, img_transforms, extra_augs+aug_count, bg_color)

    first_img = True
    for im in imgs:
      cropped_processed, cropped_lbp = process_samples(im, normalizer=normalizer, LBP=True, debug=debug)

      sample_ct = len(cropped_processed)

      if sample_ct:
        if first_img:
          print(f'found {sample_ct} sample(s).')
          first_img = False

        file_idxs.extend([i for _ in range(sample_ct)])
        species_imgs.extend(cropped_processed)
        species_imgs_lbp.extend(cropped_lbp)      
  
    img_idx += 1


  sample_sizes = np.array([img.size for img in species_imgs])

  med_size = np.median(sample_sizes)
  valid = np.argwhere((sample_sizes/med_size) >= 0.2)

  med_size = np.median(sample_sizes[valid])
  large_invalid = np.argwhere((med_size/sample_sizes) < 0.2)

  species_imgs = [img for i, img in enumerate(species_imgs) if i in valid and i not in large_invalid]
  species_imgs_lbp = [img for i, img in enumerate(species_imgs_lbp) if i in valid and i not in large_invalid]
  file_idxs = [val for i, val in enumerate(file_idxs) if i in valid and i not in large_invalid]

  species_samples[species] = {
    'img_ct': len(species_imgs_lbp),
    'sample_imgs': species_imgs_lbp,
    'img_paths': []
  }


  for (im, lbp_im, file_idx) in zip(species_imgs, species_imgs_lbp, file_idxs):
    img_path = img_paths[file_idx]
    
    if len(unique_colors(cv2.threshold(lbp_im, 100, 255, cv2.THRESH_BINARY)[1])) > 1:
      save_species_img(im, f'{save_path}/processed', species, img_path, idx=0)
      save_species_img(lbp_im, f'{save_path}/lbp', species, img_path, idx=0)

      species_samples[species]['img_paths'].append(img_path)


# now add more augmentation for lbp samples (with no cropping and only rotations at 90deg steps) to balance_datasets for samples, if necessary
if aug_count > 0 and balance_datasets:
  sample_transforms = [horiz_flip, vert_flip, rand_rotate_strict]

  max_sample_ct = max(species_samples.values(), key=lambda x: x['img_ct'])['img_ct']

  for species, data in species_samples.items():
    balance_ct = (max_img_ct-data['img_ct'])
    print(f'extra sample augmentations for species {species}: {balance_ct}.')
    balance_per_img = math.ceil(balance_ct/img_ct)

    for im, img_path in zip(data['sample_imgs'], data['img_paths']):
      extra_augs = min(balance_per_img, balance_ct)
      balance_ct -= extra_augs

      t_ct = random.randint(1, len(sample_transforms)) # select `t_ct` number of transforms from entire list
      img_transforms = random.sample(sample_transforms, t_ct)

      if extra_augs > 0:
        imgs = augment(im, img_transforms, extra_augs, include_original=False)

        for aug_im in imgs:
          save_species_img(aug_im, f'{save_path}/lbp', species, img_path, idx=0)

      if balance_ct < 1: break
