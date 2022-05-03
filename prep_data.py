from PIL import Image

from helpers import load_img, resize_img

from preprocess import process_samples, remove_bg


def process_img(img_path, normalizer=None, resize_width=1024, LBP=True):
  img = load_img(img_path)

  resized_img = resize_img(img, resize_width=resize_width)
  no_bg_img = remove_bg(resized_img)

  if LBP: _, cropped_samples = process_samples(no_bg_img, normalizer=normalizer, LBP=True)
  else: cropped_samples = process_samples(no_bg_img, normalizer=normalizer, LBP=False)

  cropped_samples = [Image.fromarray(im) for im in cropped_samples]

  return cropped_samples
