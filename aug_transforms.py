import cv2
import math
import numpy as np
from PIL import Image, ImageOps
import random
import torchvision

from helpers import cvt_img_format

# ------------------------------------------------------------------------------

horiz_flip = torchvision.transforms.RandomHorizontalFlip(p=1) # p is probably of flipping (0 <= p <= 1)

vert_flip = torchvision.transforms.RandomHorizontalFlip(p=1) # p is probably of flipping (0 <= p <= 1)

# -----------
def rand_crop(image):
  width, height = image.size

  crop_w = int(random.uniform(0.6, 1.0) * width)
  crop_h = int(random.uniform(0.6, 1.0) * height)

  crop = torchvision.transforms.RandomCrop((crop_h, crop_w), fill=0, padding_mode='constant')

  return crop(image)
  

def rand_rotate_fill_bg(image, bg_color=255):
  degrees = random.randint(0, 360)
  rand_rotation = torchvision.transforms.RandomRotation(degrees, fill=bg_color)

  return rand_rotation(image)


def rand_rotate_strict(image):
  rotations = [0, 90, 180, 270]

  degrees = random.randrange(0, 360, step=90)

  rand_rotation = torchvision.transforms.RandomRotation(degrees, fill=255)

  return rand_rotation(image)

  
# -----------
def rotate_bound(image, angle):
  # source: https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
  (h, w) = image.shape[:2]
  # h, w = image.size
  (cX, cY) = (w // 2, h // 2)
  M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
  cos = np.abs(M[0, 0])
  sin = np.abs(M[0, 1])
  nW = int((h * sin) + (w * cos))
  nH = int((h * cos) + (w * sin))
  M[0, 2] += (nW / 2) - cX
  M[1, 2] += (nH / 2) - cY
  return cv2.warpAffine(image, M, (nW, nH))



def rotatedRectWithMaxArea(w, h, angle):
  # source: https://stackoverflow.com/a/16778797
  """
  Given a rectangle of size wxh that has been rotated by 'angle' (in
  radians), computes the width and height of the largest possible
  axis-aligned rectangle (maximal area) within the rotated rectangle.
  """
  if w <= 0 or h <= 0:
    return 0,0

  width_is_longer = w >= h
  side_long, side_short = (w,h) if width_is_longer else (h,w)

  # since the solutions for angle, -angle and 180-angle are all the same,
  # if suffices to look at the first quadrant and the absolute values of sin,cos:
  sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
  if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
    # half constrained case: two crop corners touch the longer side,
    #   the other two corners are on the mid-line parallel to the longer line
    x = 0.5*side_short
    wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
  else:
    # fully constrained case: crop touches all 4 sides
    cos_2a = cos_a*cos_a - sin_a*sin_a
    wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

  return wr, hr



def rand_rotate_max_area(image):
  """ image: cv2 image matrix object
      angle: in degree
  """
  degrees = random.randint(0, 360)

  cv2_img = cvt_img_format(image) #convert to numpy array (cv2 format)

  # CREDIT: https://stackoverflow.com/a/48101555
  wr, hr = rotatedRectWithMaxArea(cv2_img.shape[1], cv2_img.shape[0], math.radians(degrees))

  rotated = rotate_bound(cv2_img, degrees)
  h, w, _ = rotated.shape
  y1 = h//2 - int(hr/2)
  y2 = y1 + int(hr)
  x1 = w//2 - int(wr/2)
  x2 = x1 + int(wr)
  result = rotated[y1:y2, x1:x2]

  return cvt_img_format(result) #convert back to PIL

# ------------------------------------------------------------------------------

default_transforms = [
  horiz_flip,
  vert_flip,
  rand_rotate_fill_bg
]