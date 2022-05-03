import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
from scipy import stats
from typing import Iterable


from skimage.exposure import rescale_intensity
from skimage.filters import threshold_yen


def get_magnification(img_path):
  img_file = os.path.split(img_path)[-1]
  ext_idx = img_file.rfind('.')
  filename, ext = img_file[:ext_idx], img_file[ext_idx:]

  try:
    magnification = int(filename.split(' ')[1].split('X')[0])
  except Exception:
    return None

  return magnification


def list_images(base_path, valid_exts=('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'), exclude_dirs=[], include_subdirs=True, magnification=-1): # exclude_dirs=['cropped']
  paths = []
  base_path = os.path.normpath(base_path)

  for root_dir, dirs, files in os.walk(base_path, topdown=True):

    files = [f for f in files if f[0] != '.']
    
    if not include_subdirs and os.path.normpath(root_dir) != os.path.normpath(base_path): break
    
    dirs[:] = [d for d in dirs if d[0] != '.' and d not in exclude_dirs]

    dirs.sort() #so goes alphabetically
    
    for file in files:
      ext = file[file.rfind('.'):].lower()

      if not ext.endswith(valid_exts) or (magnification > -1 and get_magnification(file) != magnification): continue
      else: paths.append(os.path.join(root_dir, file))

  return paths


def load_img(img):
  return cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)


# ## Source: `normalize` function from HMC CS153 'point_operators.ipynb' demo:
def normalize(img, Lmax, Lmin):
  img_out = (img - img.min()) * (Lmax - Lmin) / (img.max() - img.min()) + Lmin
  return img_out


def reduce_colors(im, col_ct=3):
  # source: https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html
  im_copy = np.copy(im)

  if len(im.shape) < 3: im_copy = cv2.cvtColor(normalize(im_copy, 255, 0).astype('uint8'), cv2.COLOR_GRAY2RGB)
  
  Z = im_copy.reshape((-1,3))

  # convert to np.float32
  Z = np.float32(Z)
  # define criteria, number of clusters(K) and apply kmeans()
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

  ret,label,center=cv2.kmeans(Z, col_ct, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  # Now convert back into uint8, and make original image
  center = np.uint8(center)
  res = center[label.flatten()]
  res2 = res.reshape((im_copy.shape))

  if len(im.shape) < 3: res2 = normalize(cv2.cvtColor(res2, cv2.COLOR_RGB2GRAY), np.amax(im), np.amin(im)) # TODO:
  
  return res2


# estimate lower and upper range for HSV color
def hsv_thresh(color_tuple, hue_shift=26, value_shift=5):
  # deconstruct tuple
  h, s, v = color_tuple

  # openCV hue range: [0, 180]
  low_h = max(0, h-hue_shift)
  up_h = min(180, h+hue_shift)

  # openCV saturation range: [0, 255]
  low_s = 5 * round((s/5)/5) #divide by 5 and then round to nearest 5
  up_s = min(255, 5 * round((s*5)/5)) # multiply by 5 and then round to nearest 5, capping at 255

  # openCV value range: [0, 255]
  if (h == 0 and s == 0): # black or white
    low_v = 5 * round((v/1.5)/5) #divide by 1.5 and then round to nearest 5
    if v > 0: up_v = min(255, 5 * round((v*1.5)/5)) # multiply by 1.5 and then round to nearest 5, capping at 255
    else: up_v = 20
  else:
    # value range: [0, 255]
    low_v = 5 * round((v/value_shift)/5) #divide by `value_shift` and then round to nearest 5
    up_v = min(255, 5 * round((v*value_shift)/5)) # multiply by `value_shift` and then round to nearest 5, capping at 255

  lower, upper = np.array([low_h, low_s, low_v]).astype(np.uint8), np.array([up_h, up_s, up_v]).astype(np.uint8)

  return lower, upper


def show_img(img, title=None, cmap=None, suptitle=None, targ=None, pred=None):
  # max fig dim: 65536
  incorrect_color = 'red'
  if targ is None: correct_color = 'white'
  else: correct_color = 'green'

  if type(img) == list or type(img) == tuple:
    img_format = getattr(img[0], '__module__', 'cv2').split('.')[0]

    if img_format == 'PIL': img_w, img_h = img[0].size
    else: img_h, img_w = img[0].shape[0], img[0].shape[1]

    img_ct = len(img)

    if not isinstance(title, Iterable): title = [title for i in range(img_ct)]
    if not isinstance(targ, Iterable): targ = [targ for i in range(img_ct)]
    if not isinstance(pred, Iterable): pred = [pred for i in range(img_ct)]

    if len(img) < 2:
      try:
        if suptitle is not None: plt.suptitle(suptitle)
        if title[0] is not None: plt.title(title[0], color=(correct_color if targ[0]==pred[0] else incorrect_color))
        plt.imshow(img[0], cmap=cmap)

        plt.tight_layout()

        plt.show()
      except Exception as e:
        print(f'show img exception: {e}')
        return
    else:
      row_ct = int(np.ceil(len(img)/10))
      col_ct = int(np.ceil(len(img)/row_ct))

      fig_w = int((img_w/100)*col_ct)
      fig_h = int((img_h/100)*row_ct)

      if fig_w > 30:
        fig_h = int(fig_h*(30/fig_w))
        fig_w = 30
      else: fig_w += 1
      
      # for title
      fig_h += 1
      figsize = (fig_w, fig_h)

      fig, axs = plt.subplots(row_ct, col_ct, subplot_kw={'xticks': [], 'yticks': []})

      renderer = fig.canvas.get_renderer()

      img_ct = len(img)

      if suptitle is not None: fig.suptitle(suptitle)

      title_widths = []

      for (ax, im, t, tar, pr) in zip(axs.flat, img, title, targ, pred):
        ax.imshow(im, cmap=cmap)
        if t is not None:
          ax_title = ax.set_title(t, color=(correct_color if tar==pr else incorrect_color), pad=20)
          bb = ax_title.get_window_extent(renderer=renderer)

          t_width = bb.width
          title_widths.append(t_width)


      if len(title_widths):
        max_title_width = max(title_widths)
        fig_width, fig_height = fig.get_size_inches()

        fig.set_figwidth(1.1*((max_title_width/fig.dpi)*col_ct))

      fig.set_tight_layout(True)
      fig.tight_layout() 
      plt.show()
  else:
    ax = plt.gca()

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    if suptitle is not None: plt.suptitle(suptitle)
    if title is not None: plt.title(title, color=(correct_color if targ==pred else incorrect_color))
    plt.imshow(img, cmap=cmap)
    plt.tight_layout()

    plt.show()


def cvt_img_format(image):
  # switch from PIL to cv2 format, or vice versa
  img_format = getattr(image, '__module__', 'cv2').split('.')[0]

  if img_format == 'PIL': return np.array(image)
  else: return Image.fromarray(image) #.convert('RGB')


def unique_colors(img):
  if len(img.shape) > 2:
    return np.unique(img.reshape(-1, img.shape[2]), axis=0)
  else: return np.unique(img)


def edge_color(img):
  edge_pixels = np.concatenate((img[0], img[-1], img[:,0], img[:,-1]))

  dom_edge_color = stats.mode(edge_pixels)[0][0]

  return dom_edge_color


def normalize_brightness(img, adjust_thresh=False):
  thresh = threshold_yen(img)

  if adjust_thresh:
    colors = unique_colors(img)
    closest_to_white = colors[-1]
    dist_from_white = np.amax(255 - closest_to_white)
    thresh += dist_from_white//2

  bright = rescale_intensity(img, (0, thresh), (0, 255))

  colors = unique_colors(bright)

  out = normalize(bright, 255, 0).astype(np.uint8)

  dom_edge_color = edge_color(out)
  if tuple(dom_edge_color) != (255, 255, 255):
    print(f'WARNING: edge color is {dom_edge_color}, which is not white. Trying again.')
    normalize_brightness(out, adjust_thresh=adjust_thresh)

  return out


def saturation_mask(img):
  img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  img_reduced = reduce_colors(img_hsv, col_ct=2)

  hsv_colors = unique_colors(img_reduced)

  hue, sat, val = hsv_colors.T

  most_saturated = hsv_colors[np.argmax(sat)] #index of highest saturation value

  sat_lower, sat_upper = hsv_thresh(tuple(most_saturated))

  sat_mask = cv2.inRange(img_hsv, sat_lower, sat_upper)

  return sat_mask


def list_species_names(parent_dir):
  parent_dir = os.path.normpath(parent_dir)
  child_paths = os.scandir(parent_dir)

  img_files = list_images(parent_dir, include_subdirs=False)

  if len(img_files): return [os.path.split(parent_dir)[-1]]
  else: return [f.name for f in child_paths if f.is_dir()]


def get_species(img_path, main_dir):
  main_dir = os.path.normpath(main_dir)

  path_components = img_path.split(os.path.sep)
  species = path_components[path_components.index(main_dir)+1]
  return species


def save_species_img(rgb_img, save_path, species, img_path, idx=None):
  bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

  img_file = os.path.split(img_path)[-1]

  if idx is None: save_as = f'{save_path}/{species}/{img_file}'
  else:
    ext_idx = img_file.rfind('.')
    filename, ext = img_file[:ext_idx], img_file[ext_idx:]
    save_as = f'{save_path}/{species}/{filename}_{idx}{ext}'
    
    while os.path.exists(save_as):
      idx += 1
      save_as = f'{save_path}/{species}/{filename}_{idx}{ext}'

  cv2.imwrite(save_as, bgr_img)

  return save_as


def clear_files(base_path, valid_exts=('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'), exclude_dirs=[]):
  if not os.path.exists(base_path): return

  for root_dir, dirs, files in os.walk(base_path, topdown=True):
    # exclude hidden folders and files #source: https://stackoverflow.com/a/13454267

    files = [f for f in files if f[f.rfind('.'):].lower().endswith(valid_exts)]
    dirs[:] = [d for d in dirs if d not in exclude_dirs]

    for file in files:
      os.remove(os.path.join(root_dir, file))


def random_file(parent_dir='data-test/Pollen Slides/cropped'):
  files = list_images(parent_dir)
  file_ct = len(files)
  rand_idx = random.randrange(file_ct)
  return files[rand_idx]


def resize_img(img, resize_width=1024):
  h, w = img.shape[:2]
  resize_scale = resize_width/w

  resized_img = cv2.resize(img, (resize_width, int(h*resize_scale)))

  return resized_img
  