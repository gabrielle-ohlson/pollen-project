import cv2
import numpy as np

from scipy import ndimage
import scipy.cluster.hierarchy as hcluster

from skimage.feature import peak_local_max, local_binary_pattern
from skimage.segmentation import watershed

import staintools

from helpers import reduce_colors, load_img, show_img, list_images, normalize_brightness, unique_colors, normalize




# source: https://pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/

class LocalBinaryPatterns:
  def __init__(self, numPoints, radius):
    # store the number of points and radius
    self.numPoints = numPoints
    self.radius = radius
  def compute(self, image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the Local Binary Pattern representation of the image, and then use the LBP representation to build the histogram of patterns
    lbp = local_binary_pattern(gray_image, self.numPoints,
      self.radius, method="uniform")
    return lbp
  def describe(self, lbp, eps=1e-7):    
    (hist, _) = np.histogram(lbp.ravel(),
      bins=np.arange(0, self.numPoints + 3),
      range=(0, self.numPoints + 2))
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    # return the histogram of Local Binary Patterns
    return hist


# initialize the local binary patterns descriptor along with the data and label lists
desc = LocalBinaryPatterns(24, 8)



def remove_bg(image, alpha_mask=True):
  # # Load image and perform kmeans
  h, w = image.shape[:2]
  
  original = image.copy()
  kmeans = reduce_colors(image, 2)

  # Convert to grayscale, Gaussian blur, adaptive threshold
  gray = cv2.cvtColor(kmeans, cv2.COLOR_RGB2GRAY).astype('uint8')

  colors = np.unique(gray)

  binary_img = np.zeros(original.shape[:2], dtype=np.uint8)
  binary_img[np.where(gray==colors[0])] = 255

  cnts = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]

  mask = cv2.fillPoly(binary_img, cnts, 255)

  if alpha_mask:
    result = cv2.cvtColor(original, cv2.COLOR_RGB2RGBA)
    result[:, :, 3] = mask
  else:
    # Bitwise-and for result
    result = cv2.bitwise_and(original, original, mask=mask)
    result[mask==0] = (255,255,255)

  return result


def generate_normalizer(target_img_path='target_imgs/ML102115 40X1C 498_1.JPG'):
  target = load_img(target_img_path)
  target_stand = staintools.LuminosityStandardizer.standardize(target)

  normalizer = staintools.StainNormalizer(method='macenko')
  normalizer.fit(target_stand)

  return normalizer


def stain_norm(img, normalizer):
  img_stand = staintools.LuminosityStandardizer.standardize(img)
  img_bright = normalize_brightness(img_stand, adjust_thresh=True)
  try:
    transformed = normalizer.transform(img_bright)
  except Exception as e:
    print(e)
    return None

  if len(unique_colors(transformed) > 1): return transformed
  else:
    print('invalid image, skipping')
    return None


def process_samples(img, normalizer=None, LBP=False, debug=False):
  if LBP: error_return = ([], [])
  else: error_return = []

  cropped_samples = []
  cropped_lbp_samples = []
  channels = img.shape[2]
  
  if channels == 4: # alpha mask
    output_img = img[:, :, 0:3].astype('uint8')
    bg_mask = img[:, :, 3]
  else:
    output_img = img.copy()
    bg_mask = np.zeros(img.shape, dtype='uint8')
    bg_mask[img == (255, 255, 255)] = 255

  if normalizer is not None:
    output_img = stain_norm(output_img, normalizer)
    if output_img is None: return error_return
  if LBP:
    lbp_img = desc.compute(output_img)
    lbp_img = cv2.cvtColor(normalize(lbp_img, 255, 0).astype('uint8'), cv2.COLOR_GRAY2RGB)
    if len(unique_colors(lbp_img)) < 2: return error_return
    if debug: show_img([img, lbp_img], ['original', 'LBP'])

  cnts = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]

  max_contour_area = cv2.contourArea(max(cnts, key=cv2.contourArea))

  # source: https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html
  distance = ndimage.distance_transform_edt(bg_mask)

  min_distance = min(int(max_contour_area/150), 50) #TODO: maybe change back to 60
  
  coords = peak_local_max(distance, footprint=np.ones((3, 3)), min_distance=min_distance, labels=bg_mask)

  seg_mask = np.zeros(distance.shape, dtype=bool)
  seg_mask[tuple(coords.T)] = True
  markers = ndimage.label(seg_mask)[0]
  labels = watershed(-distance, markers, mask=bg_mask)


  contours = []

  for label in np.unique(labels)[1:]: #start at one to exclude background
    mask = np.zeros(bg_mask.shape, dtype='uint8')
    mask[labels == label] = 255

    cnts = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)[0]

    c = max(cnts, key=cv2.contourArea)
    contours.append(c)
  
  cnts_arr = np.array(contours, dtype=object)
  cnts_area = np.array([cv2.contourArea(c) for c in contours])

  data = np.array([[c, c] for c in cnts_area])

  # clustering
  thresh = (11.0/100.0) * (np.amax(cnts_area) - np.amin(cnts_area))  #Threshold 11% of the total range of data

  clusters = hcluster.fclusterdata(data, thresh, criterion="distance")
  
  large_cnts = cnts_arr[np.where(clusters>1)]

  
  for c in large_cnts.tolist():
    x,y,w,h = cv2.boundingRect(c)

    cropped = output_img[y:y+h, x:x+w]

    if len(unique_colors(cv2.threshold(cropped, 100, 255, cv2.THRESH_BINARY)[1])) < 2: continue

    cropped_samples.append(cropped)
    
    if LBP:
      lbp_cropped = lbp_img[y:y+h, x:x+w]
      cropped_lbp_samples.append(lbp_cropped)

  if debug: show_img(cropped_samples)
  if LBP: return cropped_samples, cropped_lbp_samples
  return cropped_samples
