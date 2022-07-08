import cv2
import imutils
from skimage.filters import threshold_local
from scipy.spatial import distance as dist
import numpy as np
import random as rng
from math import sqrt

rng.seed(12345)

# read and show the input image
img_path = 'document.jpeg'
img = cv2.imread(img_path)

ratio = img.shape[0] / 500.0 # find the ratio
input_img = imutils.resize(img, height = 500) # resize the image to 500 


def get_line_coefficients(p1: tuple, p2: tuple):
  x1, y1 = p1
  x2, y2 = p2

  a = y1 - y2
  b = x2 - x1
  c = x1*y2 - x2*y1

  return (a,b,c)

  
def distance_between_points(p1: tuple, p2: tuple):
  x1, y1 = p1
  x2, y2, = p2
  distance = sqrt((x2 - x1)**2 + (y2 - y1)**2)
  return distance


def are_similar_corners(c1: tuple, c2: tuple): 
  return True if distance_between_points(c1, c2) < 3 else False


def remove_similar_corners(all_corners):
  corners = []
  for i in range(len(all_corners)): 
    similar_corner = False
    for j in range(len(all_corners)): 
      if i == j:
        break
      if are_similar_corners(all_corners[i], all_corners[j]): 
        similar_corner = True
        break
    if not similar_corner:
      corners.append(all_corners[i])
      
  return corners

def get_intersection_points(coeff_1: tuple, coeff_2: tuple):
  a1, b1, c1 = coeff_1
  a2, b2, c2 = coeff_2

  x = 0 
  y = 0

  det = a1 * b2 - a2 * b1

  x_num = b1 * c2 - b2 * c1

  y_num = c1 * a2 - c2 * a1

  if det > -0.5 and det < 0.5:
    return None
  

  if det != 0:
    x = x_num / det
    y = y_num / det
    return (x, y)
  
  return None


def get_all_possible_corners(coefficients, row_size, col_size): 
  all_corners = []

  for i in range(len(coefficients)):
    for j in range(i, len(coefficients)):
      if(i != j): 
      
        int_point = get_intersection_points(coefficients[i], coefficients[j])
        if int_point != None: 
          x, y = int_point
          if x > 0 and y > 0 and x < col_size and y < row_size:
            all_corners.append(int_point)
            
  return all_corners

def sort_contours(elem):
    return cv2.arcLength(elem, closed=True)
    
def get_corners(grayscale: cv2.Mat, output: cv2.Mat):
  convex_hull_mask = np.zeros((grayscale.shape[0], grayscale.shape[1], 3), dtype=np.uint8)


  convex_hull_mask_grayscale = cv2.cvtColor(convex_hull_mask, cv2.COLOR_BGR2GRAY)

  contours, _ = cv2.findContours(grayscale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

  contours = sorted(contours, key=sort_contours, reverse=True)[:1]

  hull_list = []
  hull = cv2.convexHull(contours[-1], True)
  hull_list.append(hull)

  cv2.drawContours(convex_hull_mask_grayscale, hull_list, -1, (255,0,0), 2, 8)

  cv2.imshow('Convex Hull Mask', convex_hull_mask_grayscale)
  cv2.waitKey(10)


  lines = cv2.HoughLinesP(image = convex_hull_mask_grayscale, rho = 2, theta = np.pi / 200, minLineLength=200, maxLineGap=0, threshold=40)

  if lines is not None:
    for line in lines:
      l = line[0]
      cv2.line(output, (l[0], l[1]), (l[2], l[3]), (0,255,0), 2, cv2.LINE_AA )
  
  if len(lines) >= 4:
    coefficients = []
    for line in lines:
      l = line[0]
      coefficients.append(get_line_coefficients((l[0], l[1]), (l[2], l[3])))
  

    rows, cols = grayscale.shape

    all_corners = get_all_possible_corners(coefficients, rows, cols)

    corners = remove_similar_corners(all_corners)
    
    return corners



def order_points(points: list):
	rect = np.zeros((4, 2), dtype = "float32")
	pts = np.array(points)
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect


def four_point_transform(image, pts):
	rect = order_points(pts)

	WIDTH = 595

	HEIGHT = 842

	dst = np.array([
		[0, 0],
		[WIDTH - 1, 0],
		[WIDTH - 1, HEIGHT - 1],
		[0, HEIGHT - 1]], dtype = "float32")

	M = cv2.getPerspectiveTransform(rect, dst)

	warped = cv2.warpPerspective(image, M, (WIDTH, HEIGHT))
	return warped


input_grey = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
input_grey = cv2.GaussianBlur(input_grey, (3,3), 0)
ret, input_grey = cv2.threshold(input_grey, 130, 200, cv2.THRESH_BINARY)

img_copy = input_img.copy()

edges = cv2.Canny(input_grey, 83, 300)

cv2.imshow('Canny', edges)
cv2.waitKey(10)

corners = get_corners(edges, input_img)

if len(corners) != 4:
  print("corner length not equal to 4")
  
    
print("FINAL CORNERS: ", corners)


for x, y in corners:
    cv2.circle(input_img, (int(x), int(y)),3, (0, 0, 255), 4)

cv2.imshow('Corners', input_img)
cv2.waitKey(10)

warped_img = four_point_transform(img_copy, corners)

cv2.imshow("warped img", warped_img)
cv2.waitKey(10)

# give a black and white feel to the image
# T = threshold_local(warped_img, 11, offset = 10, method = "gaussian")
# warped = (warped_img > T).astype("uint8") * 255
# cv2.imshow("Scanned", imutils.resize(warped, height = 842))
# cv2.waitKey(0)
# cv2.destroyAllWindows()



