import sys
import cv2
import numpy as np
from scipy import signal
    
def gauss_kernel(size: int, sizey: int=None):
    size = int(size)
    sizey = int(sizey) if sizey is not None else size
    y, x = np.mgrid[-size:size+1, -sizey:sizey+1]

    g = np.exp(-(x**2/(2*(0.33*size)**2)+y**2/(2*(0.33*sizey)**2)))
    return g

def gauss_derivative_kernels(size: int, sizey: int=None):
    size = int(size)
    sizey = int(sizey) if sizey is not None else size
    y, x = np.mgrid[-size:size+1, -sizey:sizey+1]

    gx = - x * np.exp(-(x**2/(2*(0.33*size)**2)+y**2/(2*(0.33*sizey)**2)))
    gy = - y * np.exp(-(x**2/(2*(0.33*size)**2)+y**2/(2*(0.33*sizey)**2)))

    return gx,gy

def gauss_derivatives(im: np.array, size: int, sizey: int=None):
    gx,gy = gauss_derivative_kernels(size, sizey=sizey)

    imx = signal.convolve(im, gx, mode='same')
    imy = signal.convolve(im, gy, mode='same')

    return imx,imy

def compute_harris_response(image):  #, k=0.05):
    DERIVATIVE_KERNEL_RADIUS = 3
    OPENING_SIZE = 3
    
    #derivatives
    imx,imy = gauss_derivatives(image, DERIVATIVE_KERNEL_RADIUS)

    #kernel for weighted sum
    gauss = gauss_kernel(OPENING_SIZE) # opening param

    #compute components of the structure tensor
    Wxx = signal.convolve(imx*imx,gauss, mode='same')
    Wxy = signal.convolve(imx*imy,gauss, mode='same')
    Wyy = signal.convolve(imy*imy,gauss, mode='same')

    #determinant and trace
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy
    return Wdet / (Wtr + 1)  # 1 seems to be a reasonable value for epsilon

def detect_harris_points(image_gray: np.array, max_keypoints: int=30, 
                         min_distance: int=25, threshold: float=0.1):
    # 1. Compute Harris corner response
    harris_resp = compute_harris_response(image_gray)
    
    # 2. Filtering
    # 2.0 Mask init: all our filtering is performed using a mask
    detect_mask = np.ones(harris_resp.shape, dtype=bool)
    # 2.2 Response threshold
    detect_mask &= harris_resp > harris_resp.min()+threshold*(harris_resp.max()-harris_resp.min())
    # 2.3 Non-maximal suppression
    dil = cv2.dilate(harris_resp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_distance, min_distance)))
    detect_mask &= np.isclose(dil, harris_resp)  # keep only local maximas
               
    # 3. Select, sort and filter candidates
    # get coordinates of candidates
    candidates_coords = np.transpose(detect_mask.nonzero())
    # ...and their values
    candidate_values = harris_resp[detect_mask]
    #sort candidates
    sorted_indices = np.argsort(candidate_values)
    # keep only the bests
    best_corners_coordinates = candidates_coords[sorted_indices][:max_keypoints]

    return best_corners_coordinates


def py_harris_corner_detector(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return detect_harris_points(image_gray)

def main(args):
    if len(args) < 3 or len(args) > 4:
        print("wrong arguments, expected image and txt with CPU coordinates and optionaly GPU coordinates")
        return

    img = cv2.imread(args[1])

    points_python = py_harris_corner_detector(img)
    for pts in points_python:
        img = cv2.drawMarker(img , (int(pts[1]), int(pts[0])), color=(0, 255, 0), markerSize=5, markerType=cv2.MARKER_CROSS, thickness=1)

    # CPU points
    with open(args[2]) as fp:
        for line in fp:

            point_cpu = line[:-1].split(' ')
            point_cpu = (int(point_cpu[0]), int(point_cpu[1]))
            img = cv2.circle(img , point_cpu, 2, (255, 200, 0), 1)
    
    # GPU points
    if len(args) == 4: 
        with open(args[3]) as fp:
            for line in fp:
                point_gpu = line[:-1].split(' ')
                point_gpu = (int(point_gpu[0]), int(point_gpu[1]))
                img = cv2.circle(img , point_gpu, 2, (255, 192, 203), -1)


    cv2.imwrite("corners.png", img)

main(sys.argv)
