import cv2 as cv
import numpy as np

# Function that apply filters on a image
def filters(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # image to grayscale
    blur = cv.medianBlur(gray,9)    # Apply median blur with kernel = 9
    gauss = cv.GaussianBlur(blur,(9,9),sigmaX=0,sigmaY=0)   # Apply Gaussian blur with kernel = 9
        
    return gauss  

# Adjust the treshold of a image
def treshold_adjustment (args):

    try:
        # Get current treshold and scale from trackbar
        scale = cv.getTrackbarPos('Scale','Comandos')
        treshold = cv.getTrackbarPos('Treshold','Comandos')
    except:
        scale = 100
        treshold = 254

    # Make copy of unaltered frame
    dias_frame = np.copy(max_frame)
    sys_frame = np.copy(min_frame)

    # Apply filters on both frames
    dias_filter  = filters(dias_frame)
    sys_filter  = filters(sys_frame)
    _ , dias_binary_img = cv.threshold(dias_filter,treshold,255,cv.THRESH_BINARY)
    _ , sys_binary_img = cv.threshold(sys_filter,treshold,255,cv.THRESH_BINARY)

    # Detect binary edges
    dias_binary_edges = cv.Canny(dias_binary_img, 50, 100, 5, L2gradient= True)
    sys_binary_edges = cv.Canny(sys_binary_img, 50, 100, 5, L2gradient= True)

    # Find contours on both frames
    dias_contours,_ = cv.findContours(dias_binary_edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    sys_contours,_ = cv.findContours(sys_binary_edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Detect greatest contour of both frames and get its index
    max_area = -1
    for i, c in enumerate(sys_contours):
        if cv.arcLength(c,True) > max_area: 
            max_area = cv.arcLength(c,True)
            max_i_sys = i

    max_area = -1
    for i, c in enumerate(dias_contours):
        if cv.arcLength(c,True) > max_area: 
            max_area = cv.arcLength(c,True)
            max_i_dias = i

    # Apply scale to both contours (Systole and Diastole)
    scaled_dias = scale_contour(dias_contours[max_i_dias],scale/100)
    scaled_sys = scale_contour(sys_contours[max_i_sys],scale/100)
    
    # Draw contours on both frames
    cv.drawContours(dias_frame,scaled_dias,-1,(255,0,255),1)
    cv.drawContours(sys_frame,scaled_sys,-1,(255,0,255),1)

    # Obtain ellipse param as  (center x,center y), (height,width), ângle
    min_center,min_dim,min_angle = cv.fitEllipseDirect(scaled_sys) 
    max_center,max_dim,max_angle = cv.fitEllipseDirect(scaled_dias)
       
    avg_height = (max_dim[1] + min_dim[1])/2

    max_ellipse = (max_center,(max_dim[0],avg_height),0)
    min_ellipse = (min_center,(min_dim[0],avg_height),0) 

    # Draw ellipses on both frames
    cv.ellipse(dias_frame, max_ellipse, (255,0,255), 1)
    cv.ellipse(sys_frame, min_ellipse, (255,0,255), 1)

    global ellipse_vertex
    ellipse_vertex = np.zeros((4,2))

    dias_box = rectangle_ellipse(max_ellipse)
    dias_param = ellipse_vertex

    ellipse_vertex = np.zeros((4,2))

    sys_box = rectangle_ellipse(min_ellipse)
    sys_param = ellipse_vertex
       
    del ellipse_vertex

    # Show both frames with ellipse adjusted
    cv.imshow('Systole',sys_frame)
    cv.imshow('Diastole',dias_frame)
    
    return

def get_filters_parameters (file):

    treshold = 255
    video = cv.VideoCapture(file)   # Opens video       
    global max_frame, min_frame, first_frame, frame_height, frame_width

    ret, first_frame = video.read()
    gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    _, bw_img = cv.threshold(gray,160,255,cv.THRESH_BINARY)
    wht_px = np.count_nonzero(bw_img)

    frame_height = first_frame.shape[0]
    frame_width = first_frame.shape[1]

    count = 0
    min_frame = first_frame
    max_frame = first_frame

    max_wht_px = wht_px
    min_wht_px = wht_px

    while (count <= video.get(cv.CAP_PROP_FRAME_COUNT)//3):

      ret, img = video.read()
      gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      _, bw_img = cv.threshold(gray,160,255,cv.THRESH_BINARY)

      wht_px = np.count_nonzero(bw_img)

      if wht_px > max_wht_px:
        max_frame = img
        max_wht_px = wht_px
        max_time = video.get(cv.CAP_PROP_POS_MSEC)/1000
      elif wht_px < min_wht_px:
        min_frame = img
        min_wht_px = wht_px
        min_time = video.get(cv.CAP_PROP_POS_MSEC)/1000

      count = count + 1

    # Show images on systole and diastole for trehsold adjustment
    cv.imshow('Systole',min_frame)
    cv.imshow('Diastole',max_frame)

    # Create window with 3 trackbar
    cv.namedWindow('Comandos', cv.WINDOW_NORMAL)
    cv.resizeWindow('Comandos', 700, 100)
    cv.createTrackbar('Treshold','Comandos',254,254,treshold_adjustment)
    cv.createTrackbar('Axis','Comandos',0,frame_width,axis_adjustment)
    cv.createTrackbar('Scale','Comandos',100,100,treshold_adjustment)

    # Waits for the user to press 'n' key
    key = cv.waitKey(0)
    while (key != ord('n')):
        key = cv.waitKey(0)
    
    # Get values and close windows
    treshold = cv.getTrackbarPos('Treshold','Comandos')
    axis_location = cv.getTrackbarPos('Axis','Comandos')
    scale = cv.getTrackbarPos('Scale','Comandos')/100

    cv.destroyAllWindows()   

    return treshold, axis_location, scale
