import cv2 as cv
import numpy as np

import get
import cfg

# Function that apply filters on a image
def filters(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # image to grayscale
    blur = cv.medianBlur(gray,9)    # Apply median blur with kernel = 9
    gauss = cv.GaussianBlur(blur,(9,9),sigmaX=0,sigmaY=0)   # Apply Gaussian blur with kernel = 9
        
    return gauss  

def rectangle_ellipse(parameters):
    import global_  # Get current value for global var

    # List contaning the 4 vertex of rectangle that fits the parameter
    box2 = cv.boxPoints(parameters)
    box2 = np.intp(box2)     
    
    # Inferior ellipse vertex  
    global_.ellipse_vertex[0][0] = (box2[0][0] + box2[3][0])//2
    global_.ellipse_vertex[0][1] = (box2[0][1] + box2[3][1])//2

    # Superior ellipse vertex
    global_.ellipse_vertex[1][0] = (box2[1][0] + box2[2][0])//2
    global_.ellipse_vertex[1][1] = (box2[1][1] + box2[2][1])//2

    # Right ellipse vertex
    global_.ellipse_vertex[2][0] = (box2[2][0] + box2[3][0])//2
    global_.ellipse_vertex[2][1] = (box2[2][1] + box2[3][1])//2
            
    # Left ellipse vertex
    global_.ellipse_vertex[3][0] = (box2[0][0] + box2[1][0])//2
    global_.ellipse_vertex[3][1] = (box2[0][1] + box2[1][1])//2        

    global_.ellipse_vertex = global_.ellipse_vertex.astype(int)
    
    return box2

def treshold_adjustment (args):
    import global_

    try:
        # Get current treshold and scale from trackbar
        scale = cv.getTrackbarPos('Scale','Comandos')
        treshold = cv.getTrackbarPos('Treshold','Comandos')
    except:
        scale = 100
        treshold = 254

    # Make copy of unaltered frame
    dias_frame = np.copy(global_.max_frame)
    sys_frame = np.copy(global_.min_frame)

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
    #cv.ellipse(dias_frame, max_ellipse, (255,0,255), 1)
    #cv.ellipse(sys_frame, min_ellipse, (255,0,255), 1)

    # Draw ellipses on both frames
    cv.ellipse(dias_frame, (max_center,max_dim,max_angle), (255,0,255), 1)
    cv.ellipse(sys_frame, (min_center,min_dim,min_angle), (255,0,255), 1)

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


def scale_contour(cnt, scale):
    M = cv.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx,0]
    #cnt_scaled = cnt_norm * scale
    
    cnt_scaled = np.copy(cnt_norm)

    for i in range(len(cnt_norm)):
        cnt_scaled[i][0][0] = cnt_norm[i][0][0]*scale
    
    cnt_scaled = cnt_scaled + [cx,0]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled

def ver_axis_adjustment(args):
    import global_
    dias_frame = np.copy(global_.max_frame)
    sys_frame = np.copy(global_.min_frame)

    cv.line(dias_frame,(args,0),(args,global_.frame_height),(255,0,255),1)
    cv.line(sys_frame,(args,0),(args,global_.frame_height),(255,0,255),1)

    cv.imshow('Systole',sys_frame)
    cv.imshow('Diastole',dias_frame)

    return

def hor_axis_adjustment(args):
    import global_
    dias_frame = np.copy(global_.max_frame)
    sys_frame = np.copy(global_.min_frame)

    cv.line(dias_frame,(0,args),(global_.frame_width,args),(255,0,255),1)
    cv.line(sys_frame,(0,args),(global_.frame_width,args),(255,0,255),1)

    cv.imshow('Systole',sys_frame)
    cv.imshow('Diastole',dias_frame)

    return


####################################################
#            Ellipse related functions             #
####################################################

def fit_ellipse(source_img,method,join,stop,scale):
    import global_

    # Fit the contour that fits the heart shape
    cnts,_ = cv.findContours(source_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Matrizes que guardam a area, pontos do retangulo e elipse de cada contorno
    #minEllipse = [None]*len(cnts)
    center_contours = []
    
    best_i = 0
    
    max_area = -1
    
    # Loop that gets the height, width and height of each contour 
    # Salva os parametros do contorno de maior area (parte central do coração)
    for i, c in enumerate(cnts):

        # Finds the coordinates of the rectangle that circumscribes the contour "c"
        top_left_x,top_left_y,largura,altura = cv.boundingRect(c)

        # Get rectangle center
        centerx = (2*top_left_x + largura)/2
        centery = (2*top_left_y + altura)/2
        center_contours.append([centerx,centery])

        # Encontra o contorno de maior area 
        if cv.contourArea(c) > max_area: 
            max_area = cv.contourArea(c)
            estimated_centerx = centerx
            estimated_centery = centery
            best_i = i
    
    # Se distância maior que 10, então a divisão aparente será removida
    if distance(estimated_centerx,estimated_centery,global_.avg_x_center,global_.avg_y_center) > 20 or join :
        #contours = remove_division(contours, best_i)
        if method == True:
            method = False
        else:
            stop = True
    
    max_area = -1

    for i, c in enumerate(cnts):

        if cv.contourArea(c) > max_area: 
            max_area = cv.contourArea(c)
            main_cnt = i

    
    heart_cnt = cnts[main_cnt]
    scaled_heart_cnt = scale_contour(heart_cnt,scale)

    minEllipse = cv.fitEllipseDirect(scaled_heart_cnt)

    top_left_x,top_left_y,width,height = cv.boundingRect(scaled_heart_cnt)

    centerx = (2*top_left_x + largura)/2
    centery = (2*top_left_y + altura)/2

    center_contours.append([float(centerx),float(centery)])

    area = cv.contourArea(scaled_heart_cnt)         

    drawing_img = np.zeros((source_img.shape[0], source_img.shape[1], 3), dtype=np.uint8)
    
    for i, c in enumerate(cnts):
            
        if cv.contourArea(c) == max_area: 
           
            # Obtem parametros da elipse ((centrox,centroy), (altura,largura), ângulo)
            ellipse_param =  minEllipse

            import global_  # Update global variables
            global_.ellipse_vertex = np.zeros((4,2))
            global_.ellipse_angle = ellipse_param[2]
        
            box2 = rectangle_ellipse(minEllipse)
            get.ellipse_size()

            # Draw contours and ellipse
            cv.drawContours(drawing_img, scaled_heart_cnt, -1,cfg.main_cnt_color)
            cv.ellipse(drawing_img, minEllipse, cfg.ellipse_color, 1) 

            # Draw ellipse vertex        
            cv.circle(drawing_img,(global_.ellipse_vertex[0][0],global_.ellipse_vertex[0][1]),0,cfg.ell_vrtx_color,8)
            cv.circle(drawing_img,(global_.ellipse_vertex[1][0],global_.ellipse_vertex[1][1]),0,cfg.ell_vrtx_color,8)
            cv.circle(drawing_img,(global_.ellipse_vertex[2][0],global_.ellipse_vertex[2][1]),0,cfg.ell_vrtx_color,8)
            cv.circle(drawing_img,(global_.ellipse_vertex[3][0],global_.ellipse_vertex[3][1]),0,cfg.ell_vrtx_color,8)     
        else:
            # Draw other contours
            cv.drawContours(drawing_img, cnts, i, cfg.sec_cnt_color)

    cv.rectangle(drawing_img,(top_left_x,top_left_y),
                (top_left_x+width,top_left_y+height),cfg.rec_color,2)

    return scaled_heart_cnt, main_cnt, ellipse_param, box2 ,drawing_img,area,method,stop


def distance(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)