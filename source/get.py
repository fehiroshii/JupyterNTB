import cv2 as cv
import numpy as np
import functions
import cfg


def filters_parameters(file, lenght, roi):

    treshold = 255

    video = cv.VideoCapture(file)
    video.set(cv.CAP_PROP_POS_FRAMES, lenght[0])

    first = True
    count = 0

    # Finds a frame with maximum luminous intensity (Diastole) and a
    # frame with minimum luminous intesity (Systole) in the range defined by user
    while (count <= ((lenght[1]-lenght[0])//3)):

        ret, img = video.read()
        img_crop = img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
        gray = cv.cvtColor(img_crop, cv.COLOR_BGR2GRAY)
        _, bw_img = cv.threshold(gray, 160, 255, cv.THRESH_BINARY)

        wht_px = np.count_nonzero(bw_img)

        # Execute on first interaction
        if first:
            frame_height = img.shape[0]
            frame_width = img.shape[1]

            max_wht_px = wht_px
            min_wht_px = wht_px
            max_frame = img
            min_frame = img
            first = False

        if wht_px > max_wht_px:
            max_frame = img
            max_wht_px = wht_px

        if wht_px < min_wht_px:
            min_frame = img
            min_wht_px = wht_px

        count = count + 1

    # Show images on systole and diastole for trehsold adjustment
    cv.imshow('Systole', min_frame)
    cv.imshow('Diastole', max_frame)
    
    # Nested Functions that are used as callback functions 

    # Adjust a vertical axis on frame that represents the revolution axis
    def ver_axis_adjustment(args):
        dias_frame = np.copy(max_frame)
        sys_frame = np.copy(min_frame)

        cv.line(dias_frame, (args, 0), (args, frame_height), (255, 0, 255), 1)
        cv.line(sys_frame, (args, 0), (args, frame_height), (255, 0, 255), 1)

        cv.imshow('Systole', sys_frame)
        cv.imshow('Diastole', dias_frame)

        return
    
    def treshold_adjustment(args):
        # Get current treshold and scale from trackbar
        scale = cv.getTrackbarPos('Scale', 'Comandos')
        treshold = cv.getTrackbarPos('Treshold', 'Comandos')

        # Make copy of unaltered frame
        dias_frame = np.copy(max_frame)
        sys_frame = np.copy(min_frame)

        # Crop image according to ROI
        dias_frame_crop = dias_frame[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
        sys_frame_crop = sys_frame[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]

        # Apply filters on both frames
        dias_filter = functions.filters(dias_frame_crop)
        sys_filter = functions.filters(sys_frame_crop)
        _, dias_binary_img = cv.threshold(dias_filter, treshold,
                                          255, cv.THRESH_BINARY)
        _, sys_binary_img = cv.threshold(sys_filter, treshold,
                                         255, cv.THRESH_BINARY)
        
        # Add a black border to both images 
        sys_binary_img = cv.copyMakeBorder(
            sys_binary_img,
            top=1,
            bottom=1,
            left=1,
            right=1,
            borderType=cv.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        dias_binary_img = cv.copyMakeBorder(
            dias_binary_img,
            top=1,
            bottom=1,
            left=1,
            right=1,
            borderType=cv.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        # Detect binary edges
        dias_binary_edges = cv.Canny(dias_binary_img, 50, 100, 5, L2gradient=True)
        sys_binary_edges = cv.Canny(sys_binary_img, 50, 100, 5, L2gradient=True)

        # Find contours on both frames
        dias_contours, _ = cv.findContours(dias_binary_edges,
                                           cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        sys_contours, _ = cv.findContours(sys_binary_edges,
                                          cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # Detect greatest contour of both frames and get its index
        max_area = -1
        for i, c in enumerate(sys_contours):
            if cv.arcLength(c, True) > max_area:
                max_area = cv.arcLength(c, True)
                max_i_sys = i

        max_area = -1
        for i, c in enumerate(dias_contours):
            if cv.arcLength(c, True) > max_area:
                max_area = cv.arcLength(c, True)
                max_i_dias = i

        # Apply scale to both contours (Systole and Diastole)
        scaled_dias = functions.scale_contour(dias_contours[max_i_dias],
                                              scale/100)
        scaled_sys = functions.scale_contour(sys_contours[max_i_sys],
                                             scale/100)    

        # Place the detected contour on the right place on original frame
        scaled_dias = scaled_dias + roi[0]
        scaled_sys = scaled_sys + roi[0]   
        
        # Draw contours on both frames
        cv.drawContours(dias_frame, scaled_dias, -1, (255, 0, 255), 1)
        cv.drawContours(sys_frame, scaled_sys, -1, (255, 0, 255), 1)

        # Obtain ellipse param as  (center x,center y), (height,width), ângle
        min_center, min_dim, min_angle = cv.fitEllipseDirect(scaled_sys)
        max_center, max_dim, max_angle = cv.fitEllipseDirect(scaled_dias)

        # Draw ellipses on both frames
        cv.ellipse(dias_frame, (max_center, max_dim, max_angle),
                   (255, 0, 255), 1)
        cv.ellipse(sys_frame, (min_center, min_dim, min_angle),
                   (255, 0, 255), 1)

        # Show both frames with ellipse adjusted
        cv.imshow('Systole', sys_frame)
        cv.imshow('Diastole', dias_frame)

        return
    
    def hor_axis_adjustment(args):
        dias_frame = np.copy(max_frame)
        sys_frame = np.copy(min_frame)

        # Get current threshold and scale from trackbar
        scale = cv.getTrackbarPos('Scale', 'Comandos')
        treshold = cv.getTrackbarPos('Treshold', 'Comandos')
        x_ax = cv.getTrackbarPos('Axis', 'Comandos')

        # Crop image according to ROI
        dias_frame_crop = dias_frame[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
        sys_frame_crop = sys_frame[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]

        # Apply filters on both frames
        dias_filter = functions.filters(dias_frame_crop)
        sys_filter = functions.filters(sys_frame_crop)
        _, dias_binary_img = cv.threshold(dias_filter, treshold,
                                          255, cv.THRESH_BINARY)
        _, sys_binary_img = cv.threshold(sys_filter, treshold,
                                         255, cv.THRESH_BINARY)

        # Add a black border to both images
        sys_binary_img = cv.copyMakeBorder(
            sys_binary_img,
            top=1,
            bottom=1,
            left=1,
            right=1,
            borderType=cv.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        dias_binary_img = cv.copyMakeBorder(
            dias_binary_img,
            top=1,
            bottom=1,
            left=1,
            right=1,
            borderType=cv.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        # Detect binary edges
        dias_binary_edges = cv.Canny(
            dias_binary_img, 50, 100, 5, L2gradient=True)
        sys_binary_edges = cv.Canny(
            sys_binary_img, 50, 100, 5, L2gradient=True)

        # Find contours on both frames
        dias_contours, _ = cv.findContours(dias_binary_edges,
                                           cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        sys_contours, _ = cv.findContours(sys_binary_edges,
                                          cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # Detect greatest contour of both frames and get its index
        max_area = -1
        for i, c in enumerate(sys_contours):
            if cv.arcLength(c, True) > max_area:
                max_area = cv.arcLength(c, True)
                max_i_sys = i

        max_area = -1
        for i, c in enumerate(dias_contours):
            if cv.arcLength(c, True) > max_area:
                max_area = cv.arcLength(c, True)
                max_i_dias = i

        # Apply scale to both contours (Systole and Diastole)
        scaled_dias = functions.scale_contour(dias_contours[max_i_dias],
                                              scale/100)
        scaled_sys = functions.scale_contour(sys_contours[max_i_sys],
                                             scale/100)
        
        # Place the detected contour on the right place on original frame
        scaled_dias = scaled_dias + roi[0]
        scaled_sys = scaled_sys + roi[0]

        # Draw contours on both frames
        cv.drawContours(dias_frame, scaled_dias, -1, (255, 0, 255), 1)
        cv.drawContours(sys_frame, scaled_sys, -1, (255, 0, 255), 1)

        dias_left_x = np.array([])
        dias_right_x = np.array([])

        for i in range(len(scaled_dias)):
            if abs(scaled_dias[i][0][1] - args) <= 5:
                if scaled_dias[i][0][0] <= x_ax:
                    dias_left_x = np.append(dias_left_x, scaled_dias[i][0][0])
                else:
                    dias_right_x = np.append(dias_right_x, scaled_dias[i][0][0])

        sys_left_x = np.array([])
        sys_right_x = np.array([])

        for i in range(len(scaled_sys)):
            if abs(scaled_sys[i][0][1] - args) <= 5:
                if scaled_sys[i][0][0] <= x_ax:
                    sys_left_x = np.append(sys_left_x, scaled_sys[i][0][0])
                else:
                    sys_right_x = np.append(sys_right_x, scaled_sys[i][0][0])

        if dias_left_x != [] and dias_right_x != [] and sys_left_x != [] and sys_right_x != []:
            cv.circle(dias_frame, (int(np.average(dias_left_x)), args),
                  3, (0, 0, 255), -1)
            cv.circle(dias_frame, (int(np.average(dias_right_x)), args),
                  3, (0, 0, 255), -1)

            cv.circle(sys_frame, (int(np.average(sys_left_x)), args),
                  3, (0, 0, 255), -1)
            cv.circle(sys_frame, (int(np.average(sys_right_x)), args),
                  3, (0, 0, 255), -1)

        cv.line(dias_frame, (0, args), (frame_width, args), (255, 0, 255), 1)
        cv.line(sys_frame, (0, args), (frame_width, args), (255, 0, 255), 1)

        cv.imshow('Systole', sys_frame)
        cv.imshow('Diastole', dias_frame)

        return

    # Create window with 3 trackbar
    cv.namedWindow('Comandos', cv.WINDOW_NORMAL)
    cv.resizeWindow('Comandos', 700, 100)
    cv.createTrackbar('Treshold', 'Comandos', 254, 254, treshold_adjustment)
    cv.createTrackbar('Axis', 'Comandos', 0, frame_width, ver_axis_adjustment)
    cv.createTrackbar('Diameter', 'Comandos', 0, frame_height, hor_axis_adjustment)
    cv.createTrackbar('Scale', 'Comandos', 100, 100, treshold_adjustment)

    # Waits for the user to press 'n' key
    key = cv.waitKey(0)
    while (key != cfg.confirm):
        key = cv.waitKey(0)
    
    # Get values and close windows
    treshold = cv.getTrackbarPos('Treshold','Comandos')
    axis_location = cv.getTrackbarPos('Axis','Comandos')
    scale = cv.getTrackbarPos('Scale','Comandos')/100
    hor_axis = cv.getTrackbarPos('Diameter','Comandos')

    cv.destroyAllWindows()   

    return treshold, axis_location, scale, hor_axis


# Function that calculates all timestamps in the format mm:ss
def timestamp(fps, tot_frame):
    stamp_array = np.array([])

    secs = 0
    mins = 0

    for i in range(tot_frame//fps):

        stamp_array = np.append(stamp_array, (("%02d:%02d") % (mins, secs)))

        secs = secs + 1

        if secs == 60:
            mins = mins + 1
            secs = 0
        
    return stamp_array


def average_center(file, treshold):

    # Faz a abertura do video
    video = cv.VideoCapture(file)
    
    # Inicializacao das variaveis
    n_frames = 0
    ret = True
    x_sum = 0
    y_sum = 0
    
    # Enquanto o vídeo estiver aberto, a cada frame do video obtemos a posicao central do retangulo
    # que envolve o contorno do coracao e no fim e calculado a media dos valores de centro
    while video.isOpened():

        # faz a leitura do frame 
        ret, frame = video.read()

        # testa se o frame foi aberto (ret == True)        
        if ret:

            max_area = -1   # Inicializa a area maxima para o frame atual

            # Aplica os filtros no frame
            filtered = functions.filters(frame)
            _ , binary_img = cv.threshold(filtered,treshold,250,cv.THRESH_BINARY)
            
            # Encontra os contornos da imagem binaria
            contours,_ = cv.findContours(binary_img, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS)
        
            # Obtem o centro do retangulo que engloba o maior contorno
            for i, c in enumerate(contours):

                if cv.contourArea(c) > max_area: 
                    centerx,centery,largura,altura = cv.boundingRect(c)
                    max_area = cv.contourArea(c)

            x_sum = x_sum + (2*centerx + largura)//2
            y_sum = y_sum + (2*centery + altura)//2

            n_frames = n_frames + 1 
        
        else:
            break
    
    video.release() #libera o video
    del frame
    del contours
    del video
    
    return x_sum//n_frames, y_sum//n_frames

####################################################
#        Revolution Solid related functions        #
####################################################

def vol_solid_revo(cont,axis):

    right_cont_x = np.array([])
    right_cont_y = np.array([])
    left_cont_x = np.array([])
    left_cont_y = np.array([])
    fx2 = np.array([])

    for v in range(len(cont)):
        x_cont = cont[v][0][0]
        y_cont = cont[v][0][1]

        if x_cont > axis:
            right_cont_x = np.append(right_cont_x,x_cont)
            right_cont_y = np.append(right_cont_y,y_cont)
        else:
            left_cont_x = np.append(left_cont_x,x_cont)
            left_cont_y = np.append(left_cont_y,y_cont)

    a = int(np.amin(right_cont_y) )
    b = int(np.amax(right_cont_y) )
    
    fx = get_fx(right_cont_x,right_cont_y,axis) 

    fx2 = fx**2

    v1 = np.sum(fx2)/2
    a1 = np.sum(fx)

    a = int(np.amin(left_cont_y) )
    b = int(np.amax(left_cont_y) )
    
    fx = get_fx(left_cont_x,left_cont_y,axis)

    fx2 = fx**2

    v2 = np.sum(fx2)/2
    a2 = np.sum(fx)

    v = v1 + v2
    a = a1 + a2

    return 3.5*3.5*3.5*np.pi*v, 3.5*3.5*a

def get_fx(x,y,axis):

    fx = np.array([])

    a = np.amin(y) 
    b = np.amax(y) 

    # Sort x and y
    p = y.argsort()
    y = y[p]
    x = x[p]

    values, un_index ,counts = np.unique(y,return_index=True,return_counts=True)
    
    counts = counts - 1
    
    for i in range(len(counts)):
        if counts[i] == 0:
            fx =  np.append(fx, x[un_index[i]])
        else:
            fx = np.append( fx, np.average( x [ un_index[i] : un_index[i] + counts[i] ]  ) )
    
    if np.average(fx) < axis:
        return axis - fx
    else:
        return fx - axis