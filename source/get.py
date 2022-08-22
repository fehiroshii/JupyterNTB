import cv2 as cv
import numpy as np
import functions


def filters_parameters (file):
    import global_

    treshold = 255

    # Faz a leitura do primeiro frame do video
    video = cv.VideoCapture(file)       

    ret, global_.first_frame = video.read()
    gray = cv.cvtColor(global_.first_frame, cv.COLOR_BGR2GRAY)
    _, bw_img = cv.threshold(gray,160,255,cv.THRESH_BINARY)
    wht_px = np.count_nonzero(bw_img)

    global_.frame_height = global_.first_frame.shape[0]
    global_.frame_width = global_.first_frame.shape[1]

    count = 0
    global_.min_frame = global_.first_frame
    global_.max_frame = global_.first_frame

    max_wht_px = wht_px
    min_wht_px = wht_px

    while (count <= video.get(cv.CAP_PROP_FRAME_COUNT)//3):

      ret, img = video.read()
      gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      _, bw_img = cv.threshold(gray,160,255,cv.THRESH_BINARY)

      wht_px = np.count_nonzero(bw_img)

      if wht_px > max_wht_px:
        global_.max_frame = img
        max_wht_px = wht_px
        max_time = video.get(cv.CAP_PROP_POS_MSEC)/1000
      elif wht_px < min_wht_px:
        global_.min_frame = img
        min_wht_px = wht_px
        min_time = video.get(cv.CAP_PROP_POS_MSEC)/1000

      count = count + 1

    # Show images on systole and diastole for trehsold adjustment
    cv.imshow('Systole',global_.min_frame)
    cv.imshow('Diastole',global_.max_frame)

    # Create window with 3 trackbar
    cv.namedWindow('Comandos', cv.WINDOW_NORMAL)
    cv.resizeWindow('Comandos', 700, 100)
    cv.createTrackbar('Treshold','Comandos',254,254,functions.treshold_adjustment)
    cv.createTrackbar('Axis','Comandos',0,global_.frame_width,functions.ver_axis_adjustment)
    cv.createTrackbar('Diameter','Comandos',0,global_.frame_height,functions.hor_axis_adjustment)
    cv.createTrackbar('Scale','Comandos',100,100,functions.treshold_adjustment)

    # Waits for the user to press 'n' key
    key = cv.waitKey(0)
    while (key != ord('n')):
        key = cv.waitKey(0)
    
    # Get values and close windows
    treshold = cv.getTrackbarPos('Treshold','Comandos')
    axis_location = cv.getTrackbarPos('Axis','Comandos')
    scale = cv.getTrackbarPos('Scale','Comandos')/100
    hor_axis = cv.getTrackbarPos('Diameter','Comandos')

    cv.destroyAllWindows()   

    return treshold, axis_location, scale, hor_axis



def average_center(file,treshold):

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
#            Ellipse related functions             #
####################################################

# Returns the value of ellipsis major axis "a"
def get_a():
    import global_  
    return np.sqrt((global_.ellipse_vertex[0][0] - global_.ellipse_vertex[1][0])**2 + 
            (global_.ellipse_vertex[0][1] - global_.ellipse_vertex[1][1])**2)

    
# Returns the value of ellipsis minor axis "b" 
def get_b():
    import global_
    return np.sqrt((global_.ellipse_vertex[2][0] - global_.ellipse_vertex[3][0])**2 + 
            (global_.ellipse_vertex[2][1] - global_.ellipse_vertex[3][1])**2)


# Get ellipses main parameters ("a","b",Area and Volume)
def ellipse_paramaters(d_pixel):         

    # Get ellipsis major and minor axis
    a = get_a()
    b = get_b()
            
    # Calculate ellipses area and volume
    area = a*np.pi*b
    volume = int((1/6)*np.pi*a*b**2) 

    return int(a*d_pixel), int(b*d_pixel), int(area*d_pixel**2) ,int(volume*d_pixel**3)

def ellipse_size():
    import global_
    #global ellipse_center, ellipse_height, ellipse_width, ellipse_size #ellipse_vertex
    
    # Calcula o centro da elipse
    global_.ellipse_center = ((global_.ellipse_vertex[2][0] + global_.ellipse_vertex[3][0])//2,(global_.ellipse_vertex[0][1] + global_.ellipse_vertex[1][1])//2)

    global_.ellipse_height = int(functions.distance(global_.ellipse_vertex[0][0],global_.ellipse_vertex[0][1],
                            global_.ellipse_vertex[1][0],global_.ellipse_vertex[1][1]))
    global_.ellipse_width = int(functions.distance(global_.ellipse_vertex[2][0],global_.ellipse_vertex[2][1],
                            global_.ellipse_vertex[3][0],global_.ellipse_vertex[3][1]))
    global_.ellipse_size = (global_.ellipse_width,global_.ellipse_height)

    return


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

    v1 = np.sum(fx2)
    a1 = np.sum(fx)

    a = int(np.amin(left_cont_y) )
    b = int(np.amax(left_cont_y) )
    
    fx = get_fx(left_cont_x,left_cont_y,axis)

    fx2 = fx**2

    v2 = np.sum(fx2)
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