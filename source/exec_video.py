import ipywidgets as widgets
import ipywidgets as widgets
from IPython.display import display, clear_output
from ipywidgets import Layout, Label
import numpy as np


import get

import cv2 as cv



from matplotlib import pyplot as plt
import get
import functions
import cfg

d_pixel = 3.5 # Distancia entre pixel = 3.5 um


def get_paramaters2(d_px,c,x_ax,y_ax):

    left_x = np.array([])
    right_x = np.array([])

    interval = 5

    for i in range (len(c)):
        if abs(c[i][0][1] - y_ax) <= 5:
            if c[i][0][0] <= x_ax:
                left_x = np.append(left_x,x_ax - c[i][0][0])
            else:
                right_x = np.append(right_x,c[i][0][0] - x_ax)

    topmost = tuple(c[c[:,:,1].argmin()][0])
    bottommost = tuple(c[c[:,:,1].argmax()][0])

    b = np.average(left_x) + np.average(right_x)
    a = bottommost[1] - topmost[1]
   
    volume = int((1/6)*np.pi*a*b**2) 

    return int(a*d_px), int(b*d_px) ,int(volume*d_px**3)


def video_progression(args):

    global skip
    
    if key != ord('n') or key != ord('n'):

        next_frame = video.get(cv.CAP_PROP_POS_FRAMES)
        current_frame = next_frame - 1
        previous_frame = current_frame - 1

        if args < current_frame:
            if previous_frame >= 0:
                video.set(cv.CAP_PROP_POS_FRAMES, previous_frame)
            else:
                previous_frame = 0
    
        skip = True

    return



def modify_elipse(action, x, y, flags, parameters):
    
    frame = np.copy(parameters[0])
    #rec = parameters[1]
    cache = np.copy(parameters[2])
    d_pixel = parameters[3]

    import global_

    global edit
    # Declara o vetor com os pontos da elipse como global
    global ellipse_vertex, ellipse_angle, rec

    global ellipse_center, ellipse_height, ellipse_width, ellipse_size

    
    if action == cv.EVENT_LBUTTONDOWN:

#cv.ellipse(drawing,(ellipse_center,ellipse_size,ellipse_angle), (90,90,255), 1)

        cv.ellipse(cache,global_.ellipse_param, (0,0,255), 1)
        
        if (((x> global_.ellipse_vertex[3][0]) and (x<global_.ellipse_vertex[2][0]) and (y>global_.ellipse_vertex[1][1]) and (y<global_.ellipse_vertex[0][1])
            and (global_.ellipse_angle <=90)) or ((x<global_.ellipse_vertex[3][0]) and (x>global_.ellipse_vertex[2][0]) and (y<global_.ellipse_vertex[1][1]) 
            and (y>global_.ellipse_vertex[0][1]) and (global_.ellipse_angle >90))) :

            cv.circle(cache,(global_.ellipse_vertex[0][0],global_.ellipse_vertex[0][1]),0,(0,0,255),8) # Ponto de baixo da elipse
            cv.circle(cache,(global_.ellipse_vertex[1][0],global_.ellipse_vertex[1][1]),0,(0,0,255),8) # Ponto de cima da elipse
            cv.circle(cache,(global_.ellipse_vertex[2][0],global_.ellipse_vertex[2][1]),0,(0,0,255),8) # Ponto lateral direito da elipse
            cv.circle(cache,(global_.ellipse_vertex[3][0],global_.ellipse_vertex[3][1]),0,(0,0,255),8) # Ponto lateral esquerdo da elipse
            cv.circle(cache,(global_.ellipse_center[0],global_.ellipse_center[1]),0,(0,0,255),8) # Ponto lateral esquerdo da elipse

            cv.circle(cache,(rec[0][0],rec[0][1]),0,(0,0,255),8) # Canto inferior esquerdo do retângulo
            cv.circle(cache,(rec[1][0],rec[1][1]),0,(0,0,255),8) # Canto superior esquerdo do retângulo
            cv.circle(cache,(rec[2][0],rec[2][1]),0,(0,0,255),8) # Canto superior direito do retângulo
            cv.circle(cache,(rec[3][0],rec[3][1]),0,(0,0,255),8) # Canto inferior direito do retângulo
            # Desenha o retângulo que envolve a elipse
            box = np.array([[rec[0][0],rec[0][1]],[rec[1][0],rec[1][1]],[rec[2][0],rec[2][1]],[rec[3][0],rec[3][1]]])
            cv.drawContours(cache,[box],0,(0,0,255),1)
            cv.imshow('Original',cache)
            edit = True
        else:
            cv.imshow('Original',cache)
            edit = False


    elif (action == cv.EVENT_MOUSEMOVE) and (flags == cv.EVENT_FLAG_LBUTTON):

        #cv.circle(cache,(vertex_ellipse[0][0],vertex_ellipse[0][1]),0,(0,0,255),8) # Ponto de baixo da elipse
        #cv.circle(cache,(vertex_ellipse[1][0],vertex_ellipse[1][1]),0,(0,255,255),8) # Ponto de cima da elipse
        #cv.circle(cache,(vertex_ellipse[2][0],vertex_ellipse[2][1]),0,(0,0,255),8) # Ponto lateral direito da elipse
        #cv.circle(cache,(vertex_ellipse[3][0],vertex_ellipse[3][1]),0,(0,0,255),8) # Ponto lateral esquerdo da elipse

        #cv.circle(cache,(rec[0][0],rec[0][1]),0,(0,0,255),8) # Canto inferior esquerdo do retângulo
        #cv.circle(cache,(rec[1][0],rec[1][1]),0,(0,0,255),8) # Canto superior esquerdo do retângulo
        #cv.circle(cache,(rec[2][0],rec[2][1]),0,(0,0,255),8) # Canto superior direito do retângulo
        #cv.circle(cache,(rec[3][0],rec[3][1]),0,(0,0,255),8) # Canto inferior direito do retângulo

        
        #cv.rectangle(cache,(rec[0][0],rec[0][1]),(rec[2][0],rec[2][1]),(0,255,0),1)

        # Redimensionamento da parte de baixo da elipse
        if functions.distance(global_.ellipse_vertex[0][0],global_.ellipse_vertex[0][1],x,y) < 10:
            dif = global_.ellipse_vertex[0][1] - y
            global_.ellipse_vertex[0][1] =  y
            rec[0][1] -= dif
            rec[3][1] -= dif            
        
        # Redimensionamento da parte de cima da elipse
        elif functions.distance(global_.ellipse_vertex[1][0],global_.ellipse_vertex[1][1],x,y) < 10:
            dif = global_.ellipse_vertex[1][1] - y
            global_.ellipse_vertex[1][1] =  y
            rec[1][1] -= dif
            rec[2][1] -= dif

        # Redimensionamento da parte lateral direita da elipse
        elif functions.distance(global_.ellipse_vertex[2][0],global_.ellipse_vertex[2][1],x,y) < 10:
            dif = global_.ellipse_vertex[2][0] - x
            global_.ellipse_vertex[2][0] =  x
            rec[2][0] -= dif
            rec[3][0] -= dif
        
        # Redimensionamento da parte lateral esquerdo da elipse
        elif functions.distance(global_.ellipse_vertex[3][0],global_.ellipse_vertex[3][1],x,y) < 10:
            dif = global_.ellipse_vertex[3][0] - x
            global_.ellipse_vertex[3][0] =  x
            rec[0][0] -= dif
            rec[1][0] -= dif


        # Move a elipse
        elif functions.distance(global_.ellipse_center[0],global_.ellipse_center[1],x,y) < 10:
            difx = global_.ellipse_center[0] - x
            dify = global_.ellipse_center[1] - y
            #ellipse_vertex[3][0] =  - x
            rec[0][0] -= difx
            rec[1][0] -= difx
            rec[2][0] -= difx
            rec[3][0] -= difx


            rec[1][1] -= dify
            rec[2][1] -= dify
            rec[0][1] -= dify
            rec[3][1] -= dify            

        #get_ellipse_param()
        #rectangle_ellipse((ellipse_center,ellipse_size,ellipse_angle))

        # Calcula a posição do ponto lateral direito e esquerdo da elipse
        global_.ellipse_vertex[2][1] = (rec[2][1] + rec[3][1])//2
        global_.ellipse_vertex[3][1] = (rec[0][1] + rec[1][1])//2
        
        # Calcula a posição do ponto superior e inferior da elipse
        global_.ellipse_vertex[1][0] = (rec[1][0] + rec[2][0])//2
        global_.ellipse_vertex[0][0] = (rec[0][0] + rec[3][0])//2

        global_.ellipse_vertex[2][0] = (rec[2][0] + rec[3][0])//2
        global_.ellipse_vertex[3][0] = (rec[0][0] + rec[1][0])//2
        global_.ellipse_vertex[1][1] = (rec[1][1] + rec[2][1])//2
        global_.ellipse_vertex[0][1] = (rec[0][1] + rec[3][1])//2

        get.ellipse_paramaters(d_pixel)

        # Desenha o retângulo que envolve a elipse
        #box = np.array([[rec[0][0],rec[0][1]],[rec[1][0],rec[1][1]],[rec[2][0],rec[2][1]],[rec[3][0],rec[3][1]]])
        #cv.drawContours(cache,[box],0,(0,0,255),1)

        # Desenha a elipse
        cv.ellipse(cache,global_.ellipse_param, (90,90,255), 1)

        #cv.ellipse(cache, box, color = (0, 0, 255), thickness = 1)

        cv.imshow('Original',cache)

        edit = True

        #height,width,volume = get_paramaters(d_pixel)

        #partial_info  = np.copy(info_frame)
        #cv.putText(partial_info,"Volume: " + str(int(volume)/1e+9) + " ul", (30,70),font ,1/2,font_color, 2) 
        #cv.imshow('Informacao',partial_info)

        #cv.circle(frame,(x,y),0,(0,255,255),8)
    
    elif (action == cv.EVENT_LBUTTONUP):

        if edit == True:
            print(3.5*np.sqrt((global_.ellipse_vertex[0][0] - global_.ellipse_vertex[1][0])**2 + (global_.ellipse_vertex[0][1] - global_.ellipse_vertex[1][1])**2))
            edit = False
            stop = False
            try:

                rec = functions.rectangle_ellipse((global_.ellipse_center,global_.ellipse_size,global_.ellipse_angle))
                
                cv.circle(cache,(global_.ellipse_vertex[0][0],global_.ellipse_vertex[0][1]),0,(0,0,255),8) # Ponto de baixo da elipse
                cv.circle(cache,(global_.ellipse_vertex[1][0],global_.ellipse_vertex[1][1]),0,(0,0,255),8) # Ponto de cima da elipse
                cv.circle(cache,(global_.ellipse_vertex[2][0],global_.ellipse_vertex[2][1]),0,(0,0,255),8) # Ponto lateral direito da elipse
                cv.circle(cache,(global_.ellipse_vertex[3][0],global_.ellipse_vertex[3][1]),0,(0,0,255),8) # Ponto lateral esquerdo da elipse
                cv.circle(cache,(global_.ellipse_center[0],global_.ellipse_center[1]),0,(0,0,255),8) # Ponto lateral esquerdo da elipse

                cv.circle(cache,(rec[0][0],rec[0][1]),0,(0,0,255),8) # Canto inferior esquerdo do retângulo
                cv.circle(cache,(rec[1][0],rec[1][1]),0,(0,0,255),8) # Canto superior esquerdo do retângulo
                cv.circle(cache,(rec[2][0],rec[2][1]),0,(0,0,255),8) # Canto superior direito do retângulo
                cv.circle(cache,(rec[3][0],rec[3][1]),0,(0,0,255),8) # Canto inferior direito do retângulo
                # Desenha o retângulo que envolve a elipse
                box = np.array([[rec[0][0],rec[0][1]],[rec[1][0],rec[1][1]],[rec[2][0],rec[2][1]],[rec[3][0],rec[3][1]]])
                cv.drawContours(cache,[box],0,(0,0,255),1)
                cv.ellipse(cache,global_.ellipse_param, (0,0,255), 1)
                cv.imshow('Original',cache)
            except:
                None

        
        

    return

def menu_start():

    # Configure widgets
    menu_path = widgets.Text(
        value='C:/Users/Videos',
        placeholder='Type something',
        description='Video Path:',
        disabled=False,
        layout=Layout(width='40%')
    )

    menu_px = widgets.BoundedFloatText(
        value=3.5,
        min=0,
        max = 1000,
        step=0.1,
        description='',
        disabled=False,
        layout=Layout(width='8%')
    )

    button_start = widgets.Button(
        description='Start Analysis',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='check' # (FontAwesome names without the `fa-` prefix)
    )

    
    px = widgets.HBox([Label("Distance between pixels"),menu_px,Label('μm')])
    start_menu = widgets.VBox([menu_path,px,button_start])
    display(start_menu)


    def start_analysis (a):

        file = str(menu_path.value).replace('\\' ,"/")      #Get file location
   
        binary_treshold, rot_axis_loc, scale, hor_axis = get.filters_parameters(file)

        import global_
        global_.avg_x_center, global_.avg_y_center = get.average_center(file,binary_treshold)

        video = cv.VideoCapture(file)   # Open video

        # Test if video was correctly open
        if video.isOpened: 
            print("Was open correctly")
    
            # get video fps and number of frames
            fps = int(video.get(cv.CAP_PROP_FPS))
            total_frame = video.get(cv.CAP_PROP_FRAME_COUNT)
            print("Video fps:",fps,"fps")
    
        else:
            print("Video wasn't opened corretly")
            exit    

        kernel_size = 5

        alpha = 0.6
        beta = (1.0 - alpha)

        method = False

        save = False
        canny_array = []
        superimposed_aray = []
        nai = []
        ellipse_param = ()

        time = 0

        time_axis = np.zeros(int(total_frame))
        area_axis = np.zeros(int(total_frame))
        width_axis = np.zeros(int(total_frame))
        height_axis = np.zeros(int(total_frame))
        width_axis2 = np.zeros(int(total_frame))
        height_axis2 = np.zeros(int(total_frame))
        volume_axis = np.zeros(int(total_frame))
        volume_axis2 = np.zeros(int(total_frame))
        ellipse_area_axis = np.zeros(int(total_frame))
        rev_volume_axis = np.zeros(int(total_frame))
        rev_area_axis = np.zeros(int(total_frame))

        ellipse_backup = np.zeros(int(total_frame),dtype=object)
        rec_backup = np.zeros(int(total_frame),dtype=object)
        ellipse_vertex_backup = np.zeros(int(total_frame),dtype=object)

        end = False
        join= False
        stop = False
        play = False

        first = True

        global skip, timer

        skip = False
        timer = 0

        import global_
        global_.ellipse_vertex = np.zeros((4,2))        

        #Loops video frame by frame
        while(not end):
            
            if join == False:
            
                # Get next frame
                ret, frame = video.read()   

                # Get current video time in seconds 
                video_time = video.get(cv.CAP_PROP_POS_MSEC)/1000

                # Define index of next, current and previous frame
                next_frame = video.get(cv.CAP_PROP_POS_FRAMES)
                current_frame = next_frame - 1
                previous_frame = current_frame - 1
        
                # Creates a copy of unaltered frame
                cache = np.copy(frame)
                
            else:
                frame = np.copy(cache)

            # Check if the frame is valid
            if ret:

                # Apply filters on frame
                filtered = functions.filters(frame)   

                # Execute on first iteraction
                if first == True:
                    
                    # Get frame information
                    frame_dimentions = frame.shape
                    frame_height = frame.shape[0]
                    frame_width = frame.shape[1]
                    frame_channels = frame.shape[2]
                    frame_size = (frame_width,frame_height)      
                    print(frame_size)
                    
                    # Create setup windows
                    cv.namedWindow('Comandos', cv.WINDOW_NORMAL)
                    cv.resizeWindow('Comandos', 700, 10)
                    cv.createTrackbar('Frame:', 'Comandos', int(current_frame), int(total_frame), video_progression)
                    first = False          
                    
                # Aplica o filtro binario e detecta a borda do coracao
                _ , binary_img = cv.threshold(filtered,binary_treshold,255,cv.THRESH_BINARY) # 240
                binary_edges = cv.Canny(binary_img, 50, 100, kernel_size, L2gradient= True)

                # Converte edges para opencv e sobrepoe o resultado em cima do frame original
                binary_edges_convert = cv.cvtColor(binary_img,cv.COLOR_GRAY2BGR)
                binary_superiposed = cv.addWeighted(frame, alpha, binary_edges_convert, beta, 0.0)

                # Encaixa a melhor elipse no contortno do coração
                contours, main_contour, global_.ellipse_param, rec, drawing, area, method,stop = functions.fit_ellipse(binary_img,method,join,stop,scale)
                
                if ellipse_backup[int(current_frame)] != 0:
                    global_.ellipse_param = ellipse_backup[int(current_frame)]
                    global_.ellipse_center = (int(global_.ellipse_param[0][0]),int(global_.ellipse_param[0][1]))
                    global_.ellipse_size = (int(global_.ellipse_param[1][0]),int(global_.ellipse_param[1][1]))
                    global_.ellipse_angle = int(global_.ellipse_param[2])
                    global_.ellipse_param = (global_.ellipse_center,global_.ellipse_size,global_.ellipse_angle)

                    rec = rec_backup[int(current_frame)]
                    global_.ellipse_vertex = ellipse_vertex_backup[int(current_frame)]

                cv.ellipse(frame, global_.ellipse_param, (90,90,255), 1)    # Draw ellipse on main frame
                
                # Abre janela com o frame original, filtrado, bordas detectadas e sobreposta em cima do frame original 
                cv.imshow('Original',frame)
                cv.imshow('Processado',binary_img)
                cv.imshow('Canny Binario Sobreposto',binary_superiposed)
                cv.imshow('Representacao',drawing)
                #cv.imshow('Informacao',info)

                #if time == 0:
                    #cv.createTrackbar('Angulo:', 'Comandos', int(ellipse_param[2]), 179, ellipse_rotation)
                                
                cv.setMouseCallback('Original', modify_elipse,[frame,rec,cache,d_pixel])
                    
                while True:
                    key = cv.waitKey(10)

                    if key != -1 or skip == True or play == True:
                        break

                if play == False and skip == False:

                    # Wait until user press a key that is listed on cfg
                    while not(key in cfg.key_list) :
                        if play == False:
                            key = cv.waitKey(0)
                        else:
                            key = cv.waitKey(1)
                            if key == cfg.play:
                                play = False
                                key = 0
                                print("ENTE")
                            #time.sleep(1/fps)

                    # Se o comando for "quit" ou não houver mais frames no video, entao o loop é finalizado
                    if (key == cfg.close) or (not ret):
                        end = True
                        join = False
                    elif key == cfg.play:
                        play = not(play)                
                        #method = True 
                        #join = False
                    elif key == ord('j'):
                        join = True
                    elif key == cfg.nxt:
                        join= False
                        cv.setTrackbarPos('Frame:', 'Comandos',int(next_frame))
                    elif key == cfg.prv:
                        join= False
                        cv.setTrackbarPos('Frame:', 'Comandos',int(previous_frame))

                else:
                    if not ret:
                        join = False
                        end = True

                if join == False:

                    height,width,ellipse_area,volume = get.ellipse_paramaters(d_pixel)
                    height2,width2,volume2 = get_paramaters2(d_pixel,contours,rot_axis_loc,hor_axis)     
                    volume_axis2[int(current_frame)] = volume2
                    height_axis2[int(current_frame)] = height2
                    width_axis2[int(current_frame)] = width2
                    vol_rev, area_rev = get.vol_solid_revo(contours,rot_axis_loc) 

                    # Guarda os valores do tempo, area e largura
                    time_axis[int(current_frame)] = video_time
                    area_axis[int(current_frame)] = area*d_pixel**2
                    volume_axis[int(current_frame)] = volume
                    height_axis[int(current_frame)] = height
                    width_axis[int(current_frame)] = width
                    ellipse_area_axis[int(current_frame)] = ellipse_area
                    rev_volume_axis[int(current_frame)] = vol_rev
                    rev_area_axis[int(current_frame)] = area_rev
                    import global_
                    ellipse_backup[int(current_frame)] = global_.ellipse_center,global_.ellipse_size,global_.ellipse_angle
                    rec_backup[int(current_frame)] = rec
                    
                    import global_
                    ellipse_vertex_backup[int(current_frame)] = global_.ellipse_vertex
                    
                    if key == cfg.prv:
                        if previous_frame >= 0:
                            video.set(cv.CAP_PROP_POS_FRAMES, previous_frame)
                            time = time - 1/fps
                        else:
                            previous_frame = 0
                            time = 0
                    else:
                        time = time + 1/fps

                    skip = False      


            else:
                break  
            
        video.release()
        cv.destroyAllWindows()    

        del video

        if (key == cfg.close):
            time_axis = time_axis[0:int(current_frame)]
            area_axis = area_axis[0:int(current_frame)]
            volume_axis2 = volume_axis2[0:int(current_frame)]
            height_axis2 = height_axis2[0:int(current_frame)]
            width_axis2 = width_axis2[0:int(current_frame)]
            ellipse_area_axis = ellipse_area_axis[0:int(current_frame)]
            volume_axis = volume_axis[0:int(current_frame)]
            height_axis = height_axis[0:int(current_frame)]
            width_axis = width_axis[0:int(current_frame)]
            rev_volume_axis = rev_volume_axis[0:int(current_frame)]
            rev_area_axis = rev_area_axis[0:int(current_frame)]

        ellipse_vol_const = (1/6)*np.pi*np.average(height_axis)*(width_axis)**2

        import results

        detection_method = 1  # 1 = Detecção por area do contorno do coracao
        weight = 120.1 # Peso = 120.1 mg

        results.show_results(weight,time_axis,width_axis,height_axis,width_axis2,height_axis2,
                        area_axis,rev_area_axis,ellipse_area_axis,
                        volume_axis,rev_volume_axis,ellipse_vol_const,volume_axis2,
                        detection_method)

        return

        


    button_start.on_click(start_analysis)  # Define callback function


    return