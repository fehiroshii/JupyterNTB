import ipywidgets as widgets
from IPython.display import display, clear_output
from ipywidgets import Layout, Label
import numpy as np
import cv2 as cv




import entities
import results
import get
import functions
import cfg


try:
    from google.colab import drive
except:
    None


d_pixel = 3.5  # Distancia entre pixel = 3.5 um


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


# Displays UI (menu) that should be used in Google Colab
def menu_colab():
    global menu_path, button_start, start_menu

    # Configure widgets
    menu_path = widgets.Text(
        value='/content/drive/MyDrive/my_video.avi',
        placeholder='Type something',
        description='Video Path:',
        disabled=False,
        layout=Layout(width='40%')
    )

    button_login = widgets.Button(
        description='Mount Google Drive',
        disabled=False,
        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='check'  # (FontAwesome names without the `fa-` prefix)
    )

    button_start = widgets.Button(
        description='Load Video',
        disabled=True,
        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='check'  # (FontAwesome names without the `fa-` prefix)
    )

    start_menu = widgets.VBox([menu_path, 
                               widgets.HBox([button_login, button_start])])
    

    def login_request(a):
        drive.mount('/content/drive')
        button_login.disabled = True
        button_start.disabled = False

    button_login.on_click(login_request)  # Define callback function
    button_start.on_click(show_menu)  # Define callback function

    display(start_menu)


def show_menu(a):
    clear_output(wait=False)
    global prev_x0, prev_x1, image, main, start_menu,logs_out,folder
    from ipywidgets import interact, interactive, fixed, interact_manual

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Initialize main classes used for calculations
    VideoInfo = entities.VideoInfo(menu_path.value)
    Ellipse = entities.Ellipse(0)
    ManualEllipse = entities.ManualEllipse(0,0)
    RevSolid = entities.RevolutionSolid(0)
    
    
    file = VideoInfo.file
    folder = VideoInfo.folder
    name = VideoInfo.name

    vd_display = cv.VideoCapture(file)   # Open video

    total_frame = VideoInfo.total_frame
    fps = VideoInfo.fps

    # Get list of all video timestamp array as (00:00, 00:01, ...)
    options = VideoInfo.time_options 

    # Read first frame and converts to binary
    res, frame = vd_display.read()
    is_success, im_buf_arr = cv.imencode(".png", frame)
    byte_im1 = im_buf_arr.tobytes()

    prev_x0 = 0
    prev_x1 = total_frame - 1

    prev_x0 = 0
    prev_x1 = len(options) - 1

    # Configure widgets for left menu
    load_path = widgets.Text(
        value= menu_path.value,
        placeholder='Type something',
        description='Path:',
        disabled=False,
        layout=Layout(width='95%')
    )

    button_load = widgets.Button(
        description='Load Video',
        disabled=False,
        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='check',  # (FontAwesome names without the `fa-` prefix)
        layout=Layout(width='95%')
    )

    select_video = widgets.Select(
        options=[name],
        value=name,
        # rows=10,
        description='Video:',
        disabled=False,
        layout=Layout(width='95%')
    )

    left_menu = widgets.VBox([load_path, button_load, select_video])

    # Configure widgets for main menu
    range_bar = widgets.SelectionRangeSlider(
        options=options,
        index=(0, len(options)-1),
        description='',
        disabled=False
    )

    image = widgets.Image(
        value=byte_im1,
        format='png',
        width=320,
        height=240
    )

    main_menu = widgets.VBox([image, range_bar])

    button_roi = widgets.Button(
        description='Select ROI',
        disabled=False,
        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='square'  # (FontAwesome names without the `fa-` prefix)
    )

    button_calibrate = widgets.Button(
        description='Calibration',
        disabled=False,
        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='gear'  # (FontAwesome names without the `fa-` prefix)
    )

    button_preview = widgets.Button(
        description='Preview Video',
        disabled=True,
        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='eye'  # (FontAwesome names without the `fa-` prefix)
    )

    button_analysis = widgets.Button(
        description='Analyse Video',
        disabled=True,
        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='crosshairs'  # (FontAwesome names without the `fa-` prefix)
    )

    loading_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=10,
        description='Loading',
        bar_style='',  # 'success', 'info', 'warning', 'danger' or ''
        style={'bar_color': '#66BB6A'},
        orientation='horizontal',
        layout=Layout(width='90%', height='13px')
    )

    weight = 120.1  # Peso = 120.1 mg

    AllWidgets = results.AllWidgets(weight, loading_bar)

    # Hides loading menu until user press analyse button
    loading_bar.layout.visibility = 'hidden'
    
    right_menu = widgets.VBox([button_roi, button_calibrate, button_preview,
                               button_analysis], layout=Layout(justify_content='flex-start'))

    main_ui = widgets.HBox([left_menu, main_menu, right_menu], layout=Layout(justify_content='center'))


    all_ui = widgets.VBox([main_ui, loading_bar])

    global out, first

    def f(x):   
        global prev_x0, prev_x1, image, main, out
        frame_number = 0
        if x[1] != prev_x1:
            frame_number = x[1]
            prev_x1 = x[1]
        elif x[0] != prev_x0:
            frame_number = x[0]
            prev_x0 = x[0]

        vd_display.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        res, frame = vd_display.read()
        is_success, im_buf_arr = cv.imencode(".png", frame)
        byte_im1 = im_buf_arr.tobytes()
        image.value = byte_im1

        return
    
    def update_image(curr_indx):
        global prev_x0, prev_x1, image, main, out
        frame_number = 0
        if range_bar.index[1] != prev_x1:
            frame_number = range_bar.index[1]
            prev_x1 = range_bar.index[1]
        elif range_bar.index[0] != prev_x0:
            frame_number = range_bar.index[0]
            prev_x0 = range_bar.index[0]

        vd_display.set(cv.CAP_PROP_POS_FRAMES, frame_number*fps)
        res, frame = vd_display.read()
        is_success, im_buf_arr = cv.imencode(".png", frame)
        byte_im1 = im_buf_arr.tobytes()
        image.value = byte_im1

    out = widgets.interactive_output(update_image, {'curr_indx': range_bar})

    display(all_ui, out)

    def load_new(a):
        VideoInfo.get_current(load_path.value)
        select_video.options = VideoInfo.video_options

    def start_roi(a):
        VideoInfo.roi = calibrate_roi(file, range_bar.index)

    def start_calibration(a):
        length = (range_bar.index[0]*fps, range_bar.index[1]*fps)
        thold_bin, rotational_axis, scale, hor_axis = get.filters_parameters(
            file, length, VideoInfo.roi)

        # Update classes parameters
        Ellipse.scale = scale
        Ellipse.thold = thold_bin
        ManualEllipse.hor_ax = hor_axis
        ManualEllipse.vrt_ax = rotational_axis
        RevSolid.rot_axis = rotational_axis

        # Enable analysis and preview button
        button_analysis.disabled = False
        button_preview.disabled = False
    
    def start_preview(a):
        length = (range_bar.index[0]*fps, range_bar.index[1]*fps)
        preview(file, length, VideoInfo.roi, Ellipse, ManualEllipse)

    def start_analysis2(a):
        Ellipse.clear_records()
        ManualEllipse.clear_records()
        RevSolid.clear_records()

        length = (range_bar.index[0]*fps, range_bar.index[1]*fps)

        results = analyse(file, length, VideoInfo.roi, Ellipse.thold,
                          Ellipse, ManualEllipse, RevSolid, AllWidgets)
        
        AllWidgets.save_records(results)
        AllWidgets.displayed = True

    # Define callback function
    button_load.on_click(load_new)
    button_roi.on_click(start_roi)
    button_calibrate.on_click(start_calibration)
    button_preview.on_click(start_preview)
    button_analysis.on_click(start_analysis2)


# Returns a ROI (Region Of Interest) based on a rectangle defined by user
def calibrate_roi(file, lenght):

    video = cv.VideoCapture(file)        
    video.set(cv.CAP_PROP_POS_FRAMES, lenght[0])

    first = True
    count = 0

    # Finds a frame with maximum luminous intensity (Diastole) and a 
    # frame with minimum luminous intesity (Systole) in the range defined by user
    while (count <= ((lenght[1]-lenght[0])//3)):

        ret, img = video.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, bw_img = cv.threshold(gray,160,255,cv.THRESH_BINARY)

        wht_px = np.count_nonzero(bw_img)

        # Execute on first interaction
        if first:
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

    original_min_frame = np.copy(min_frame)

    # Loop that ends when user finishes ROI calibration
    while True:
    
        # Create a window and enables user to select a ROI
        roi = cv.selectROI("ROI on Diastole", max_frame, showCrosshair=False)

        # Draw selected ROI on a systole frame
        systole_rec = cv.rectangle(min_frame, 
                                  (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), 
                                  (0, 255, 0), 3)
        cv.imshow("ROI on Systole", systole_rec)

        key = -1

        while not (key in[13, 27]):
            key = cv.waitKey(1)
    
        # Checks if user pressed "Enter" and finishes ROI calibration
        if key == 13:
            roi = np.array([(roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3])])
            video.release()
            cv.destroyAllWindows()
            break

        # Checks if user pressed "Esc" and redo ROI calibration
        elif key == 27:
            min_frame = np.copy(original_min_frame)
            cv.destroyAllWindows()
    
    return roi


def preview(file, range, roi, Ellipse, ManualEllipse):

    video = cv.VideoCapture(file)   # Open video
    video.set(cv.CAP_PROP_POS_FRAMES, range[0])

    end = False
    first = True
    display_img = 0

    # Loops video frame by frame
    while (not end):

        # Get next frame
        ret, frame = video.read()

        # Define index of next, current and previous frame
        next_frame = int(video.get(cv.CAP_PROP_POS_FRAMES))
        current_frame = int(next_frame - 1)

        # Check if the frame is valid
        if ret:

            frame_crop = frame[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]

            # Apply filters on frame
            filtered = functions.filters(frame_crop)

            # Execute on first iteraction
            if first == True:

                # get video fps and number of frames
                fps = int(video.get(cv.CAP_PROP_FPS))

                first = False

            # Apply binary filter
            _, binary_img = cv.threshold(filtered, Ellipse.thold,
                                         255, cv.THRESH_BINARY)
            
            # Add a black border to both images
            binary_img = cv.copyMakeBorder(
                binary_img,
                top= roi[0][1] + 1,
                bottom=frame.shape[0] - roi[1][1] + 1,
                left=roi[0][0] +1,
                right=frame.shape[1] - roi[1][0] + 1,
                borderType=cv.BORDER_CONSTANT,
                value=(0, 0, 0)
            )

            # Fit ellipse on the binary image and returns an image
            heart_contour, drawing = Ellipse.fit(binary_img, roi)

            if display_img == 0:
                cv.imshow('Applied Settings', drawing)
            elif display_img == 1:
                cv.rectangle(frame, roi[0], roi[1], cfg.REC_CLR, 2)
                cv.ellipse(frame, Ellipse.param, cfg.ellipse_color, 1)
                cv.imshow('Applied Settings', frame)
            
            key = cv.waitKey(1000//fps)
            
            if key == ord('1'):
                display_img = 0
            elif key == ord('2'):
                display_img = 1
            elif key == cfg.close:
                end = True

        else:
            break

        if (not ret) or (current_frame == range[1]):
            end = True

    video.release()
    cv.destroyAllWindows()


def analyse(file, range, roi, thold_bin, ellipse, manual_ellipse, rev_solid, AllWidgets):

    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    
    video = cv.VideoCapture(file)   # Open video
    video.set(cv.CAP_PROP_POS_FRAMES, range[0])

    total_frame = video.get(cv.CAP_PROP_FRAME_COUNT)

    time_axis = np.zeros(int(total_frame), dtype=object)
    cont_area_record = np.zeros(int(total_frame), dtype=object)

    end = False

    first = True

    outt0 = widgets.Output()
    
    
    AllWidgets.loading_bar.max = range[1]
    AllWidgets.loading_bar.min = range[0]
    AllWidgets.loading_bar.value = range[0]
    AllWidgets.loading_bar.layout.visibility = 'visible'

    # Loops video frame by frame
    while (not end):

        # Get next frame
        ret, frame = video.read()

        # Get current video time in seconds
        video_time = video.get(cv.CAP_PROP_POS_MSEC)/1000

        # Define index of next, current and previous frame
        next_frame = int(video.get(cv.CAP_PROP_POS_FRAMES))
        current_frame = int(next_frame - 1)

        AllWidgets.loading_bar.value = current_frame

        # Check if the frame is valid
        if ret:

            frame_crop = frame[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]

            # Apply filters on frame
            filtered = functions.filters(frame_crop)

            # Execute on first iteraction
            if first == True:

                # get video fps and number of frames
                fps = int(video.get(cv.CAP_PROP_FPS))

                first = False

            # Apply binary filter
            _, binary_img = cv.threshold(filtered, thold_bin,
                                         255, cv.THRESH_BINARY)
            
            # Add a black border to both images
            binary_img = cv.copyMakeBorder(
                binary_img,
                top= roi[0][1] + 1,
                bottom=frame.shape[0] - roi[1][1] + 1,
                left=roi[0][0] +1,
                right=frame.shape[1] - roi[1][0] + 1,
                borderType=cv.BORDER_CONSTANT,
                value=(0, 0, 0)
            )

            # Fit ellipse on the binary image and returns an image
            heart_contour, drawing = ellipse.fit(binary_img, roi)

            cont_area_record[current_frame - range[0]] = cv.contourArea(heart_contour)


            # Update ellipse's area, volume, height and width records
            ellipse.record_properties(current_frame - range[0])               

            # Update revolution solid's area and volume records
            rev_solid.record_properties(heart_contour, 
                                        current_frame - range[0])

            # Update manual ellipse's area, volume, height and width records
            manual_ellipse.record_properties(heart_contour, 
                                             current_frame- range[0])

            # Keep information of current time
            time_axis[int(current_frame - range[0])] = video_time

        else:
            break


        if (not ret) or (current_frame == range[1]):
            end = True
            cont_area_record = cont_area_record[0:next_frame - range[0]]
            time_axis = time_axis[0:next_frame - range[0]]
            break


    video.release()


    ellipse_vol_const = (
        1/6)*np.pi*np.average(ellipse.height_record)*(ellipse.width_record)**2

    import results

    detection_method = 1  # 1 = Detecção por area do contorno do coracao
    weight = 120.1  # Peso = 120.1 mg

    AllWidgets.loading_bar.layout.visibility = 'hidden' # Hides loading bar

    results.show_results(weight, time_axis,
                         ellipse, manual_ellipse, rev_solid, ellipse_vol_const,
                         cont_area_record, 3.5, detection_method, folder, outt0, AllWidgets)   
    
    return (weight, time_axis,
            ellipse, manual_ellipse, rev_solid, ellipse_vol_const,
            cont_area_record, 3.5, detection_method, folder, outt0)