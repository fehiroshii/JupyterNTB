import ipywidgets as widgets
from IPython.display import display, clear_output
from ipywidgets import Layout
import numpy as np
import cv2 as cv
import sys
import os

from google.colab import drive

# adding Folder_2 to the system path
sys.path.insert(0, '/content/JupyterNTB/source/')

import entities
import results
import get
import functions
import cfg


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


# Displays UI (menu) that should be used in Google Colab
def menu_colab():
    global menu_path, button_start, start_menu

    # Configure widgets
    menu_path = widgets.Text(
        value='/content/drive/MyDrive/my_video.avi',
        placeholder='Type something',
        description='Video Name:',
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
        disabled=False,
        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='check'  # (FontAwesome names without the `fa-` prefix)
    )

    button_select = widgets.RadioButtons(
        options=['Google Colab', 'Google Drive'],
        #   value='pineapple', # Defaults to 'pineapple'
        #    layout={'width': 'max-content'}, # If the items' names are long
        description='Load Video From:',
        disabled=False
    )

    if os.path.exists('/content/drive/MyDrive'):
        button_login.disabled = True
        button_select.disabled = False
    else:
        button_select.disabled = True
        button_login.disabled = False

    left_menu = widgets.VBox([menu_path,
                              widgets.HBox([button_login, button_start])])
   
    start_menu = widgets.HBox([left_menu, button_select])


    def login_request(a):
        drive.mount('/content/drive')
        button_login.disabled = True
        button_start.disabled = False

    button_login.on_click(login_request)  # Define callback function
    button_start.on_click(show_menu)  # Define callback function

    display(start_menu)

def verfiy_file():

    None


def show_menu(a):
    clear_output(wait=False)
    global prev_x0, prev_x1, image, main, start_menu, logs_out,folder
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
        disabled=False,
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
        layout=Layout(width='90%', height='60%')
    )

    weight = 120.1  # Peso = 120.1 mg

    AllWidgets = results.AllWidgets(weight, loading_bar)

    # Hides loading menu until user press analyse button
    loading_bar.layout.visibility = 'hidden'

    trsh_bar = widgets.IntSlider(254, 0, 254, description='Threshold',
                                 continuous_update=False)
    vert_bar = widgets.IntSlider(0, 0, VideoInfo.height,
                                 description='Vertical Axis',
                                 continuous_update=False)
    hor_bar = widgets.IntSlider(0, 0, VideoInfo.width,
                                description='Horizontal Axis',
                                continuous_update=False)
    scale_bar = widgets.IntSlider(100, 0, 100,
                                  description='Scale',
                                  continuous_update=False)
    all_bar = widgets.VBox([trsh_bar, vert_bar, hor_bar, scale_bar])

    right_tabs = widgets.Accordion(children=[all_bar])
    right_tabs.set_title(0, 'Calibrate')

    right_menu = widgets.VBox([right_tabs, button_analysis])

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
        VideoInfo.current_frame = frame
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

    def trsh_calibration(a):
        # Get current treshold and scale from trackbar
        scale = scale_bar.value
        treshold = trsh_bar.value

        # Make copy of unaltered frame
        frame_copy = np.copy(VideoInfo.current_frame)

        # Crop image according to ROI
        roi = VideoInfo.roi
        frame_crop = frame_copy[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]

        # Apply filters
        frame_filtered = functions.filters(frame_crop)
        _, binary_img = cv.threshold(frame_filtered, treshold,
                                     255, cv.THRESH_BINARY)

        # Add a black border
        binary_img = cv.copyMakeBorder(
            binary_img,
            top=1,
            bottom=1,
            left=1,
            right=1,
            borderType=cv.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        # Detect binary edges
        binary_edges = cv.Canny(
            binary_img, 50, 100, 5, L2gradient=True)

        # Find contours
        contours, _ = cv.findContours(binary_edges,
                                      cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # Detect greatest contour and get its index
        max_area = -1
        for i, c in enumerate(contours):
            if cv.arcLength(c, True) > max_area:
                max_area = cv.arcLength(c, True)
                max_i = i

        # Apply scale to contours
        scaled_cont = functions.scale_contour(contours[max_i],
                                              scale/100)

        # Place the detected contour on the right place on original frame
        scaled_cont = scaled_cont + roi[0]

        # Draw contours 
        cv.drawContours(frame_copy, scaled_cont, -1, (255, 0, 255), 1)

        # Obtain ellipse param as  (center x,center y), (height,width), ângle
        min_center, min_dim, min_angle = cv.fitEllipseDirect(scaled_cont)

        # Draw ellipses on frame
        cv.ellipse(frame_copy, (min_center, min_dim, min_angle),
                   (255, 0, 255), 1)

        # Show both frames with ellipse adjusted
        is_success, im_buf_arr = cv.imencode(".png", frame_copy)
        byte_im1 = im_buf_arr.tobytes()
        image.value = byte_im1

        Ellipse.scale = scale/100
        Ellipse.thold = treshold

    def vert_calibration(a):
        frame_copy = np.copy(VideoInfo.current_frame)
        cv.line(frame_copy, (vert_bar.value, 0),
                (vert_bar.value, VideoInfo.height), (255, 0, 255), 1)
        
        is_success, im_buf_arr = cv.imencode(".png", frame_copy)
        byte_im1 = im_buf_arr.tobytes()
        image.value = byte_im1

        ManualEllipse.vrt_ax = vert_bar.value
        RevSolid.rot_axis = vert_bar.value

    def hor_calibration(a):
        frame_copy = np.copy(VideoInfo.current_frame)
        cv.line(frame_copy, (0, hor_bar.value),
                (VideoInfo.width, hor_bar.value,), (255, 0, 255), 1)
        
        is_success, im_buf_arr = cv.imencode(".png", frame_copy)
        byte_im1 = im_buf_arr.tobytes()
        image.value = byte_im1

        ManualEllipse.hor_ax = hor_bar.value

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
    button_calibrate.on_click(vert_calibration)
    button_preview.on_click(start_preview)
    button_analysis.on_click(start_analysis2)

    trsh_bar.observe(trsh_calibration)
    vert_bar.observe(vert_calibration)
    hor_bar.observe(hor_calibration)
    scale_bar.observe(trsh_calibration)


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