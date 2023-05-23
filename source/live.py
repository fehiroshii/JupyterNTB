import multiprocessing
import cv2 as cv
import numpy as np
from scipy.signal import argrelextrema

import global_
import cfg
import functions

from datetime import date
from os.path import exists


# Function that returns a list of max and min points of a array
def max_min_pts(lst):

    i = 0
    test = argrelextrema(lst, np.less_equal, order=10)

    min_list = test[0].tolist()
    max_list = []
    min_list.pop(0)

    while True:
        if i + 2 > len(min_list):
            break

        if (min_list[i+1] - min_list[i] < 4):
            min_list.pop(i)
        else:
            i = i + 1

    for i in range(len(min_list)):

        max_area = -1

        if i + 1 < len(min_list):
            temp = lst[min_list[i]:min_list[i+1]]

            for j in range(len(temp)):
                if temp[j] > max_area:
                    max_area = temp[j]
                    area_index = min_list[i] + j

            max_list.append(area_index)

    return max_list, min_list



# Function that apply filters on a image
def filters(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # image to grayscale
    blur = cv.medianBlur(gray,9)    # Apply median blur with kernel = 9
    gauss = cv.GaussianBlur(blur,(9,9),sigmaX=0,sigmaY=0)   # Apply Gaussian blur with kernel = 9
        
    return gauss  


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

from threading import Thread


# Function that returns a list of max and min points of a array
def max_min_pts(lst):

    i = 0
    test = argrelextrema(lst, np.less_equal, order=10)

    min_list = test[0].tolist()
    max_list = []
    min_list.pop(0)

    while True:
        if i + 2 > len(min_list):
            break
    
        if (min_list[i+1] - min_list[i] < 4):
            min_list.pop(i)
        else:
            i = i + 1


    for i in range(len(min_list)):

        max_area = -1

        if i + 1 < len(min_list):
            temp = lst[min_list[i]:min_list[i+1]]        

            for j in range(len(temp)):           
                if temp[j] > max_area :
                    max_area = temp[j]
                    area_index = min_list[i] + j

            max_list.append(area_index)   

    return max_list, min_list


class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=1,w = 320,h = 240):
        self.stream = cv.VideoCapture(src)
        self.stream.set(cv.CAP_PROP_FRAME_WIDTH, w)
        self.stream.set(cv.CAP_PROP_FRAME_HEIGHT, h)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.new_frame = False
    
    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()
                self.new_frame = True 

    def stop(self):
        self.stopped = True

class cam:
    '''Class that has live camera settings and configurations'''
    #def __init__(self):
    #    return self.open_cam(0,320,240)

    width = 320
    height = 240
    fps = 30
    
    current_frame = []

    recording = False
    adjustment = False
    analysis = False

    thold = 0
    ver_axis = width//2
    hor_axis = height//2
    scale = 100

    current_video = 'None'
    last_video = 'None'
    rec_id = 1

    def open_instance(self,index,w,h):
        with global_.live_logs:
            print("Attempt to open camera %1d" %(index))

        cap = cv.VideoCapture(index)

        cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)

        if not cap.read()[0]:
            with global_.live_logs:
                print("ERROR !! Can't connect to camera %1d, camera 0 will be chosen instead"%(index))
        
            cap.release()
            cap = cv.VideoCapture(0)
            cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)

        self.width  = int(cap.get(3))  # float `width`
        self.height = int(cap.get(4))  # float `height`
        self.fps = cap.get(5)

        with global_.live_logs:
            print("Camera %1d opened successfully with %3dx%3d"%(index,w,h))

        return cap
    

    def start_rec(self,local):
        path = str(local).replace('\\' ,"/") #Get file location from whidget
        today = str(date.today())

        # Videos are saved as "DD/MM/YY_NUMBER.avi", this loop verify the next
        # available NUMBER to create a new file and don't overwrite existing ones
        while (1):
            file = path + '/' + today + '_' + str(self.rec_id) + '.avi'
            file_exists = exists(file)
            if file_exists:
                self.rec_id =  self.rec_id + 1
            else:
                break

        # Set file location and create video file
        self.current_video = today + '_' + str(self.rec_id) + '.avi'
        file = path + '/' + self.current_video
        cv.setWindowTitle("Camera",self.current_video)

        self.recording = True

        with global_.live_logs:
            print("Recording Camera")

        return cv.VideoWriter(file, cv.VideoWriter_fourcc(*'DIVX'),30,(self.width,self.height))
    
    def stop_rec(self,video_obj):
        
        video_obj.release()     # Closes video
        cv.setWindowTitle("Camera","Camera")    # Set window to original name
        
        # Indicates that the video recording has stopped
        self.last_video = self.current_video
        self.current_video = 'None'
        self.recording = False      
        with global_.live_logs:
            print("Video {0} was saved successfuly".format(self.last_video))

    
    # Function that apply filters on a image
    def filters(img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # image to grayscale
        blur = cv.medianBlur(gray,9)    # Apply median blur with kernel = 9
        gauss = cv.GaussianBlur(blur,(9,9),sigmaX=0,sigmaY=0)   # Apply Gaussian blur with kernel = 9
        
        return gauss  


    def get_filter_parameters(self):
        # Create window with 3 trackbar
        cv.namedWindow('Comandos', cv.WINDOW_NORMAL)
        cv.resizeWindow('Comandos', 700, 100)
        cv.createTrackbar('Threshold','Comandos',253,253,self.thold_adjust)
        cv.createTrackbar('Rot Axis','Comandos',self.height//2,self.width,self.ver_axis_adjust)
        cv.createTrackbar('Diameter','Comandos',self.height//2,self.height,self.adjust_hor_axis)
        cv.createTrackbar('Scale','Comandos',100,100,self.scale_adjust)
    

    def thold_adjust(args):
        cam.adjustment = True
        cam.thold = args

    def ver_axis_adjust(args):
        cam.adjustment = True
        cam.ver_axis = args

    def adjust_hor_axis(args):
        cam.adjustment = True
        cam.hor_axis = args

    def scale_adjust(args):
        cam.adjustment = True
        cam.scale = args


class contour:

    scale = 100
    

    
    # Indicates if new information is stored and not analysed yet
    new_data = False



class ellipse:

    scale = 100

    # Array that contains the location of the 4 vertices of ellipse
    vertex = np.zeros((4,2))
    
    # Information of ellipse angle (range: 0 - 90)
    angle = 0

    # This tuple contains all elipsis information as
    # ((center_x,center_y),(height,width),angle)
    param = []

    # Array containing information
    area_record = np.array([])
    height_record = np.array([])
    width_record = np.array([])


    def fit(self,source_img):
        
        # Fit the contour that fits the heart shape
        cnts,_ = cv.findContours(source_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # Matrix that keep records of rectangle that encompass the ellipse of each contour detected  
        center_contours = []

        best_i = 0
        max_area = -1
    
        # Loop that gets the height, width and height of each contour 
        # Salva os parametros do contorno de maior area (parte central do coração)
        for i, c in enumerate(cnts):

            # Finds the coordinates of the rectangle that encompass the contour "c"
            top_left_x,top_left_y,largura,altura = cv.boundingRect(c)

            # Get rectangle center
            centerx = (2*top_left_x + largura)/2
            centery = (2*top_left_y + altura)/2
            center_contours.append([centerx,centery])

            # Find the largest contour 
            if cv.contourArea(c) > max_area: 
                max_area = cv.contourArea(c)
                estimated_centerx = centerx
                estimated_centery = centery
                best_i = i
        
        # Find largest contour 
        max_area = -1
        for i, c in enumerate(cnts):
            if cv.contourArea(c) > max_area: 
                max_area = cv.contourArea(c)
                main_cnt = i

        heart_cnt = cnts[main_cnt]
        scaled_heart_cnt = scale_contour(heart_cnt,self.scale/100)

        minEllipse = cv.fitEllipseDirect(scaled_heart_cnt)

        top_left_x,top_left_y,width,height = cv.boundingRect(scaled_heart_cnt)

        centerx = (2*top_left_x + largura)/2
        centery = (2*top_left_y + altura)/2

        center_contours.append([float(centerx),float(centery)])
       
        drawing_img = np.zeros((source_img.shape[0], source_img.shape[1], 3), dtype=np.uint8)
    
        for i, c in enumerate(cnts):
            
            if cv.contourArea(c) == max_area: 
           
                #  Obtain ellipse parameters 
                self.param =  minEllipse
                self.angle = self.param[2]
        
                box2 = self.rect_to_ellipse(self,minEllipse)
                self.get_ellipse_size(self)

                # Draw contours and ellipse
                cv.drawContours(drawing_img, scaled_heart_cnt, -1,cfg.main_cnt_color)
                cv.ellipse(drawing_img, minEllipse, cfg.ellipse_color, 1) 

                # Draw ellipse vertex        
                cv.circle(drawing_img,(self.vertex[0][0],self.vertex[0][1]),0,cfg.ELL_VRTX_CLR,8)
                cv.circle(drawing_img,(self.vertex[1][0],self.vertex[1][1]),0,cfg.ELL_VRTX_CLR,8)
                cv.circle(drawing_img,(self.vertex[2][0],self.vertex[2][1]),0,cfg.ELL_VRTX_CLR,8)
                cv.circle(drawing_img,(self.vertex[3][0],self.vertex[3][1]),0,cfg.ELL_VRTX_CLR,8)     
            else:
                # Draw other contours
                cv.drawContours(drawing_img, cnts, i, cfg.sec_cnt_color)

        cv.rectangle(drawing_img,(top_left_x,top_left_y),
                    (top_left_x+width,top_left_y+height),cfg.rec_color,2)
    
        
        return scaled_heart_cnt, box2 ,drawing_img


    def rect_to_ellipse(self,parameters):

        # List contaning the 4 vertex of rectangle that fits the parameter
        box2 = cv.boxPoints(parameters)
        box2 = np.intp(box2)     
    
        # Inferior ellipse vertex  
        self.vertex[0][0] = (box2[0][0] + box2[3][0])//2
        self.vertex[0][1] = (box2[0][1] + box2[3][1])//2

        # Superior ellipse vertex
        self.vertex[1][0] = (box2[1][0] + box2[2][0])//2
        self.vertex[1][1] = (box2[1][1] + box2[2][1])//2

        # Right ellipse vertex
        self.vertex[2][0] = (box2[2][0] + box2[3][0])//2
        self.vertex[2][1] = (box2[2][1] + box2[3][1])//2
            
        # Left ellipse vertex
        self.vertex[3][0] = (box2[0][0] + box2[1][0])//2
        self.vertex[3][1] = (box2[0][1] + box2[1][1])//2        

        self.vertex = self.vertex.astype(int)
    
        return box2

    # 
    def get_ellipse_size(self):
    
        # Calculates the position of the center of the ellipse
        self.ellipse_center = ((self.vertex[2][0] + self.vertex[3][0])//2,
                                  (self.vertex[0][1] + self.vertex[1][1])//2)

        self.ellipse_height = int(functions.distance(self.vertex[0][0],self.vertex[0][1],
                            self.vertex[1][0],self.vertex[1][1]))
        self.ellipse_width = int(functions.distance(self.vertex[2][0],self.vertex[2][1],
                            self.vertex[3][0],self.vertex[3][1]))
        self.ellipse_size = (self.vertex,self.vertex)

        return
    
    # Returns the value of ellipsis major axis "a"
    def get_a(self):  
        return np.sqrt((self.vertex[0][0] - self.vertex[1][0])**2 + 
            (self.vertex[0][1] - self.vertex[1][1])**2)

    
    # Returns the value of ellipsis minor axis "b" 
    def get_b(self):
        return np.sqrt((self.vertex[2][0] - self.vertex[3][0])**2 + 
            (self.vertex[2][1] - self.vertex[3][1])**2)


    # Get ellipses main properties ("a","b",Area and Volume)
    def get_properties(self,d_pixel):         

        # Get ellipsis major and minor axis
        a = self.get_a(self)
        b = self.get_b(self)
            
        # Calculate ellipses area and volume
        area = a*np.pi*b
        volume = int((1/6)*np.pi*a*b**2) 

        return int(a*d_pixel), int(b*d_pixel), int(area*d_pixel**2) ,int(volume*d_pixel**3)
    
class all_data:

    time_record = np.array([])

    # Store contour's information in array
    area_record = np.array([])

    new_data = False



class DataAnalysis:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self):
        self.new_data = False
        self.area_record = np.array([])
        self.time_record = np.array([])
        self.freq = 0       


    def start(self):
        Thread(target=self.Analysis, args=()).start()

        #with multiprocessing.Pool() as pool:
        #    pool.map(self.Analysis,([]))
        return self


    def Analysis(self):   
            if self.new_data:
                        #print(self.area_record)

                        max_points, min_points = max_min_pts(self.area_record)

                        freq = np.array([])

                        # Creates lists with frequency of each beat and volume on each systole
                        for i in range(len(min_points)):
                            if i + 1 < len(min_points):
                                t = self.time_record[min_points[i+1]] - self.time_record[min_points[i]]
                                freq = np.append(freq, 1/t)
                        try:
                            self.freq = np.average(freq)
                        except:
                            None
                        self.new_data = False

        


    def stop(self):
        self.stopped = True

