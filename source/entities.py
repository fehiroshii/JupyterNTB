import numpy as np
import cv2 as cv
import cfg

import functions
import get


class VideoInfo():
    '''Class for keeping tracking of general video information'''

    def __init__(self, path) -> None:
        self.video_options = []
        self.get_current(path)
        self.current_frame = np.array([])

    def get_current(self, path):
        self.file = str(path).replace('\\', "/")  # Get file location

        self.file = self.file.replace('"', '')
        self.folder = self.file[0:self.file.rfind("/")+1]
        self.name = self.file[self.file.rfind("/")+1:]

        video = cv.VideoCapture(self.file)
        self.fps = int(video.get(cv.CAP_PROP_FPS))
        self.total_frame = int(video.get(cv.CAP_PROP_FRAME_COUNT))

        self.video_options.append(self.name)
        self.time_options = get.timestamp(self.fps, self.total_frame)

        res, frame = video.read()
        self.height = frame.shape[0]
        self.width = frame.shape[1]
        self.roi = [(0, 0), (self.width, self.height)]

        video.release()

    def add_video(self, path):
        None


class RevolutionSolid():
    # Rotation axis location on the frame
    rot_axis = 0

    # Array containing information
    area_record = np.array([], dtype=object)
    volume_record = np.array([], dtype=object)
    height_record = np.array([], dtype=object)
    width_record = np.array([], dtype=object)

    def __init__(self, rot_axis):
        self.rot_axis = rot_axis

    def get_fx(self, x, y, axis):

        fx = np.array([], dtype=object)

        # Sort x and y
        p = y.argsort()
        y = y[p]
        x = x[p]

        values, un_index, counts = np.unique(
            y, return_index=True, return_counts=True)

        counts = counts - 1

        for i in range(len(counts)):
            if counts[i] == 0:
                fx = np.append(fx, x[un_index[i]])
            else:
                fx = np.append(fx, np.average(
                    x[un_index[i]: un_index[i] + counts[i]]))

        if np.average(fx) < axis:
            return axis - fx
        else:
            return fx - axis

    # Get area and volume from the revolution solid
    def get_properties(self, cont):
        # Array that will hold x and y coordinates from
        # the part of the contour that is to the right and left 
        # of the revolution axis
        right_cont_x = np.array([], dtype=object)
        right_cont_y = np.array([], dtype=object)
        left_cont_x = np.array([], dtype=object)
        left_cont_y = np.array([], dtype=object)
        fx2 = np.array([], dtype=object)

        # Loops throught the contour's points and separates into two parts
        # those points that are to the left and right of the revolution axis
        for v in range(len(cont)):
            x_cont = cont[v][0][0]
            y_cont = cont[v][0][1]

            if x_cont > self.rot_axis:
                right_cont_x = np.append(right_cont_x, x_cont)
                right_cont_y = np.append(right_cont_y, y_cont)
            else:
                left_cont_x = np.append(left_cont_x, x_cont)
                left_cont_y = np.append(left_cont_y, y_cont)

        # a = int(np.amin(right_cont_y))
        # b = int(np.amax(right_cont_y))
        
        # Corrects the heart contour to a injective function
        fx = self.get_fx(right_cont_x, right_cont_y, self.rot_axis)

        fx2 = fx**2

        v1 = np.sum(fx2)/2
        a1 = np.sum(fx)

        fx = self.get_fx(left_cont_x, left_cont_y, self.rot_axis)
        fx2 = fx**2

        v2 = np.sum(fx2)/2
        a2 = np.sum(fx)

        v = v1 + v2
        a = a1 + a2

        self.volume = np.pi*v
        self.area = a

    # Update volume, area, width and height record array
    def record_properties(self, cont, frame):
        # Get revolution solid's current height, width, area and volume
        self.get_properties(cont)

        # Update all records
        if self.area_record.size == frame:
            self.area_record = np.append(self.area_record, self.area)
            self.volume_record = np.append(self.volume_record, self.volume)
        else:
            self.area_record[frame] = self.area
            self.volume_record[frame] = self.volume
        return

    # Clear all previous records
    def clear_records(self):
        self.area_record = np.array([], dtype=object)
        self.volume_record = np.array([], dtype=object)


class Ellipse():
    # Array that contains the location of the 4 vertices of ellipse
    vertex = np.zeros((4, 2), dtype=object)

    # Array that conatains the location of the 4 arestas of rectangle
    # that entagles the contour
    rect = np.array([], dtype=object)

    # Percentage of horizontal scale applied on the contour
    scale = 1.0

    # Binary Threshold that needs to be applied to the image
    thold = 249

    # This tuple contains all elipsis information as
    # ((center_x,center_y),(height,width),angle)
    param = np.array([], dtype=object)

    # Array containing information about ellipse area,
    # volume, height and width trought video anaylised
    area_record = np.array([], dtype=object)
    volume_record = np.array([], dtype=object)
    height_record = np.array([], dtype=object)
    width_record = np.array([], dtype=object)

    # Array that contains the ellipse parameters trought video
    # Useful when user goes back on video
    param_record = []
    vertex_record = np.array([], dtype=object)

    def __init__(self, scale):
        self.param_record = []
        self.scale = scale
        return

    def fit(self, src_img, roi):

        # Fit the contour that fits the heart shape
        cnts, _ = cv.findContours(src_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # Find largest contour
        max_area = -1
        for i, c in enumerate(cnts):
            if cv.contourArea(c) > max_area:
                max_area = cv.contourArea(c)
                main_cnt = i

        # Get the heart contour and scale it properly
        heart_cnt = cnts[main_cnt]
        scaled_heart_cnt = functions.scale_contour(
            heart_cnt, self.scale)

        self.param = cv.fitEllipseDirect(scaled_heart_cnt)

        top_left_x, top_left_y, width, height = cv.boundingRect(scaled_heart_cnt)

        dwg_img = np.zeros((src_img.shape[0], src_img.shape[1], 3), dtype=np.uint8)

        # Draws every contour and ellipse around the insect heart
        for i, c in enumerate(cnts):

            if cv.contourArea(c) == max_area:
                #  Obtain ellipse parameters
                self.angle = self.param[2]
                self.get_vertex()
                self.get_size()

                # Draw contours and ellipse
                cv.drawContours(dwg_img, scaled_heart_cnt, -1,
                                cfg.HEART_CLR)
                cv.ellipse(dwg_img, self.param, cfg.ellipse_color, 1)

                # Draw ellipse vertex
                cv.circle(dwg_img, (self.vertex[0][0], self.vertex[0][1]), 0,
                          cfg.ELL_VRTX_CLR, 8)
                cv.circle(dwg_img, (self.vertex[1][0], self.vertex[1][1]), 0,
                          cfg.ELL_VRTX_CLR, 8)
                cv.circle(dwg_img, (self.vertex[2][0], self.vertex[2][1]), 0,
                          cfg.ELL_VRTX_CLR, 8)
                cv.circle(dwg_img, (self.vertex[3][0], self.vertex[3][1]), 0,
                          cfg.ELL_VRTX_CLR, 8)
            else:
                # Draw other contours
                cv.drawContours(dwg_img, cnts, i, cfg.SEC_CNT_CLR)

        cv.rectangle(dwg_img, roi[0], roi[1], cfg.REC_CLR, 2)

        return scaled_heart_cnt, dwg_img

    # Converts the coordinates of the 4 vertex of rectangle that fits the ellipse
    # to a list (vertex) containing the coordinates of ellipse's 4 vertex
    def get_vertex(self):
        # List contaning the 4 vertex of rectangle that fits the parameter
        self.rect = cv.boxPoints(self.param)
        self.rect = np.intp(self.rect)

        # Inferior ellipse vertex
        self.vertex[0][0] = (self.rect[0][0] + self.rect[3][0])/2
        self.vertex[0][1] = (self.rect[0][1] + self.rect[3][1])/2

        # Superior ellipse vertex
        self.vertex[1][0] = (self.rect[1][0] + self.rect[2][0])/2
        self.vertex[1][1] = (self.rect[1][1] + self.rect[2][1])/2

        # Right ellipse vertex
        self.vertex[2][0] = (self.rect[2][0] + self.rect[3][0])/2
        self.vertex[2][1] = (self.rect[2][1] + self.rect[3][1])/2

        # Left ellipse vertex
        self.vertex[3][0] = (self.rect[0][0] + self.rect[1][0])/2
        self.vertex[3][1] = (self.rect[0][1] + self.rect[1][1])/2

        self.vertex = self.vertex.astype(int)

        return

    # Get ellipse's current size
    def get_size(self):

        self.get_vertex()   # Get ellipse's current vertex coordinates

        # Calculates the position of the center of the ellipse
        self.center = ((self.vertex[2][0] + self.vertex[3][0])/2,
                       (self.vertex[0][1] + self.vertex[1][1])/2)

        self.height = functions.distance(self.vertex[0][0], self.vertex[0][1],
                                         self.vertex[1][0], self.vertex[1][1])
        self.width = functions.distance(self.vertex[2][0], self.vertex[2][1],
                                        self.vertex[3][0], self.vertex[3][1])
        self.size = (self.width, self.height)

        self.param = (self.center, self.size, self.angle)

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
    def get_properties(self):

        # Get ellipsis major and minor axis
        self.a = self.get_a()
        self.b = self.get_b()

        # Calculate ellipses area and volume
        self.area = self.a*np.pi*self.b
        self.volume = int((1/6)*np.pi*self.a*self.b**2)
        return

    # Update volume, area, width and height record array
    def record_properties(self, frame):
        # Get ellipse's current height, width, area and volume
        self.get_size()
        self.get_properties()

        # Update all records
        if self.area_record.size == frame:
            self.area_record = np.append(self.area_record, self.area)
            self.volume_record = np.append(self.volume_record, self.volume)
            self.height_record = np.append(self.height_record, self.height)
            self.width_record = np.append(self.width_record, self.width)
            self.param_record.append(self.param)
            self.vertex_record = np.append(self.vertex_record, self.vertex)
        else:
            self.area_record[frame] = self.area
            self.volume_record[frame] = self.volume
            self.height_record[frame] = self.height
            self.width_record[frame] = self.width
            self.param_record[frame] = self.param
            self.vertex_record[frame] = self.vertex
        return
    
    # Clear all previous records
    def clear_records(self):
        self.area_record = np.array([], dtype=object)
        self.volume_record = np.array([], dtype=object)
        self.height_record = np.array([], dtype=object)
        self.width_record = np.array([], dtype=object)
        self.param_record = []
        self.vertex_record = np.array([], dtype=object)


class ManualEllipse():

    # Initialize manual ellipse's constant 
    def __init__(self, hor_ax, vrt_ax):
        self.hor_ax = hor_ax    # Location of the horizontal axis
        self.vrt_ax = vrt_ax    # Location of the vertical axis

        # Array containing information about manual ellipse area,
        # volume, height and width trought video anaylised
        self.volume_record = np.array([], dtype=object)
        self.height_record = np.array([], dtype=object)
        self.width_record = np.array([], dtype=object)

    #
    def get_properties(self, cont):

        left_x = np.array([], dtype=object)
        right_x = np.array([], dtype=object)

        interval = 5

        for i in range(len(cont)):
            if abs(cont[i][0][1] - self.hor_ax) <= 5:
                if cont[i][0][0] <= self.vrt_ax:
                    left_x = np.append(left_x, self.vrt_ax - cont[i][0][0])
                else:
                    right_x = np.append(right_x, cont[i][0][0] - self.vrt_ax)

        topmost = tuple(cont[cont[:, :, 1].argmin()][0])
        bottommost = tuple(cont[cont[:, :, 1].argmax()][0])

        self.width = np.average(left_x) + np.average(right_x)
        self.height = bottommost[1] - topmost[1]
        self.volume = (1/6)*np.pi*self.height*self.width**2

        return

    # Update volume, area, width and height record array
    def record_properties(self, cont, frame):
        # Get ellipse's current height, width and volume
        self.get_properties(cont)

        # Update all records
        if self.volume_record.size == frame:
            self.volume_record = np.append(self.volume_record, self.volume)
            self.height_record = np.append(self.height_record, self.height)
            self.width_record = np.append(self.width_record, self.width)
        else:
            self.volume_record[frame] = self.volume
            self.height_record[frame] = self.height
            self.width_record[frame] = self.width
        return

    # Clear all previous records
    def clear_records(self):
        self.volume_record = np.array([], dtype=object)
        self.height_record = np.array([], dtype=object)
        self.width_record = np.array([], dtype=object)



