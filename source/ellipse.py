import cv2 as cv
import numpy as np
import functions


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
def get_paramaters(d_pixel):         

    # Get ellipsis major and minor axis
    a = get_a()
    b = get_b()
            
    # Calculate ellipses area and volume
    area = a*np.pi*b
    volume = int((1/6)*np.pi*a*b**2) 

    return int(a*d_pixel), int(b*d_pixel), int(area*d_pixel**2) ,int(volume*d_pixel**3)