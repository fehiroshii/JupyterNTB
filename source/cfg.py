####################################################
#                Keyboard Commands                 #
####################################################
"""
This section covers the keyboard shortcuts

The shortcut is specified by its decimal ASCII representation

For more details, use the "decimal" column available at
https://python-reference.readthedocs.io/en/latest/docs/str/ASCII.html

Or use the python command ord("X") in order to obtain ]
the decimal representaton of "X"
"""

confirm = 13        # Confirm user config       Enter = 10
close = 27          # Closes video              Esc = 27
nxt = ord(".")      # Next frame                ord(".") = 46
prv = ord(",")      # Previous frame            ord(",") = 44
play = ord(" ")     # Play/Pause                ord(" ") = 32

key_list = [close, nxt, prv, play]

####################################################
#              Color Configurations                #
####################################################
"""
Colors are defined as tuple in the format (R,G,B)
In order to deactivate any element, replace color with (0,0,0)
"""

HEART_CLR = (0, 255, 0)          # Color of main contour
SEC_CNT_CLR = (255, 255, 255)       # color of other contours

ellipse_color = (255, 0, 255)         # Color of ellipse  (0,255,255)
ELL_VRTX_CLR = (255, 0, 255)        # Color of ellipse vertex (0,255,255)

REC_CLR = (0, 255, 0)               # Color of rectangle
