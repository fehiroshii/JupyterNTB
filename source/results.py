import pandas as pd
from matplotlib import pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
import tables
import numpy as np
from scipy.signal import argrelextrema

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

# Function that creates adn draw plots 
def plot_img2(x_axes, y_axes, x_title, y_title, plot_title, h,w, p_enable,minimum, maximum,units):
   
    fig = plt.figure()  # an empty figure with no Axes
    fig, ax = plt.subplots(figsize=(h, w))
    ax.plot(x_axes, y_axes)  # Plot some data on the axes.

    # Set title to Axes
    ax.set(xlabel= x_title, ylabel= y_title + "(" + units + ")",title= plot_title)
         
    ax.grid() # Put grids on plot

    # If enabled, put systole and dyastole points on plot
    if p_enable:
        ax.plot(x_axes[minimum],  y_axes[minimum], 'o',color='orange', markersize = 7, label = 'Systole')
        ax.plot(x_axes[maximum],  y_axes[maximum], 'o',color='green', markersize = 7, label = 'Diastole')
        ax.legend()
    
    plt.show(fig)   # Show plot

# Function that plot all results 
def plot_results(point_enable,units,conversion,x,y,min_list,max_list,trgg):
    
    time_axis = x
    area_axis = y[0]
    ellipse_area_axis = y[1]
    width_axis = y[2]
    height_axis = y[3]
    volume_axis = y[4]

    if trgg == 1:
        plot_img2(time_axis,area_axis*conversion[1], "Time (sec)" , "Area ", "Heart's area ", 30 ,6, point_enable, min_list, max_list, units[1])
    elif trgg == 2:
        plot_img2(time_axis,ellipse_area_axis*conversion[1], "Time (sec)" , "Area ", "Ellipsis area", 30 ,6,point_enable , min_list, max_list, units[1])
    
    plot_img2(time_axis,width_axis*conversion[0], "Time (sec)" , "Diameter ", "Minor Axis ellipsoid", 30 ,6,point_enable, min_list, max_list,units[0])
    plot_img2(time_axis,height_axis*conversion[0], "Time (sec)" , "Major Axis ", "Major Axis ellipsoid", 30 ,6,point_enable , min_list, max_list,units[0])
    plot_img2(time_axis,volume_axis*conversion[2], "Time (sec)" , "Volume ", "Insect's heart volume", 30 , 6, point_enable, min_list, max_list,units[2])

# Show table with all data
def show_maintable(units,conversion,min_lst,max_lst,freq):

    min_volume = min_lst[0]
    min_width = min_lst[1]
    min_height = min_lst[2]

    max_volume = max_lst[0]
    max_width = max_lst[1]
    max_height = max_lst[2]
    
    min_volume = np.append(min_volume,np.average(min_volume))
    min_width = np.append(min_width,np.average(min_width))
    min_height = np.append(min_height,np.average(min_height))

    max_volume = np.append(max_volume,np.average(max_volume))
    max_width = np.append(max_width,np.average(max_width))
    max_height = np.append(max_height,np.average(max_height))
    
    freq  = np.append(freq,np.average(freq))

    all_data = {
    "Frequency " + "(" + "hz" +")": freq,
    "Systolic Volume " + "(" + units[2] + ")": min_volume*conversion[2],
    "Diastolic Volume " + "(" + units[2] + ")": max_volume*conversion[2],
    "Systolic Diameter " + "(" + units[0] + ")": min_width*conversion[0],
    "Diastolic Diameter " + "(" + units[0] + ")": max_width*conversion[0],
    "Sistolic Height " + "(" + units[0] + ")": min_height*conversion[0],
    "Diastolic Height " + "(" + units[0] + ")": max_height*conversion[0]
    }

    row_labels = np.array(range(1,len(freq)))
    row_labels = np.append(row_labels,"Average")
    np.shape(freq)
    df0 = pd.DataFrame(data=all_data, index=row_labels)
    s0 = df0.style.set_table_styles(tables.cfg1,overwrite=False)
    return s0


# Show table with hemodynamics params
def show_resultstable(units,conversion,weight,avg_lst):

    # Define avg values
    avg_freq = avg_lst[0]
    avg_max_volume = avg_lst[1]
    avg_min_volume = avg_lst[2]
    avg_max_dia = avg_lst[3]
    avg_min_dia = avg_lst[4]

    # Calculate hemodynamics parameteres    
    avg_cardiac_out = str(format(avg_freq*60.0*(avg_max_volume - avg_min_volume)*conversion[3]/(weight*1e-6),"3.2f")).replace(".",",")
    beats_per_min = str(format(avg_freq*60,"3.2f")).replace(".",",")
    avg_ejection_volume = str(format((avg_max_volume - avg_min_volume)*conversion[2],"3.2f")).replace(".",",")
    avg_ejection_fraction = str(format((avg_max_volume - avg_min_volume)*100/avg_max_volume,"3.2f")).replace(".",",")
    avg_fractional_short = str(format((avg_max_dia-avg_min_dia)*100/avg_max_dia,"3.2f")).replace(".",",")

    print(avg_ejection_volume, avg_cardiac_out, avg_ejection_fraction )

    # 
    df1 = pd.DataFrame([beats_per_min + ' bpm',
                        avg_ejection_volume + ' ' + units[2],
                        avg_cardiac_out + ' ' + units[3] ,
                        avg_ejection_fraction + '%',
                        avg_fractional_short + '%'],
                    index = pd.Index(["BPM:","Ejection Volume:","Cardiac Output:","Ejection Fraction:","Fractional Shortening:"]),
                    columns =['Hemodynamics Parameters'] 
    )

    s1 = df1.style.set_table_styles(tables.config1,overwrite=False)
    return s1

# Show auxiliary plots
def show_auxplots(units,conversion,x,rev_vol,ellipse_vol,rev_area,area):
    
    time_axis = x
    
    rev_volume_axes = rev_vol*conversion[2]
    ellipse_volume_axes = ellipse_vol*conversion[2]
    rev_area_axes = rev_area*conversion[1]
    area_axis = area*conversion[1]


    fig = plt.figure()  # an empty figure with no Axes
    fig, ax = plt.subplots(figsize=(30,6))
    ax.plot(time_axis, rev_volume_axes,label = 'Revolution Solid')  # Plot some data on the axes.
    ax.plot(time_axis, ellipse_volume_axes, label = 'Ellipsoid')  # Plot some data on the axes.
    ax.set(xlabel= "Time (sec)", ylabel= "Volume" + "(" + units[2] + ")",
       title= "Revolution solid and ellipsoid Volume")
         
    ax.grid() 

    ax.legend()
    
    plt.show(fig)

    fig = plt.figure()  # an empty figure with no Axes
    fig, ax = plt.subplots(figsize=(30,6))
    ax.plot(time_axis, rev_area_axes,label = 'Area by contour integral')  # Plot some data on the axes.
    ax.plot(time_axis, area_axis, label = 'Area by OpenCV method')  # Plot some data on the axes.

    ax.set(xlabel= "Time (sec)", ylabel= "Area" + "(" + units[2] + ")",
       title= "Area of heart by integral and opencv")
         
    ax.grid() 

    ax.legend()
    
    plt.show(fig)

    return

def get_result_lst(min_list,max_list,time_axis,volume_axis,width_axis,height_axis):

    # Definição dos vetores que serão utilizados para armezenar os dados
    freq = np.array([])
    max_volume = np.array([])
    min_volume = np.array([])
    min_width = np.array([])
    max_width = np.array([])
    min_height = np.array([])
    max_height = np.array([])
    stroke_volume = np.array([])
    ejection_fraction = np.array([])

    # Creates lists with frequency of each beat and volume on each systole
    for i in range(len(min_list)):
  
        if i + 1 < len(min_list):
            t = time_axis[min_list[i+1]] - time_axis[min_list[i]]
            freq = np.append(freq,1/t)        
            #1e+9
            min_volume = np.append(min_volume,volume_axis[min_list[i]])

    # Creates lists with volume, diameter and height of heart on each diastole
    for i in range(len(max_list)):

        #1e+9
        max_volume = np.append(max_volume,volume_axis[max_list[i]])

        max_width = np.append(max_width, width_axis[max_list[i]])
        max_height = np.append(max_height, height_axis[max_list[i]])
       
    # Creates lists with diameter and height of heart on each systole
    for i in range(len(min_volume)):

        min_width = np.append(min_width, width_axis[min_list[i]])
        min_height = np.append(min_height, height_axis[min_list[i]])


    # get average values
    avg_freq = np.average(freq)
    avg_max_volume = np.average(max_volume)
    avg_min_volume = np.average(min_volume)
    avg_min_height = np.average(min_height)
    avg_min_width = np.average(min_width)
    avg_max_height = np.average(max_height)
    avg_max_width = np.average(max_width)
        
    return (freq,[min_volume,min_width,min_height],
                [max_volume,max_width,max_height],
                [avg_freq,avg_max_volume,avg_min_volume,avg_max_width,avg_min_width])


# Function that saves a data as csv file
def save_as_csv(time,ellipse_dia,manual_dia,scale):

    df = pd.DataFrame([time,
                        ellipse_dia*scale[0],
                        manual_dia*scale[0],
                        ],
                    index = pd.Index(list(range(len(time)))),
                    columns =['Time','Ellipse Diameter','Manual Ellipse diameter'] 
    )

    df.to_excel('saved_file2.xlsx')

    return

def save_as_csv2(time,ellipse_dia,manual_dia,scale,units):

    df = pd.DataFrame({'Time [s]':time,
                        'Ellipse diameter [' + units + ']':ellipse_dia*scale[0],
                        'Manual Ellipse diameter [' + units + ']':manual_dia*scale[0]}
                    #index = pd.Index(["BPM:","Ejection Volume:","Cardiac Output:","Ejection Fraction:","Fractional Shortening:"]),
                    #index = list(range(len(time))),
    )

    df.to_excel('saved_file.xlsx')

    return

    
# Show all results widgets 
def show_results(weight,time,minor_axis,major_axis,manual_minor,manual_major,
                area,rev_area,ellipse_area,
                ellipse_vol,rev_vol,const_vol,manual_vol,trgg):

    # Configure widgets
    menu_lenght = widgets.Dropdown(
        options=['μm', 'mm','cm'],
        value='μm',
        description='Lenght:',
    )

    menu_area = widgets.Dropdown(
        options=['μm²', 'mm²','cm²'],
        value='μm²',
        description='Area:',
    )

    menu_volume = widgets.Dropdown(
        options=['μm³', 'nl','μl', 'ml', 'Droplets'],
        value='nl',
        description='Volume:',
    )

    menu_cout = widgets.Dropdown(
        options=['ml/min/kg'],
        value='ml/min/kg',
        description='Cardiac Output:',
    )

    menu_ch_volume = widgets.Dropdown(
        options=[('Prolate Spheroid',1), ('Revolution Solid',2),
                ('Ellipse Major Axis Const',3), ('Manual Ellipse',4)],
        value= 1,
        description='Volume Calculation:',
    )

    menu_weight = widgets.BoundedFloatText(
        value=weight,
        min=0,
        max = 10000,
        step=0.1,
        description='Insect Weight',
    
        disabled=False
    )

    button = widgets.Button(
        description='Update Units',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
        icon='check' # (FontAwesome names without the `fa-` prefix)
    )

    check = widgets.Checkbox(
        value=False,
        description='Show Systole/Diastole',
        disabled=False,
        indent=True
    )

    # Stack widgets
    main_menu = widgets.VBox([menu_lenght,menu_area,menu_volume,menu_weight,
                              menu_cout,menu_ch_volume,check,button])
    display(main_menu)

    outt1 = widgets.Output()
    outt2 = widgets.Output()
    outt3 = widgets.Output()
    outt4 = widgets.Output()

    tab = widgets.Tab(children = [outt1, outt2, outt3, outt4])
    tab.set_title(0, 'Plots')
    tab.set_title(1, 'Charts')
    tab.set_title(2, 'Results')
    tab.set_title(3, 'Auxiliary Plots')
    display(tab)

    # Select area to calculate periods, from contour area (1) or ellipsis area (2)
    if trgg == 1:
        area_trg = np.array(area)
    elif trgg == 2:
        area_trg = np.array(ellipse_area)

    max_points, min_points = max_min_pts(area_trg)
    
    plot_y = [area,ellipse_area,minor_axis,major_axis,ellipse_vol]
    frq,min_list,max_list,avg_list = get_result_lst(min_points,max_points,time,ellipse_vol,minor_axis,major_axis)

    # Show results in separate tabs
    with outt1:
        plot_results(check.value,['μm^3','μm^2','μm'],[1,1,1e-9],time,plot_y,min_points,max_points,trgg)
    with outt2:
        display(show_maintable(['μm','μm^2','nl'],[1,1,1e-6],min_list,max_list,frq))
    with outt3:
        display(show_resultstable(["","",'nl','ml/min/kg'],[1,1,1e-6,1e-12],weight,avg_list))
    with outt4:
        show_auxplots(["_","_",'nl'],[1,1,1e-6],time,rev_vol,ellipse_vol,rev_area,area)

    # Define nested function to update results when button is clicked
    def on_button_clicked (a):

        multiplier = [1,1,1,1]
        units = [menu_lenght.value,menu_area.value,menu_volume.value,menu_cout.value]

        # Convert units of lenght
        if menu_lenght.value == 'mm':
            multiplier[0] = 1e-3
        elif menu_lenght.value == 'cm':
            multiplier[0] = 1e-4
        
        # Convert units of area
        if menu_area.value == 'mm²':
            multiplier[1] = 1e-6
        elif menu_area.value == 'cm²':
            multiplier[1] = 1e-8

        # Convert units of volume
        if menu_volume.value == 'μl':
            multiplier[2] = 1e-9
        elif menu_volume.value == 'ml':
            multiplier[2] = 1e-12
        elif menu_volume.value == 'Droplets':
            multiplier[2] = 50e-9    
        elif menu_volume.value == 'nl':
            multiplier[2] = 1e-6
        
        # Convert units of volume
        if menu_cout.value == 'ml/min/kg':
            multiplier[3] = 1e-12

        # Define which volume to use
        if menu_ch_volume.value == 1:
            volume = ellipse_vol
            major = major_axis
            minor = minor_axis
        elif menu_ch_volume.value == 2:
            volume = rev_vol
            major = major_axis
            minor = minor_axis
        elif menu_ch_volume.value == 3:
            volume = const_vol
            major = major_axis
            minor = minor_axis
        elif menu_ch_volume.value == 4:
            volume = manual_vol
            major = manual_major
            minor = manual_minor

        plot_y = [area,ellipse_area,minor_axis,major_axis,volume]
        frq,min_list,max_list,avg_list = get_result_lst(min_points,max_points,time,volume,minor,major)
        
        # Show results in separate tabs
        with outt1:
            clear_output()
            plot_results(check.value,units,multiplier,time,plot_y,min_points,max_points,trgg)
        with outt2:
            clear_output()
            display(show_maintable(units,multiplier,min_list,max_list,frq))
        with outt3:
            clear_output()
            display(show_resultstable(units,multiplier,menu_weight.value,avg_list))
        with outt4:
            clear_output()
            show_auxplots(units,multiplier,time,rev_vol,ellipse_vol,rev_area,area)
        
        save_as_csv2(time,minor_axis,manual_minor,multiplier,'um')


        return
    
    button.on_click(on_button_clicked)  # Define callback function