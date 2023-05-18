import pandas as pd
from matplotlib import pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
import tables
import numpy as np
from scipy.signal import argrelextrema
import os
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
                if temp[j] > max_area :
                    max_area = temp[j]
                    area_index = min_list[i] + j

            max_list.append(area_index)   

    return max_list, min_list


# Function that creates adn draw plots 
def plot_img2(x_axes, y_axes, x_title, y_title, plot_title, h,w, p_enable,minimum, maximum,units,save,path):

    if save:
        plt.ioff()
    else:
        plt.ion()

    plt.style.use("fivethirtyeight")
   
    #fig = plt.figure()  # an empty figure with no Axes
    fig, ax = plt.subplots(figsize=(h, w))
    ax.plot(x_axes, y_axes)  # Plot some data on the axes.

    t_font = {'fontsize': '25',
              'weight': 'bold'}

    # Set title to Axes
    fig.suptitle(t=plot_title, **t_font)
    plt.xlabel(x_title, fontsize=17)
    plt.ylabel(y_title + "$(" + units + ")$", fontsize=17)

    
    
    #ax.set(xlabel= x_title, ylabel= y_title + "(" + units + ")",title= plot_title)
         
    #ax.grid() # Put grids on plot

    # If enabled, put systole and dyastole points on plot
    if p_enable:
        ax.plot(x_axes[minimum],  y_axes[minimum], 'o',color='orange', markersize = 7, label = 'Systole')
        ax.plot(x_axes[maximum],  y_axes[maximum], 'o',color='green', markersize = 7, label = 'Diastole')
        ax.legend()

    
    if save:
        plt.savefig(path+plot_title+'.png',dpi = fig.dpi)
        plt.close(fig)
    else:
        plt.show(fig)   # Show plot


# Function that plot all results 
def plot_results(point_enable, units, conversion, px_convert, x, y,
                 min_list, max_list, trgg, save, folder):
    time_axis = x
    area_axis = y[0]*px_convert**2
    ellipse_area_axis = y[1]
    width_axis = y[2]*px_convert
    height_axis = y[3]*px_convert**2
    volume_axis = y[4]*px_convert**3

    if trgg == 1:
        plot_img2(time_axis, area_axis*conversion[1],
                  "Time (s)", "Area ", "Heart's area",
                  30, 6, point_enable, min_list, max_list, units[1], save, folder)
    elif trgg == 2:
        plot_img2(time_axis, ellipse_area_axis*conversion[1],
                  "Time (s)", "Area ", "Ellipsis area",
                  30, 6, point_enable, min_list, max_list, units[1], save, folder)
    
    plot_img2(time_axis, width_axis*conversion[0], 
              "Time (s)", "Diameter ", "Minor Axis ellipsoid", 
              30, 6, point_enable, min_list, max_list, units[0], save, folder)
    
    plot_img2(time_axis, height_axis*conversion[0],
              "Time (s)", "Major Axis ", "Major Axis ellipsoid",
              30, 6, point_enable, min_list, max_list, units[0], save, folder)
    
    plot_img2(time_axis, volume_axis*conversion[2],
              "Time (s)", "Volume ", "Insect's heart volume",
              30, 6, point_enable, min_list, max_list, units[2], save, folder)


# Show table with all data
def show_maintable(units, conversion, px_convert, min_lst, max_lst, freq):

    min_volume = min_lst[0]
    min_width = min_lst[1]
    min_height = min_lst[2]

    max_volume = max_lst[0]
    max_width = max_lst[1]
    max_height = max_lst[2]

    min_volume = np.append(min_volume, np.average(min_volume))
    min_width = np.append(min_width, np.average(min_width))
    min_height = np.append(min_height, np.average(min_height))

    max_volume = np.append(max_volume, np.average(max_volume))
    max_width = np.append(max_width, np.average(max_width))
    max_height = np.append(max_height, np.average(max_height))

    freq = np.append(freq, np.average(freq))

    all_data = {"Frequency " + "(" + "hz" + ")": freq,
                "Systolic Volume " + "(" + units[2] + ")": min_volume*conversion[2]*px_convert,
                "Diastolic Volume " + "(" + units[2] + ")": max_volume*conversion[2]*px_convert,
                "Systolic Diameter " + "(" + units[0] + ")": min_width*conversion[0]*px_convert,
                "Diastolic Diameter " + "(" + units[0] + ")": max_width*conversion[0]*px_convert,
                "Sistolic Height " + "(" + units[0] + ")": min_height*conversion[0]*px_convert,
                "Diastolic Height " + "(" + units[0] + ")": max_height*conversion[0]*px_convert
                }

    row_labels = np.array(range(1, len(freq)))
    row_labels = np.append(row_labels, "Average")
    df0 = pd.DataFrame(data=all_data, index=row_labels)
    df0 = df0.round(2)
    s0 = df0.style.set_table_styles(tables.cfg1, overwrite=False)
    return s0


# Show table with hemodynamics params
def show_resultstable(units, conversion, px_convert, weight, avg_lst):

    # Define avg values
    avg_freq = avg_lst[0]
    avg_max_volume = avg_lst[1]*(px_convert)**3
    avg_min_volume = avg_lst[2]*(px_convert)**3
    avg_max_dia = avg_lst[3]*(px_convert)
    avg_min_dia = avg_lst[4]*(px_convert)

    # Calculate hemodynamics parameteres    
    avg_cardiac_out = str(format(avg_freq*60.0*(avg_max_volume - avg_min_volume)*conversion[3]/(weight*1e-6),"3.2f")).replace(".",",")
    beats_per_min = str(format(avg_freq*60,"3.2f")).replace(".",",")
    avg_ejection_volume = str(format((avg_max_volume - avg_min_volume)*conversion[2],"3.2f")).replace(".",",")
    avg_ejection_fraction = str(format((avg_max_volume - avg_min_volume)*100/avg_max_volume,"3.2f")).replace(".",",")
    avg_fractional_short = str(format((avg_max_dia-avg_min_dia)*100/avg_max_dia,"3.2f")).replace(".",",")

    # 
    df1 = pd.DataFrame([beats_per_min + ' bpm',
                        avg_ejection_volume + ' ' + units[2],
                        avg_cardiac_out + ' ' + units[3] ,
                        avg_ejection_fraction + '%',
                        avg_fractional_short + '%'],
                    index = pd.Index(["BPM:","Ejection Volume:","Cardiac Output:","Ejection Fraction:","Fractional Shortening:"]),
                    columns =['Hemodynamics Parameters'] 
    )

    df2 = pd.DataFrame([[avg_ejection_volume,avg_cardiac_out,avg_ejection_fraction]])
    display(df2)

    s1 = df1.style.set_table_styles(tables.config1,overwrite=False)
    return s1


# Show auxiliary plots
def show_auxplots(units, conversion, px_convert, x, rev_vol, ellipse_vol, rev_area, area):
    
    time_axis = x
    
    rev_volume_axes = rev_vol*conversion[2]*(px_convert)**3
    ellipse_volume_axes = ellipse_vol*conversion[2]*(px_convert)**3
    rev_area_axes = rev_area*conversion[1]*(px_convert)**2
    area_axis = area*conversion[1]*(px_convert)**2


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


# Function that calculate an array containing a multiplier constant
# that convert the units
def update_multiplier(mult_array,units_array):

    # Convert units of lenght
    if units_array[0] == 'mm':
        mult_array[0] = 1e-3
    elif units_array[0] == 'cm':
        mult_array[0] = 1e-4
        
    # Convert units of area
    if units_array[1] == 'mm²':
        mult_array[1] = 1e-6
    elif units_array[1] == 'cm²':
        mult_array[1] = 1e-8

    # Convert units of volume
    if units_array[2] == 'μl':
        mult_array[2] = 1e-9
    elif units_array[2] == 'ml':
        mult_array[2] = 1e-12
    elif units_array[2] == 'Droplets':
        mult_array[2] = 50e-9    
    elif units_array[2] == 'nl':
        mult_array[2] = 1e-6
        
    # Convert units of volume
    if units_array[3] == 'ml/min/kg':
        mult_array[3] = 1e-12

    return mult_array


# Function that saves a data as csv file
def save_as_csv(location,time,scale,units,raw_data,rev_vol,min_lst,max_lst,frq,weight,avg_lst):

    # Creates dataframe with raw data
    raw_df = pd.DataFrame({'Time [s]':time,
                            'Heart area ['+units[1]+']':raw_data[0]*scale[1],
                            'Minor Axis Ellipse['+units[0]+']':raw_data[2]*scale[0],
                            'Major Axis Ellipse['+units[0]+']':raw_data[3]*scale[0],
                            'Ellipse Volume['+units[2]+']':raw_data[4]*scale[2],
                            'Revolution Solid Volume['+units[2]+']':rev_vol*scale[2]}
    )

    # Creates dataframe with average data of each heart beat
    avg_df = show_maintable(units,scale,min_lst,max_lst,frq)

    # Creates dataframe with all hemodynamic parameters
    result_df = show_resultstable(units,scale,weight,avg_lst)

    # create a excel writer object
    with pd.ExcelWriter(location + 'saved_file.xlsx') as writer:
        raw_df.to_excel(writer, sheet_name="Raw Data", index=False)
        avg_df.to_excel(writer, sheet_name="Average Values Per Beat", index=True)
        result_df.to_excel(writer, sheet_name="Parameters", index=True)

    return

class AllWidgets:

    def __init__(self, weight, loading):
        # Configure widgets
        self.menu_lenght = widgets.Dropdown(
            options=['μm', 'mm', 'cm'],
            value='μm',
            description='Lenght:',
        )

        self.menu_area = widgets.Dropdown(
            options=['μm²', 'mm²', 'cm²'],
            value='μm²',
            description='Area:',
        )

        self.menu_volume = widgets.Dropdown(
            options=['μm³', 'nl', 'μl', 'ml', 'Droplets'],
            value='nl',
            description='Volume:',
        )

        menu_units = widgets.HBox([self.menu_lenght, 
                                  self.menu_area, self.menu_volume])

        self.menu_cout = widgets.Dropdown(
            options=['ml/min/kg'],
            value='ml/min/kg',
            description='Cardiac Output:',
        )

        self.menu_ch_volume = widgets.Dropdown(
            options=[('Prolate Spheroid', 1), ('Revolution Solid', 2),
                     ('Ellipse Major Axis Const', 3), ('Manual Ellipse', 4)],
            value=1,
            description='Volume Calculation:',
        )

        self.menu_weight = widgets.BoundedFloatText(
            value=weight,
            min=0,
            max=10000,
            step=0.1,
            description='Insect Weight',

            disabled=False
        )

        self.button = widgets.Button(
            description='Update Units',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            icon='rotate-right'  # (FontAwesome names without the `fa-` prefix)
        )

        self.exp_button = widgets.Button(
            description='Export Results',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            icon='download'  # (FontAwesome names without the `fa-` prefix)
        )

        self.check = widgets.Checkbox(
            value=False,
            description='Show Systole/Diastole',
            disabled=False,
            indent=True
        )

        # Stack widgets
        all_button = widgets.HBox([self.button, self.exp_button])
        self.main_menu = widgets.VBox([menu_units, self.menu_weight,
                                      self.menu_cout, self.menu_ch_volume, self.check, all_button])
        
        self.main_out = widgets.Output()

        self.outt1 = widgets.Output()
        self.outt2 = widgets.Output()
        self.outt3 = widgets.Output()
        self.outt4 = widgets.Output()

        self.tab = widgets.Tab(
            children=[self.outt1, self.outt2, self.outt3, self.outt4])
        self.tab.set_title(0, 'Plots')
        self.tab.set_title(1, 'Charts')
        self.tab.set_title(2, 'Results')        
        self.tab.set_title(3, 'Auxiliary Plots')

        self.displayed = False

        self.loading_bar = loading

        # Define callback function for "update units" button
        self.button.on_click(self.on_button_clicked)
    
    def save_records(self, parameter):
        
        self.ellipse = parameter[2]
        self.rev_solid = parameter[4]
        self.manu_ellipse = parameter[3]

        self.area = parameter[6]
        self.time = parameter[1]

        self.weight = parameter[0]
        self.px_convert = parameter[7]
        self.trgg = parameter[8]
        self.const_vol = parameter[5]

        # Define nested function to update results when button is clicked
    def on_button_clicked(self, a):

        multiplier = [1, 1, 1, 1]
        units = [self.menu_lenght.value, self.menu_area.value,
                 self.menu_volume.value, self.menu_cout.value]
        multiplier = update_multiplier(multiplier, units)
        
        # Select area to calculate periods, from contour area (1) or ellipsis area (2)
        area_trg = np.array(self.area)
        max_points, min_points = max_min_pts(area_trg)
        

        # Define which volume to use
        if self.menu_ch_volume.value == 1:
            volume = self.ellipse.volume_record
            major = self.ellipse.height_record
            minor = self.ellipse.width_record
        elif self.menu_ch_volume.value == 2:
            volume = self.rev_solid.volume_record
            major = self.ellipse.height_record
            minor = self.ellipse.width_record
        elif self.menu_ch_volume.value == 3:
            volume = self.const_vol
            major = self.ellipse.height_record
            minor = self.ellipse.width_record
        elif self.menu_ch_volume.value == 4:
            volume = self.manu_ellipse.volume_record
            major = self.manu_ellipse.height_record
            minor = self.manu_ellipse.width_record

        plot_y = [self.area, self.ellipse.area_record, minor, major, volume]
        frq, min_list, max_list, avg_list = get_result_lst(
            min_points, max_points, self.time, volume, minor, major)

        # Show results in separate tabs
        with self.outt1:
            clear_output(True)
            plot_results(self.check.value, units, multiplier, self.px_convert,
                         self.time, plot_y, min_points, max_points, self.trgg, False, 0)
        with self.outt2:
            clear_output(True)
            display(show_maintable(units, multiplier, self.px_convert,
                                   min_list, max_list, frq))
        with self.outt3:
            clear_output(True)
            display(show_resultstable(units, multiplier, self.px_convert,
                                      self.weight, avg_list))
        with self.outt4:
            clear_output(True)
            show_auxplots(units, multiplier, self.px_convert, self.time,
                          self.rev_solid.volume_record, self.ellipse.volume_record, self.rev_solid.area_record, self.area)

        return
        

# Show all results widgets 
def show_results(weight, time,
                 ellipse, manu_ellipse, rev_solid, const_vol,
                 area, px_convert, trgg, path, logs, AllWidgets):
    

    if AllWidgets.displayed == False:
        with AllWidgets.main_out:
            display(AllWidgets.main_menu)
            display(AllWidgets.tab)
        display(AllWidgets.main_out)

    # Select area to calculate periods, from contour area (1) or ellipsis area (2)
    area_trg = np.array(area)
    max_points, min_points = max_min_pts(area_trg)
    
    plot_y = [area, ellipse.area_record, ellipse.width_record,
              ellipse.height_record, ellipse.volume_record]
    frq, min_list, max_list, avg_list = get_result_lst(
        min_points, max_points, time, ellipse.volume_record, ellipse.width_record, ellipse.height_record)
    
    multiplier = [1, 1, 1, 1]
    units = [AllWidgets.menu_lenght.value, AllWidgets.menu_area.value,
             AllWidgets.menu_volume.value, AllWidgets.menu_cout.value]
    multiplier = update_multiplier(multiplier, units)

    # Show results in separate tabs
    with AllWidgets.outt1:
        clear_output(True)
        plot_results(AllWidgets.check.value, units, multiplier, px_convert,
                     time,plot_y, min_points, max_points, trgg, False, 0)
    with AllWidgets.outt2:
        clear_output(True)
        display(show_maintable(units, multiplier, px_convert,
                               min_list, max_list, frq))
    with AllWidgets.outt3:
        clear_output(True)
        display(show_resultstable(units, multiplier, px_convert,
                                  weight, avg_list))
    with AllWidgets.outt4:
        clear_output(True)
        show_auxplots(units, multiplier, px_convert, time,
                      rev_solid.volume_record, ellipse.volume_record, rev_solid.area_record, area)
        
    
    def export_results(a):

        # Array that is contains the multipliers that convert scale of units in the order
        # lenght, area, volume and cardiac output
        multiplier = [1,1,1,1]
        units = [AllWidgets.menu_lenght.value,AllWidgets.menu_area.value,AllWidgets.menu_volume.value,AllWidgets.menu_cout.value]
        multiplier = update_multiplier(multiplier,units)
        
        today = str(date.today())
        recn = 0

        while (1):
            folder = path + today + '_' + str(recn) + '/'
            folder_exists = exists(folder)
            if folder_exists:
                recn =  recn + 1
            else:
                os.mkdir(folder)
                break

        folder = path + today + '_' + str(recn)+ '/'

        # Create CSV with raw data, average values per beat and 
        # final hemodynamics parameters
        save_as_csv(folder,time,multiplier,units,plot_y,
                    rev_vol,min_list,max_list,frq,weight,avg_list)
        
        # Save all displayed plots on the same folder of the CSV file
        plot_results(AllWidgets.check.value, units, multiplier, time,
                     plot_y, min_points, max_points, trgg, True, folder)

        with logs:
            print("Saved all results at: ", folder)

        return


    # Define callback function for "Export results" button
    AllWidgets.exp_button.on_click(export_results)

    return