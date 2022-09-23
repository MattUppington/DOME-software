# This code is provided to guide users through the process of calibration to find a mathematical
# transformation that describes the mapping of camera pixels to projector pixels in the DOME. The
# DOME (Dynamic Optical Micro Environment) was developed by Ana Rubio Denniss. This code requires
# the "DOME_caibration_projector.py" file to be run in parallel on the Raspberry Pi 0 connected to
# the DOME projector. To calibrate the DOME, run this file and follow the on screen instructions.
# #################################################################################################
# Authors = Matthew Uppington <mu15531@bristol.ac.uk>
# Affiliation = Farscope CDT, University of Bristol, University of West England
# #################################################################################################
# This work is licensed under a Creative Commons Attribution 4.0 International License.

import DOME_communication as DOMEcomm
import DOME_imaging_utilities as DOMEutil

import numpy as np
import cv2
import json
import time


class SettingsFileFormatError(Exception):
    '''
    Exception class for handling errors raised when calibration settings cannot be read from a
    file. The file should specify values for "brightness", "threshold", "region size" and
    "scan increment".
    '''
    
    def __init__(self, key : str):
        '''
        Sets up the parent class for exceptions and provides an error message.
        ---
        Parameters
            key : str
                The settings dictionary key that is missing from the json file.
        '''
        self.error_message = f'Format of calibration settings file is not recognised.\n Setting ' \
                             f'key "{key}" not specified.'
        super().__init__(self.error_message)
    
    def print_error_message(self):
        '''
        Prints the error message without interupting execution.
        '''
        print(self.error_message)


def load_settings(file_name : str, keys : list):
    '''
    Reads calibration setting values from a json file.
    ---
    Parameters
        file_name : str
            Name of json file to read settings from.
    ---
    Outputs
        stored_settings : dict
            Dictionary of calibration settings.
    '''
    with open(file_name, 'r') as file:
        stored_settings = json.load(file)
    for setting in keys:
        # Check that the file contains all of the expected parameters.
        if not setting in stored_settings.keys():
            raise SettingsFileFormatError(setting)
    return stored_settings

def custom_input(message : str, step : int):
    '''
    Augmented version of input command to facilitate adjustments to variable representing
    progression through calibration procedure {step}: inputting "back" reduces value by 1;
    "restart" resets value back to 0.
    ---
    Parameters
        message : str
            Message to display when user input is requested.
        step : int
            Current stage of calibration procedure.
    ---
    Outputs
        user_input_segments : list[str,...]
            User input split by space character " ".
        new_step
            New stage of calibration procedure.
    '''
    user_input = input(message)
    user_input_segments = ['']
    if user_input == 'skip':
        new_step = 7
    elif user_input == 'next':
        new_step = step + 1
    elif user_input == 'back':
        new_step = max(step - 1, 1)
    elif user_input == 'restart':
        new_step = 1
    elif user_input == 'exit':
        new_step = 0
    else:
        user_input_segments = user_input.split(' ')
        new_step = step
    return user_input_segments, new_step

def overlay_grid(image : np.ndarray, spacing : int, thickness : int, colour : tuple):
    '''
    Draw a grid pattern over an image.
    ---
    Parameters
        image : np.ndarray
            Original image.
        spacing : int
            Separation between the start of grid lines in number of pixels.
        thickness : int
            Thickness of grid lines in number of pixels.
        colour : tuple[int, int, int]
            Colour of grid lines, specified in BGR order.
    ---
    Outputs
        image_with_grid : np.ndarray
            Original image with grid drawn over it.
    '''
    image_with_grid = image.copy()
    for c in range(0, len(colour)):
        for s in range(1, int(np.ceil(image.shape[0] / spacing))):
            image_with_grid[s * spacing:s * spacing + thickness, :, c] = colour[c]
        for s in range(1, int(np.ceil(image.shape[1] / spacing))):
            image_with_grid[:, s * spacing:s * spacing + thickness, c] = colour[c]
    return image_with_grid

def pixelate(image : np.ndarray, pixel_dims : list):
    '''
    Pixelate a gray-scale image by averaging pixel values over tiled areas.
    ---
    Parameters
        image : np.ndarray
            Grey-scale image to be pixelated.
        pixel_dims : list[int, int]
            A list of two integers containing the dimensions of tiled areas over which pixel values
            will be averaged.
    ---
    Outputs
        pixelated_image : np.ndarray
            A pixelated version of the input image.
        reduced_image : np.ndarry
            A version of the pixelated image with the dimensions reduced by a factor equal to the
            specified pixel dimensions.
    '''
    pixelated_image = np.zeros(image.shape, dtype=np.uint8)
    reduced_image = np.zeros((int(np.ceil(image.shape[0] / pixel_dims[0])),
                              int(np.ceil(image.shape[1] / pixel_dims[1]))))###################
    for x_block in range(0, reduced_image.shape[0]):
        limits_x = [x_block * pixel_dims[0],
                    min([(x_block + 1) * pixel_dims[0], image.shape[0]])]
        for y_block in range(0, reduced_image.shape[1]):
            limits_y = [y_block * pixel_dims[1],
                        min([(y_block + 1) * pixel_dims[1], image.shape[1]])]
            pixel = image[limits_x[0]:limits_x[1], limits_y[0]:limits_y[1]]
            pixel_value = int(np.sum(pixel) / (np.prod(pixel_dims)))
            pixelated_image[limits_x[0]:limits_x[1], limits_y[0]:limits_y[1]] = pixel_value
            reduced_image[x_block, y_block] = pixel_value
    return pixelated_image, reduced_image

def find_grid_corners(binary_image : np.ndarray, margin : float, pixel_dims : list):
    '''
    Find the four corner locations of a square which is centered on the middle of the largest
    region of high intensity in the image and is not closer than margin (percent) to the
    surrounding region with 0 intensity.
    ---
    Parameters
        binary_image : np.ndarray
            Image containing a region of high intensity pixels surrounded by pixels with 0
            intensity.
        margin : float
            Value between 0 - 0.5 specifying the relative distance between the square corners and
            the boundary of the region of high intensity.
        pixel_dim : list[int, int]
            Size of pixelation to apply.
    ---
    Outputs
        corners : np.ndarray
            4x2 matrix of corner coordinates.
        reduced_frontier_scaled : np.ndarray
            Inverse intensity map of proximity to black or border pixels.
    '''
    # Start with a lower resolution image for time efficiency.
    pixelated_image, reduced_image = pixelate(binary_image, pixel_dims)
    # Map the pixel values in the reduced image to 1 if on a border of the image, or the pixel...
    # ...intensity is 0, and map to 0 otherwise.
    reduced_frontier = np.where(reduced_image == 0, 1, 0)
    reduced_frontier[[0, -1], :] = 1
    reduced_frontier[:, [0, -1]] = 1
    # Propagate through the frontier to record the hamiltonian distance between each pixel and...
    # ...the nearest pixel in the reduced image that either has 0 intensity or is at a border.
    for k in range(1, min(reduced_image.shape)):
        if np.sum(np.where(reduced_frontier == 0, 1, 0)) == 0:
            break
        neighbours = np.zeros(reduced_frontier.shape)
        neighbours[1:, :] = neighbours[1:, :] + reduced_frontier[:-1, :]
        neighbours[:-1, :] = neighbours[:-1, :] + reduced_frontier[1:, :]
        neighbours[:, :-1] = neighbours[:, :-1] + reduced_frontier[:, 1:]
        neighbours[:, 1:] = neighbours[:, 1:] + reduced_frontier[:, :-1]
        reduced_frontier = np.where((reduced_frontier == 0) & (neighbours > 0),
                                    k + 1, reduced_frontier)
    # Normalise the frontier on to the scale 0 - 255.
    reduced_frontier_scaled = (reduced_frontier - 1) * 255 / np.max(reduced_frontier - 1)
    reduced_rows = np.tile(np.array([range(0, reduced_frontier_scaled.shape[0])]).T,
                           (1, reduced_frontier_scaled.shape[1]))
    reduced_cols = np.tile(np.array([range(0, reduced_frontier_scaled.shape[1])]),
                           (reduced_frontier_scaled.shape[0], 1))
    # Find the furthest pixel from black or border pixels by taking a weighted average.
    reduced_center = np.array([np.sum(reduced_rows * reduced_frontier_scaled) /
                               reduced_frontier_scaled.sum(),
                               np.sum(reduced_cols * reduced_frontier_scaled) /
                               reduced_frontier_scaled.sum()])
    reduced_center_rounded = np.floor(reduced_center).astype(int)
    reduced_diags = np.array([0, 1])
    # Search diagonally from the centroid pixel until specified margin is reached.
    for s in range(0, 2 * min(reduced_image.shape)):
        square = reduced_frontier_scaled[reduced_center_rounded[0] - reduced_diags[0]:
                                         reduced_center_rounded[0] + reduced_diags[1],
                                         reduced_center_rounded[1] - reduced_diags[0]:
                                         reduced_center_rounded[1] + reduced_diags[1]]
        if square.min() < margin * 255:
            break
        else:
            reduced_diags[s % 2] += 1
    # Calculate position of centroid pixel and corners in original image.
    center = np.floor(reduced_center * np.array(pixel_dims)).astype(int)
    side_lengths = (reduced_diags.sum() + 1) * np.array(pixel_dims)
    corners = [[int(center[0] + 0.5 * side_lengths[0]), int(center[1] - 0.5 * side_lengths[1])],
               [int(center[0] + 0.5 * side_lengths[0]), int(center[1] + 0.5 * side_lengths[1])],
               [int(center[0] - 0.5 * side_lengths[0]), int(center[1] + 0.5 * side_lengths[1])],
               [int(center[0] - 0.5 * side_lengths[0]), int(center[1] - 0.5 * side_lengths[1])]]
    return corners, reduced_frontier_scaled.astype(np.uint8)

def measure_intensities(image : np.ndarray, points : np.ndarray, scan_range : int):
    '''
    Extracts the total, summed intensities of pixels around a set of points.
    ---
    Parameters
        image : np.ndarray
            Picture from which pixel intensities will be measured.
        points : np.ndarray
            Nx2 array containing N coordinates around which pixel intensities will be summed.
        scan_range : int
            Maximum distance in each axis of measured pixels from the specified coordinates.
    ---
    Outputs
        total_intensities : np.ndarray
            Nx0 array of total measured pixel intensities for each specified coordinate.
    '''
    total_intensities = np.zeros(points.shape[0])
    for p in range(0, points.shape[0]):
        square = image[points[p][0] - scan_range:points[p][0] + scan_range,
                       points[p][1] - scan_range:points[p][1] + scan_range]
        total_intensities[p] = square.sum()
    return total_intensities

def get_bright_lines(intensities : list):
    '''
    Identifies indexes with globally maximal recorded intensities and returns the approximate
    locations of the sampling regions.
    ---
    Parameters
        intensities : list[np.ndarray, np.ndarray]
            List containing two arrays of shapes NxA / NxB respectively, where N is the number
            of sampled regions and A / B is the number of scanning rows / columns.
    ---
    Outputs
        bright_lines : 
    '''
#     bright_lines= [[[] for _ in range(0, intensities[d].shape[0])]
#                    for d in range(0, len(intensities))]
    bright_lines = -1 * np.ones((intensities[0].shape[0], 2))
#     envelope = np.array([-1, 0, 1])
    for d in range(0, len(intensities)):
        #sorted_intensities = np.sort(intensities[d], 1)
        for c in range(0, intensities[d].shape[0]):
            dir_corner_ints = intensities[d][c, :]
            envelope_totals = dir_corner_ints[:-2] + dir_corner_ints[1:-1] + dir_corner_ints[2:]
            main_line = np.argmax(envelope_totals)
            bright_lines[c, d] = (((main_line - 1) * dir_corner_ints[main_line - 1] +
                                   main_line * dir_corner_ints[main_line] +
                                   (main_line + 1) * dir_corner_ints[main_line + 1]) /
                                  np.sum(dir_corner_ints[main_line - 1:main_line + 2]))
#             #set threshold as twice the median intensity value, add one to set above 0
#             threshold = 2 * np.sort(dir_corner_ints)[int(len(dir_corner_ints) / 2)] + 1#########
#             num_checks = np.sum((dir_corner_ints >= threshold).astype(int))############
#             lines_checked = []
#             for _ in range(0, num_checks):
#                 max_index = np.argsort(dir_corner_ints)[-1]###########
#                 if len(set.intersection(set(max_index + envelope),
#                                         set(lines_checked))) == 0:
#                     bright_lines[d][c].append(max_index)
#                 lines_checked.append(max_index)
#                 dir_corner_ints[max_index] = 0
#                 # For simplicity, only output the single brightest line
#                 break
    return bright_lines

def main(margin : float, pixelation : list, camera_grid_spacing : int,
         camera_grid_thickness : int, camera_settings=None, gpio_light=None):
    projector_dims = (480, 854, 3)
    settings = {'brightness': None,
                'threshold': None,
                'margin': margin,
                'pixelation': pixelation,
                'corners': None,
                'spacing cam': camera_grid_spacing,
                'thickness cam': camera_grid_thickness,
                'spacing proj': None,
                'thickness proj': None,
                'magnification': None,
                'increment': None}
    camera_mode = 'default'
    response = None
    bright_lines = None
    with DOMEcomm.NetworkNode() as dome_pi4node, \
            DOMEutil.CameraManager() as dome_camera, \
            DOMEutil.PinManager() as dome_gpio:
        try:
            if not camera_settings is None:
                camera_mode = 'custom'
                dome_camera.store_settings(camera_mode, camera_settings)
            if not gpio_light is None:
                dome_gpio.add_pin_label('light source', gpio_light)
                dome_gpio.toggle('light source', 'on')
            print('On the projector Pi run "DOME_calibration_projector.py" and wait for a black ' \
                  'screen to appear (this may take several seconds). Once a black screen is ' \
                  'shown, click anywhere on the black screen, then press any key (such as ALT).')
            dome_pi4node.accept_connection()
            print(f'Welcome to the DOME calibration set up procedure.\nAt any point in the ' \
                  f'calibration, if all requisite parameters have been specified, input "skip" ' \
                  f'to begin scanning. Alternatively, enter "next" to proceed to the next step, ' \
                  f'"back" to return to the previous step, "restart" to begin the calibration ' \
                  'from the start, or "exit" to end the program.')
            threshold_picture = None
            loaded_file = None
            saved_file = None
            step = 1
            while 1 <= step <= 8:
#                 if step == 0:
#                     break
                duration = 5
                while step == 1:
                    message = f'--- STEP 1 ---\nTo begin, we will focus the camera by adjusting ' \
                              f'the height of the sample stage on the DOME. Input the number of ' \
                              f'seconds for which to display a live feed of the camera frame as ' \
                              f'an integer. By turning the lead screw on the DOME, move the ' \
                              f'sample stage up or down until the image comes in to focus. Once ' \
                              f'this is done, input "next" to continue. Henceforth, it is ' \
                              f'crucial that the camera, projector, sample stage and any lenses ' \
                              f'maintain their relative positions. Any change to the physical ' \
                              f'setup of the DOME may invalidate the calibration.\n--- Duration ' \
                              f'= {duration} ; integer (positive, non-zero)\n'
                    user_args, step = custom_input(message, step)
                    if len(user_args) == 1 and len(user_args[0]) != 0:
                        try:
                            duration = int(user_args[0])
                        except ValueError:
                            print('Please input an integer.')
                            continue
                    elif step != 1:
                        continue
                    dome_pi4node.transmit('all' + 3 * ' 128')
                    response = dome_pi4node.receive()
                    time.sleep(1)
                    dome_camera.show_live_feed(duration)
                    dome_pi4node.transmit('all' + 3 * ' 0')
                    response = dome_pi4node.receive()
                if step != 0:
                    loaded_file = None
                while step == 2:
                    message = f'--- STEP 2 ---\nTo proceed with manually defining settings for ' \
                              f'calibration, type "next".\nTo load previously saved calibration ' \
                              f'settings, type "load", followed by the file name (separated by ' \
                              f'a space character).\nCurrent file loaded = {loaded_file}\n'
                    user_args, step = custom_input(message, step)
                    if not user_args[0] == 'load' or len(user_args) != 2:
                        continue
                    try:
                        settings = load_settings(user_args[1], list(settings.keys()))
                    except FileNotFoundError:
                        print(f'No such file: {user_args[1]}')
                    except SettingsFileFormatError as error:
                        error.print_error_message()
                    else:
                        loaded_file = user_args[1]
#                         step = 5
                while step == 3:
                    message = f'--- STEP 3 ---\nFirst, we will identify the illuminatible area ' \
                              f'of the camera frame. Input a value for the BRIGHTNESS of the ' \
                              f'projector to illuminate the sample area (start with a ' \
                              f'low value, then increase until the illuminatible region is ' \
                              f'clearly visible). Also, input a value for the intensity ' \
                              f'THRESHOLD to use when detecting the illuminatible space (start ' \
                              f'with a high value then decrease until only the illuminatible ' \
                              f'region is detected). Separate these two values with a space ' \
                              f'character. Close the two windows that appear displaying the raw ' \
                              f'and processed images to try a different set of values. Once an ' \
                              f'appropriate set of values have been found, close the windows ' \
                              f'and enter "next" to continue.\n--- BRIGHTNESS = ' \
                              f'{settings["brightness"]} ; integer (0 - 255)\n--- THRESHOLD = ' \
                              f'{settings["threshold"]} ; integer (0 - 255)\n'
                    user_args, step = custom_input(message, step)
                    if len(user_args) != 2:
                        continue
                    try:
                        [bright, thresh] = [min(max(0, int(arg)), 255) for arg in user_args]
                    except ValueError:
                        print('Please specify integer values in the range 0 - 255, separated ' \
                              'with a space character.')
                    else:
                        dome_pi4node.transmit(f'all' + 3 * f' {bright}')
                        response = dome_pi4node.receive()
                        time.sleep(1)
                        picture = dome_camera.capture_image(camera_mode)
                        gray_picture = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
                        blur_value = 10
                        smoothed_gray_picture = cv2.blur(gray_picture, (blur_value, blur_value))
                        ret, threshold_picture = cv2.threshold(smoothed_gray_picture,
                                                               thresh, 255, 0)
                        cv2.imshow(f'Camera frame, brightness used: {bright}', picture)
                        cv2.imshow(f'Threshold applied: {thresh}', threshold_picture)
                        cv2.waitKey(0)
                        settings['brightness'] = bright
                        settings['threshold'] = thresh
                if not threshold_picture is None and step != 0:
                    corners_list, reduced_map = find_grid_corners(threshold_picture, settings['margin'],
                                                                  settings['pixelation'])
                    settings['corners'] = corners_list
                    scan_region_picture = threshold_picture.copy()
                    for corner in corners_list:
                        scan_region_picture[corner[0] - 5:corner[0] + 5,
                                            corner[1] - 5:corner[1] + 5] = 128
                        
                    cv2.imshow('Scan regions', scan_region_picture)
                    cv2.imshow('reduced', reduced_map)
                    cv2.waitKey(0)
                while step == 4:
                    # Check that the brightness and threshold values have been set.
                    if settings['brightness'] is None or settings['threshold'] is None:
                        print('Values for BRIGHTNESS and/or THRESHOLD have not been specified.')
                        step -= 1
                        break
                    message = f'--- STEP 4 ---\nNext we will approximate the level of ' \
                              f'magnification between the camera frame and the projector frame. ' \
                              f'To do this, a grid pattern will be displayed by the projector, ' \
                              f'whilst the camera frame is overlayed with a reference grid. ' \
                              f'Input the SPACING and THICKNESS of the grid lines that the ' \
                              f'projector should display, in terms of number of pixels, ' \
                              f'separated by a space character. Ideally, the SPACING should be ' \
                              f'small enough to allow several grid squares to be seen by the ' \
                              f'camera. The THICKNESS of the grid lines should be as small as ' \
                              f'possible whilst still being clearly visible by the camera. ' \
                              f'When both grids can be seen clearly, record the SQUARE LENGTHS ' \
                              f'of each grid, using any unit of measure. To continue, close the ' \
                              f'window displaying the grids, then enter "next".\n--- SPACING = ' \
                              f'{settings["spacing proj"]} ; integer (positive, non-zero)\n--- ' \
                              f'THICKNESS = {settings["thickness proj"]} ; integer (positive, ' \
                              f'non-zero, << SPACING)\n'
                    user_args, step = custom_input(message, step)
                    if len(user_args) != 2:
                        continue
                    try:
                        [spacing, thickness] = [min(max(0, int(arg)), 255) for arg in user_args]
                        settings['spacing proj'] = spacing
                        settings['thickness proj'] = thickness
                    except ValueError:
                        print('Please specify integer values.')
                    else:
#                         dome_pi4node.transmit('dimensions')
#                         projector_dims = dome_pi4node.receive()
                        grid_proj = overlay_grid(np.zeros(projector_dims, dtype=np.uint8),
                                                 settings['spacing proj'],
                                                 settings['thickness proj'],
                                                 (255, 255, 255))
                        dome_pi4node.transmit(grid_proj.astype(np.uint8))
                        response = dome_pi4node.receive()
                        picture = dome_camera.capture_image(camera_mode)
                        picture_with_grid = overlay_grid(picture, settings['spacing cam'],
                                                         settings['thickness cam'],
                                                         (128, 128, 128))
                        cv2.imshow('Camera frame with grid.', picture_with_grid)
                        cv2.waitKey(0)
                length_cam = None
                length_proj = None
                while step == 5:
                    # Check that the brightness and threshold values have been set.
                    if settings['spacing proj'] is None or settings['thickness proj'] is None:
                        print('Values for projector grid SPACING and/or THICKNESS have not been ' \
                              'specified.')
                        step -= 1
                        break
                    message = f'--- STEP 5 ---\nNow, enter the visual grid sizes that you ' \
                              f'recorded during the previous step as two numbers separated by a ' \
                              f'space character. Enter the SQUARE LENGTH for the projector grid ' \
                              f'first, followed by the SQUARE LENGTH for the camera grid. The ' \
                              f'lengths can be entered according to any unit of measurement, ' \
                              f'such as number of screen pixels - for example, the camera grid ' \
                              f'square length was {settings["spacing cam"]} screen pixels. When ' \
                              f'the values have been entered, type next to continue.\n--- ' \
                              f'Projector grid square length = {length_proj} ; float (positive, ' \
                              f'non-zero number)\n---Camera grid square length = {length_cam} ; ' \
                              f'float (positive, non-zero number)\n'
                    user_args, step = custom_input(message, step)
                    if len(user_args) != 2:
                        continue
                    try:
                        [length_proj, length_cam] = [float(arg) for arg in user_args]
                    except ValueError:
                        print('Please specify numerical values.')
                    else:
                        length_per_pixel_cam = length_cam / settings['spacing cam']
                        length_per_pixel_proj = length_proj / settings['spacing proj']
                        settings['magnification'] = length_per_pixel_cam / length_per_pixel_proj
                while step == 6:
                    if settings['magnification'] is None:
                        print('Values for SQUARE LENGTHs of camera grid and projector grid have ' \
                              'not been specified.')
                        step -= 1
                        break
                    corner_distance = ((np.array(settings['corners'])[0, :] -
                                        np.array(settings['corners'])[1, :]) ** 2).sum() ** 0.5
                    max_region_size = int(min(corner_distance, settings['margin'] * 255))
                    max_increment = int(max_region_size / (2 * settings['magnification']))
                    min_increment = max(1, int(1 / (settings['magnification'])))
                    message = f'--- STEP 6 ---\nFinally, specify the INCREMENT (/width) of scan ' \
                              f'lines that should be illuminated per iteration during the ' \
                              f'calibration. Based on the parameters that have been set in the ' \
                              f'previous step, it would be advisable for the INCREMENT to be no ' \
                              f'smaller than {min_increment} and no greater {max_increment}. ' \
                              f'Generally, using a larger INCREMENT will reduce the time taken ' \
                              f'to complete the calibration procedure, at the cost of ' \
                              f'potentially reducing the accuracy of the calculated ' \
                              f'transformation. Once a value for the INCREMENT has been ' \
                              f'entered, input "next" to continue.\n--- INCREMENT = ' \
                              f'{settings["increment"]} ; integer ({min_increment} - ' \
                              f'{max_increment})\n'
                    user_args, step = custom_input(message, step)
                    if len(user_args[0]) != 0:
                        try:
                            settings['increment'] = int(user_args[0])
                        except ValueError:
                            print('Please specify an integer value.')
                if step != 0:
                    saved_file = None
                while step == 7:
                    if settings['corners'] is None or settings['increment'] is None:
                        print('Value for INCREMENT has not been specified.')
                        step -= 1
                        break
                    message = f'--- STEP 7 ---\nIf you would like to save the currently stored ' \
                              f'calibration parameters, enter a file name without an extension ' \
                              f'(a ".json" file extension will be applied automatically). The ' \
                              f'saved file can be reloaded to hasten the process of future ' \
                              f'calibrations. When ready to run the scanning procedure, input ' \
                              f'"next" to proceed.\n--- Calibration parameters saved to file: ' \
                              f'{saved_file}\n'
                    user_args, step = custom_input(message, step)
                    if len(user_args[0]) != 0:
                        saved_file = user_args[0] + '.json'
                        with open(saved_file, 'w') as calibration_parameters_file:
                            json.dump(settings, calibration_parameters_file)
                if step == 8:
#                     dome_pi4node.transmit('dimensions')
#                     projector_dims = dome_pi4node.receive()
                    num_scan_lines = np.ceil(np.array(projector_dims) /
                                             settings['increment']).astype(int)
                    corners = np.array(settings['corners'])
                    intensities = [np.zeros((corners.shape[0], num_scan_lines[d]))
                                   for d in range(0, 2)]
                    # Iterate over rows first, then columns.
                    for rc, row_column in enumerate(['row', 'column']):
                        for l in range(0, num_scan_lines[rc]):
                            # Reset the projected pattern.
                            dome_pi4node.transmit('all' + 3 * ' 0')
                            response = dome_pi4node.receive()
                            # Illuminate the current row / columm
                            line_start = l * settings['increment']
                            line_end = min(projector_dims[rc], (l + 1) * settings['increment'])
                            dome_pi4node.transmit(row_column + f' {line_start} {line_end}')
                            response = dome_pi4node.receive()
                            print(f'Scanning {row_column}s: Progress = {l} / ' \
                                  f'{num_scan_lines[rc]}')
                            # Take a picture of the illuminated scan lines and record...
                            #...intensities in scanning regions.
                            line_picture = dome_camera.capture_image(camera_mode)
                            gray_line_picture = cv2.cvtColor(line_picture, cv2.COLOR_BGR2GRAY)
                            half_region_size = int(np.ceil(settings['magnification'] *
                                                      settings['increment']))
                            intensities[rc][:, l] = measure_intensities(gray_line_picture,
                                                                        corners, half_region_size)
                    bright_lines = get_bright_lines(intensities)
                    projector_points = (bright_lines * settings['increment']).astype(np.float32)
                    camera_points = corners.astype(np.float32)
                    print('Camera points')
                    print(camera_points)
                    print('Projector points')
                    print(projector_points)
                    camera2projector = cv2.getAffineTransform(camera_points[0:3, :],
                                                              projector_points[0:3, :])
                    camera2projector = np.concatenate((camera2projector, np.array([[0, 0, 1]])), 0)
                    print(camera2projector)
                    mapping_filename = 'camera2projector.npy'
                    np.save(mapping_filename, camera2projector)
                    print(f'Calibration complete.\n--- Affine transform saved to {mapping_filename}')
                    dome_pi4node.transmit('all' + 3 * ' 0')
                    response = dome_pi4node.receive()
                    step += 1
        finally:
            dome_pi4node.transmit('exit')
            response = dome_pi4node.receive()


if __name__ == '__main__':
    default_margin = 0.33
    default_pixelation = [10, 10]
    default_spacing_cam = 100
    default_thickness_cam = 5
    main(default_margin, default_pixelation, default_spacing_cam, default_thickness_cam)