# This code is provided to assist with managing operations on the Raspberry Pi 4 connected to the
# camera used in the DOME. The DOME (Dynamic Optical Micro Environment) was developed by Ana Rubio
# Denniss. These include capturing images with the camera and managing gpio outputs on the
# Raspberry Pi 4. Examples of how various operations are performed can be found in the "main()"
# function below, which can be run with some default parameters by running this file as a script.
# To access the functionalities in custom scripts, ensure that a copy of this file is stored in
# the same directory. Then run the following command to import the code at the begining of the
# custom files:
#     import DOME_imaging_utilities as DOMEutil
# The manager classes can then be accessed and used by replicating a similar code structure as in
# the "main()" function below, where "CameraManager()" and "PinManager()" should be replaced with
# "DOMEutil.CameraManager()" and "DOMEutil.PinManager()" respectively.
# #################################################################################################
# Authors = Matthew Uppington <mu15531@bristol.ac.uk>
# Affiliation = Farscope CDT, University of Bristol, University of West England
# #################################################################################################
# This work is licensed under a Creative Commons Attribution 4.0 International License.

from picamera import PiCamera
from picamera.array import PiRGBArray
import RPi.GPIO as GPIO
import time
import numpy as np
import cv2
import os


class EmptyLabelError(Exception):
    '''
    Exception class for handling errors raised when an empty string is provided as a key in a
    dictionary of camera settings. Empty string is used internally to represent an empty
    memory of last used settings.
    '''
    
    def __init__(self):
        '''
        Sets up the parent class for exceptions and provides an error message.
        '''
        error_message = 'Empty string should not be used as label for camera settings.'
        super().__init__(error_message)


class FailedImageWriteError(Exception):
    '''
    Exception class for handling errors which would otherwise occur silently when an attempt
    to save an image to a file fails. This can occur when the provided directory does not exist.
    '''
    
    def __init__(self, destination):
        '''
        Sets up the parent class for exceptions and provides an error message with the directory
        that was given for the location of the save file.
        '''
        error_message = f'Image could not be saved to the following directory: {destination}'
        super().__init__(error_message)


class ConflictingPinLabel(Exception):
    '''
    Exception class for handling errors raised when a new label is assigned to a pin that is
    already registered or when an attempt is made to register a new pin under a label that is
    already assigned to another pin.
    '''
    
    def __init__(self, registered_pins, new_label, new_pin):
        '''
        Sets up the parent class for exceptions and provides an error message with the directory
        that was given for the location of the save file.
        '''
        error_message('Something went wrong...')
        super().init(error_message)


class CameraManager:
    '''
    Class for managing operations with the Raspberry Pi camera.
    '''
    
    def __init__(self, save_dir='', init_settings=None, max_duration=60):
        '''
        Sets up the default values to be used when capturing images.
        ---
        Optional Parameters
            save_dir : str
                Base directory to save images to.
            init_settings : dict[str : dict['resolution': (int, int), 'iso' : int, ...]]
                Dictionary containing camera settings as dictionaries. Values can be specified for
                'resolution', 'iso', 'shutter speed', 'exposure mode' and 'framerate'.
            max_duration : int
                Maximum number of seconds for which to display live video in fullscreen.
        '''
        self.save_dir = save_dir
        self.max_duration = max_duration
        self.camera = PiCamera()
        default_settings = {}
        # Default resolution = (1920, 1080).
        default_settings['resolution'] = tuple(self.camera.resolution)
        # Default iso = 0.
        default_settings['iso'] = self.camera.iso
        # Default exposure mode = 'auto'.
        default_settings['exposure mode'] = self.camera.exposure_mode
        # Default rotation = 0.
        default_settings['rotation'] = self.camera.rotation
        # Default hflip = False, override with True.
        default_settings['hflip'] = True
        # Default vflip = False.
        default_settings['vflip'] = self.camera.vflip
        # Default framerate = 30.
        default_settings['framerate'] = int(self.camera.framerate)
        # Default shutter speed = 0.
        default_settings['shutter speed'] = self.camera.shutter_speed
        # Default crop = (0.0, 0.0, 1.0, 1.0), possibly depreciated, see zoom instead.
        default_settings['crop'] = self.camera.crop
        # Default awb mode = 'auto'.
        default_settings['awb mode'] = self.camera.awb_mode
        # Default meter mode = 'average'.
        default_settings['meter mode'] = self.camera.meter_mode
        # Default brightness = 50.
        default_settings['brightness'] = self.camera.brightness
        self.settings = {'default': default_settings}
        if not init_settings is None:
            for mode, settings in init_settings.items():
                self.store_settings(mode, settings)
        self.last_used_mode = ''
        self.load_settings('default')
        self.camera.hflip = True
        # Allow time for the camera to warm up.
        time.sleep(2)
    
    
    def __enter__(self):
        """
        Compatibility method to allow class to be used in "with" statements.
        ---
        Outputs
            self : CameraManager
                The instance of the CameraManager class.
        """
        return self
    
    
    def __exit__(self, type, value, traceback):
        '''
        Close camera object upon exiting a "with" statement.
        '''
        self.camera.close()
    
    
    def store_settings(self, mode_name : str, mode_settings : dict):
        '''
        Store or overwrite a mode of camera settings.
        ---
        Parameters
            mode_name : str
                Name of camera mode to store.
            mode_settings : dict
                Dictionary containing camera settings.
        '''
        if mode_name == '':
            raise EmptyLabelError()
        self.settings[mode_name] = mode_settings
        # Use default values for any settings not defined in the dictionary of settings provided.
        if 'default' in self.settings.keys():
            for key, value in self.settings['default'].items():
                if not key in mode_settings.keys():
                    self.settings[mode_name][key] = value
        # Empty memory of last used settings to account for potential overwrites.
        self.last_used_mode = ''
    
    
    def load_settings(self, mode_name : str):
        '''
        Transcribe a mode of stored settings to the camera.
        ---
        Parameters
            mode_name : str
                Label of stored settings to load.
        '''
        # Check whether a new mode needs to be loaded.
        if not mode_name == self.last_used_mode:
            for key, value in self.settings[mode_name].items():
                if key == 'resolution':
                    self.camera.resolution = value
                elif key == 'iso':
                    self.camera.iso = value
                elif key == 'exposure mode':
                    self.camera.exposure_mode = value
                elif key == 'rotation':
                    self.camera.rotation = value
                elif key == 'hflip':
                    self.camera.hflip = value
                elif key == 'vflip':
                    self.camera.vflip = value
                elif key == 'framerate':
                    self.camera.framerate = value
                elif key == 'shutter speed':
                    self.camera.shutter_speed = value
                elif key == 'crop':
                    self.camera.crop = value
                elif key == 'awb mode':
                    self.camera.awb_mode = value
                elif key == 'meter mode':
                    self.camera.meter_mode = value
                elif key == 'brightness':
                    self.camera.brightness = value    
            # Update memory of last used settings.
            self.last_used_mode = mode_name
    
    
    def capture_image(self, mode_name='default'):
        '''
        Capture a still image of the camera's field of view using video port.
        ---
        Optional Parameters
            mode_name : str
                Label indicating which stored camera settings to use.
        ---
        Outputs
            image : ndarray
                Image captured by camera.
        '''
        self.load_settings(mode_name)
        resolution = self.camera.resolution
        rawCapture = PiRGBArray(self.camera, size=resolution)
        image = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        for frame in self.camera.capture_continuous(rawCapture, format = "bgr",
                                                    use_video_port=True):
            image = frame.array
            break
        return image
    
    
    def save_image(self, file_name : str, image : np.ndarray):
        '''
        Save an image to a file in the assigned save directory.
        ---
        Parameters
            image : ndarray
                Image to be saved to a file.
            file_name : str
                Name of file the image will be saved to.
        '''
        destination = os.path.join(self.save_dir, file_name)
        if not cv2.imwrite(destination, image):
            print(destination)
            raise FailedImageWriteError(destination)
    
    def show_live_feed(self, duration=5):
        '''
        Display the live video from the camera frame in a full screen window.
        ---
        Parameters
            duration : int
                Number of second for which to show the live video.
        '''
        num_seconds = min(duration, self.max_duration)
        self.camera.start_preview()
        time.sleep(num_seconds)
        self.camera.stop_preview()


class PinManager:
    '''
    Class for managing the output voltages across Raspberry Pi gpio pins.
    '''
    
    def __init__(self, pins=None):
        '''
        Sets up a list of gpio pins to be managed and their associated labels.
        ---
        Parameters
            pins : dict[str: int]
                Dictionary of pin labels to be referenced and their corresponding pin numbers.
        '''
        self.pins = pins
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        self.pins = {}
        if not pins is None:
            for pin_label, pin_number in pins.items():
                self.add_pin_label(pin_label, pin_number)
                GPIO.setup(pin_number, GPIO.OUT)
    
    
    def __enter__(self):
        """
        Compatibility method to allow class to be used in "with" statements.
        ---
        Outputs
            self : PinManager
                The instance of the PinManager class.
        """
        return self
    
    
    def __exit__(self, type, value, traceback):
        '''
        Turn off all registered gpio pins upon exiting a "with" statement.
        '''
        for pin_number in self.pins.values():
            GPIO.output(pin_number, GPIO.LOW)
    
    
    def add_pin_label(self, label : str, number : int):
        '''
        Register a pin as a gpio output with a specified label.
        ---
        Parameters
            label : str
                Label used to refer to pin.
            number : int
                Pin number to be assigned as a gpio output.
        '''
        if label in self.pins.keys() or number in self.pins.values():
            raise ConflictingPinLabel(self.pins, label, number)
        self.pins[label] = number
        GPIO.setup(number, GPIO.OUT)
            
    
    def toggle(self, label : str, action):
        '''
        Change voltage output of a pin.
        ---
        Parameters
            label : str
                Assigned label of pin to be changed.
            action
                True, 1, 'on' or 'high' to turn on; False, 0, 'off' or 'low' to turn off.
        '''
        if isinstance(action, str):
            if action.lower() == 'on' or action.lower() == 'high':
                turn_on = True
            elif action.lower() == 'off' or action.lower() == 'low':
                turn_on = False
        else:
            turn_on = bool(action)
        if turn_on:
            GPIO.output(self.pins[label], GPIO.HIGH)
        else:
            GPIO.output(self.pins[label], GPIO.LOW)


def set_system_time(date : str):
    '''
    Manually set the date and time for the Raspberry Pi OS.
    ---
    Parameters
        date : str
            String containing data and time info; format should be 'YYYY-MM-DD HH:MM:SS'.
    '''
    os.system(f'sudo date -s \'{date}\'')


def main(save_directory : str, camera_modes : dict, pin_labels : dict, terminate='exit'):
    '''
    Allows for some basic functionalities to be tested with the Raspberry Pi camera and the gpio
    pins via a simple, text-based user interface.
    ---
    Parameters
        save_directory : str
            Directory to save images to.
        camera_modes : dict[str : dict['iso' : int, 'shutter speed' : int]]
            Example camera settings to test.
        pin_labels : dict[str: int]
            Example pins and their associated labels to test.
    ---
    Optional Parameters
        terminate : str
            String that will be used to recognise when testing is finished.
    '''
    date_time_format = 'YYYY-MM-DD HH:MM:SS'
    picture = np.zeros((10, 10, 3), dtype=np.uint8)
    with CameraManager(save_directory, camera_modes) as DOMEcam, \
            PinManager(pin_labels) as DOMEgpio:
        while True:
            instruction = input('Please specify an operation to test:\n')
            instruction_segments = instruction.split(' ')
            # Check for termination instruction.
            if instruction_segments[0] == terminate:
                cv2.destroyAllWindows()
                break
            # Example instructions to test different operations.
            elif instruction_segments[0] == 'capture':
                picture = DOMEcam.capture_image('myUVmode')
                cv2.imshow('Captured image', picture)
                cv2.waitKey(0)
            elif instruction_segments[0] == 'save':
                if len(instruction_segments) > 1:
                    image_file_name = instruction_segments[1]
                    try:
                        DOMEcam.save_image(image_file_name, picture)
                    except FailedImageWriteError:
                        print('Image has not been saved.')
                        continue
                    else:
                        print('Image was saved.')
                else:
                    print('Please also provide a name for the image file, e.g. ' \
                          'save test_picture.png\'')
            elif instruction_segments[0] == 'live':
                if len(instruction_segments) > 1:
                    DOMEcam.show_live_feed(int(instruction_segments[1]))
                else:
                    DOMEcam.show_live_feed(5)
            elif instruction_segments[0] == 'gpio':
                if len(instruction_segments) < 3:
                    print('Please also specify the pin label and action, e.g. \'gpio UV on\'')
                    continue
                else:
                    DOMEgpio.toggle(instruction_segments[1], instruction_segments[2])
                    print(f'Set {instruction_segments[1]} pin to {instruction_segments[2]}.')
            elif instruction == 'set system time':
                manual_date_time = input(f'Please input new system time in ' \
                                         f'{date_time_format}\' format:')
                confirm = input(f'Are you sure you wish to proceed with changing the ' \
                                f'system time to \'{manual_date_time}\'? Y/N')
                if confirm == 'Y':
                    set_system_time(manual_date_time)
            else:
                print('Examples have been prepared for the following actions:\n' \
                      'capture, save, live, gpio or set system time.')


if __name__ == '__main__':
    image_folder = os.getcwd()
    example_camera_modes = {'myUVmode': {'shutter speed': 6000000}}
    example_gpio_labels = {'UV': 12, 'BrightField': 18}
    main(image_folder, example_camera_modes, example_gpio_labels)
