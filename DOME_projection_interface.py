import DOME_communication as DOMEcomm
import DOME_transformation as DOMEtran

import numpy as np
import time
import cv2

class ScreenManager():
    
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.current = ''
        self.images = []
        self.screens = {}
        self.store_image(np.zeros(self.dimensions))
        self.switch_to_screen('default')
    
    def store_image(self, image):
        if image.shape == self.dimensions:
            self.images.append(image.astype(np.uint8))
        else:
            print(f'WARNING: image dimensions do not match screen' +
                  f'dimensions:\n--- image = {image.shape}\n' +
                  f'--- screen = {self.dimensions}')
    
    def set_image_to_screen(self, image_index):
        if image_index >= len(self.images):
            self.screens[self.current]['image'] = image_index
        else:
            images_list = [i for i in self.images.keys()]
            print(f'WARNING: image index {image_index} exceeds ' + \
                  f'number of stored images ({len(self.images)})')
    
    def switch_to_screen(self, new_screen):
        self.current = new_screen
        if not new_screen in self.screens.keys():
            poses = DOMEtran.PoseManager()
            self.screens[new_screen] = {'image': 0,
                                        'poses': poses,
                                        'shown': []}
    
    def shapes_shown_to_screen(self, shown_list):
        self.screens[self.current]['shown'] = shown_list
    
    def add_shape_to_screen(self, label, shape_type, pose, colour):
        self.screens[self.current]['poses'].add_shape(label,
                                                      shape_type,
                                                      pose, colour)
        if not label in self.screens[self.current]['shown']:
            self.screens[self.current]['shown'].append(label)
    
    def transform_shapes_on_screen(self, matrix, labels=None):
        self.screens[self.current]['poses'].apply_transform(matrix,
                                                            labels)
    
    def set_colour_on_screen(self, colour, label, indices=None):
        self.screens[self.current]['poses'].set_colour(colour,
                                                       label,
                                                       indicies)
    
    def get_pattern_for_screen(self, axis_offset=None, labels=None):
        image_name = self.screens[self.current]['image']
        background = self.images[image_name]
        pose_manager = self.screens[self.current]['poses']
        display = pose_manager.draw_shapes(background, axis_offset,
                                           labels)
        return display


def main(dimensions, refresh_delay):
    screen_manager = ScreenManager(dimensions)
    pattern = screen_manager.get_pattern_for_screen()
    cv2.namedWindow('Pattern', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Pattern', cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    with DOMEcomm.NetworkNode() as dome_pi0node:
        cv2.imshow('Pattern', pattern)
        cv2.waitKey(0)
        dome_pi0node.establish_connection()
        all_command_types = ['screen', 'image', 'add',
                             'transform', 'colour', 'shown']
#         x = {'screen': 'default',
#              'image': image_index,
#              'add': {'label': 'spotlight', 'shape': 'circle',
#                      'pose': [[1,0,0],[0,1,0],[0,0,1]], 'colour':[255]},
#              'add2': {'label': 'spotlight', 'shape': 'circle',
#                       'pose': [[1,0,0],[0,1,0],[0,0,1]], 'colour':[255]},
#              'transform': {'matrix': [[1,0,0],[0,1,0],[0,0,1]], 'labels': ['a']},
#              'transform2': {'matrix': [[1,0,0],[0,1,0],[0,0,1]], 'labels': ['b']},
#              'colour': {'colour': [255], 'label': 'splotlight', 'indices': [0]},
#              'shown': ['name1', 'name2']}
        projecting = True
        while projecting:
            message = dome_pi0node.receive()
            if isinstance(message, np.ndarray):
                pattern = message.copy().astype(np.uint8)
                screen_manager.store_image(pattern)
            elif isinstance(message, dict):
                for command_type in all_command_types:
                    commands = [c for c in message.keys()
                                if command_type in c]
                    for c in commands:
                        if command_type == 'screen':
                            screen_manager.switch_to_screen(
                                    message[c])
                        elif command_type == 'image':
                            screen_manager.set_image_to_screen(
                                    message[c])
                        elif command_type == 'add':
                            label = message[c]['label']
                            shape_type = message[c]['shape type']
                            pose = np.array(message[c]['pose'])
#                             blah = [[1, 0, 0], [0, 1, 0],[0, 0, 1]]
#                             [list(x) for x in np.array(blah)]
                            colour = None
                            if 'colour' in message[c].keys():
                                colour = message[c]['colour']
                            screen_manager.add_shape_to_screen(
                                    label, shape_type, pose, colour)
                        elif command_type == 'transform':
                            matrix = np.array(message[c]['matrix'])
                            labels = None
                            if 'labels' in message[c].keys():
                                labels = message[c]['labels']
                            screen_manager.transform_shapes_on_screen(
                                    matrix, labels)
                        elif command_type == 'colour':
                            colour = message[c]['colour']
                            label = message[c]['label']
                            indices = None
                            if 'indices' in message[c].keys():
                                indices = message[c]['indices']
                            screen_manager.set_colour_on_screen(
                                    colour, label, indices)
                        elif command_type == 'shown':
                            screen_manager.shapes_shown_to_screen(
                                    message[c])
                pattern = screen_manager.get_pattern_for_screen()
            elif isinstance(message, str):
                if message == 'exit':
                    cv2.destroyAllWindows()
                    dome_pi0node.transmit('exit')
                    time.sleep(1)
                    projecting = False
                segments = message.split(' ')
                if segments[0] == 'dimensions':
                    dome_pi0node.transmit(dimensions)
                    continue
                elif segments[0] == 'all':
                    for c in range(0, 3):
                        pattern[:, :, c] = int(segments[c + 1])
                elif segments[0] == 'row':
                    pattern[int(segments[1]):int(segments[2]),
                            :, :] = 255
                elif segments[0] == 'column':
                    pattern[:, int(segments[1]):int(segments[2]),
                            :] = 255
                else:
                    dome_pi0node.transmit(f'Unrecognised string:' + \
                                          f'{message}')
                    continue
            else:
                dome_pi0node.transmit('Unexpected data type.')
                continue
            cv2.imshow('Pattern', pattern)
            cv2.waitKey(refresh_delay)
            dome_pi0node.transmit('Done.')

if __name__ == '__main__':
    dims = (480, 854, 3)
    refresh_delay = 33
    main(dims, refresh_delay)
        