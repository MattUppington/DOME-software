
import DOME_communication as DOMEcomm
# import DOME_pattern_display as DOMEdisp

import numpy as np
import cv2
import time


def main(width=854, height=480):
#     pattern_store = {}
#     pattern_store['blank'] = \
#             DOMEdisp.DOME_PatternManager(width, height)
#     pattern_store['scanner'] = \
#             DOMEdisp.DOME_PatternManager(width, height)
#     pattern_store['grid'] = \
#             DOMEdisp.DOME_PatternManager(width, height)
#     current = 'blank'
#     pattern = pattern_store[current].get_pattern()
#     labels_to_display = []
    pattern = np.zeros([height, width, 3], dtype=np.uint8)
    cv2.namedWindow('Pattern', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Pattern', cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Pattern', pattern)
    cv2.waitKey(0)
    with DOMEcomm.DOME_NetworkNode() as dome_pi0node:
        dome_pi0node.establish_connection()
        while True:
            message = dome_pi0node.receive()
            if isinstance(message, str):
                if message == 'exit':
                    cv2.destroyAllWindows()
                    dome_pi0node.transmit('exit')
                    time.sleep(1)
                    break
                segments = message.split(' ')
                if segments[0] == 'dimensions':
                    dome_pi0node.transmit(pattern.shape)
                elif segments[0] == 'all':
                    pattern[:, :, 0] = int(segments[1])
                    pattern[:, :, 1] = int(segments[2])
                    pattern[:, :, 2] = int(segments[3])
                elif segments[0] == 'row':
                    pattern[int(segments[1]):int(segments[2]),
                            :, :] = 255
                elif segments[0] == 'column':
                    pattern[:, int(segments[1]):int(segments[2]),
                            :] = 255
            elif isinstance(message, np.ndarray):
                pattern = message.copy().astype(np.uint8)
            else:
                dome_pi0node.transmit('Unexpected data type.')
                continue
            cv2.imshow('Pattern', pattern)
            cv2.waitKey(33)
            dome_pi0node.transmit('Done')


if __name__ == '__main__':
    main()
#             elif isinstance(message, list):
#                 labels_to_display = message
#                 dome_pi0node.transmit('Labels to display updated')
#             elif isinstance(message, np.ndarray):
#                 pattern = message
#                 dome_pi0node.transmit('Image applied')
#             elif isinstance(message, dict):
#                 if message['action'] == 'add':
#                     display_store.add_shape(message['label'],
#                                             message['type'],
#                                             message['pose'],
#                                             message['colour'])
#                 elif message['action'] == 'transform':
#                     display_store.apply_transform(message['matrix'],
#                                                   message['labels'])
#                 elif message['action'] == 'paint':
#                     display_store.edit_colours(message['colours'],
#                                                message['labels'])
#                 dome_pi0node.transmit('Action done')
            
            # dictionaries to be used for sending transformation
            # commands: {'group labels': [label, label, label, ...]
            #            'transform': np.ndarray
            #            'colour': list[tuple, tuple]
            #            'action': add / transform
