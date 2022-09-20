# This code is provided to facilitate a socket communication channel between the Raspberry Pi's
# used in the DOME. The DOME (Dynamic Optical Micro Environment) was developed by Ana Rubio
# Denniss. Ensure that there is a copy of this file on both of the DOME's Raspberry Pi's. Edit the
# copy of this file on the Raspberry Pi 0 connected to the projector as follows:
#     1. Comment out the line "identity = 'host'" >>> "#identity = 'host'"
#     2. Uncomment the line >>> "#identity = 'client'" >>> "identity = 'client'"
# First, run the copy of this script from the Raspberry Pi 4 connected to the camera, then run the
# editted version of this script from the Raspberry Pi 0 connected to the projector.
# To set up DOME communications in custom files, ensure that a copy of this file is stored in the
# same directory as the custom files that will be run, both on the camera Pi and the projector Pi.
# The code can then be imported using the following command at the begining of the custom files:
#     import DOME_communications as DOMEcomm
# A communication channel can then be set up using similar code structures to those outlined in the
# "main()" function below, where "NetworkNode()" should be replaced with "DOMEcomm.NetworkNode()".
# #################################################################################################
# Authors = Matthew Uppington <mu15531@bristol.ac.uk>
# Affiliation = Farscope CDT, University of Bristol, University of West England
# #################################################################################################
# This work is licensed under a Creative Commons Attribution 4.0 International License.

import socket
import json
import numpy as np
import functools as ft
import time


class NetworkNode:
    """
    Class for setting up a communication channel between the DOME's camera Pi and projector Pi.
    ---
    Static Attributes
        encryt : str
            Encoding used to convert messages in to bytes sequences, utf8.
        formatting : dict
            Dictionary containing bytes objects to be used as separators and labels in prefixes
            when transmitting messages.
    """
    
    encrypt = 'utf8'
#             np_equiv : dict
#             Dictionary containing the equivalent numpy data type for utf8.
#     np_equiv = {'utf8': np.uint8, 'utf16': np.uint16}
    formatting = {'separator': b'|',
                  'default': b'J',
                  np.ndarray: b'A'}
    
    def __init__(self, port=65455, max_packet=4096):
        """
        Sets up a socket server.
        ---
        Optional Parameters
            max_packet : int
                Maximum number of bytes to be received at a time.
            port : int
                Port that will be used for communication.
        """
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.max_packet = max_packet
        self.port = port
        self.connection = None
    
    def __enter__(self):
        """
        Compatibility method to allow class to be used in "with" statements.
        ---
        Outputs
            self : NetworkNode
                The instance of the NetworkNode class.
        """
        return self
    
    def __exit__(self, type, value, traceback):
        """
        Closes socket servers upon exiting a "with" statement.
        """
        self.server.close()
        if not self.connection is None:
            # socket.shutdown() method is called internally by socket.close().
            self.connection.close()
    
    def accept_connection(self):
        """
        Accept a connection request.
        """
#         try:
        self.server.bind(('', self.port))
        self.server.listen()
        self.connection, address = self.server.accept()
#         except OSError:
#             print(f'WARNING: Port {self.port} is still in a TIME_WAIT state, please restart the ' \
#                   f'environment and try again. The port should exit the TIME_WAIT state ' \
#                   f'naturally after a few minutes.')
    
    def establish_connection(self, address='192.168.1.2'):
        """
        Connect to host server.
        ---
        Optional Parameters
            address : str
                IP address to which connection request is sent.
        """
        self.server.connect((address, self.port))
        self.connection = self.server
    
    def transmit(self, message, sock=None):
        """
        Send data over a socket channel.
        ---
        Parameters
            message
                Data to be transmitted.
        ---
        Optional Parameters
            sock : socket.socket
                Socket to which data is sent.
        """
        if sock is None:
            sock = self.connection
        if not sock is None:
            # Encode message to send according to data type and calculate the size or length of...
            # ... the bytes sequence to send.
            if isinstance(message, np.ndarray):
#                 message_data = message.astype(self.np_equiv[self.encrypt]).tobytes()
                message_data = encode_array(message, self.encrypt)
                # Prefix includes the array shape as a comma separated sequence of dimension...
                # ... sizes, wrapped inside a set of brackets.
#                 num_bytes_to_send = bytes(str(message.shape), self.encrypt)
                label = self.formatting[np.ndarray]
            else:
                message_data = bytes(json.dumps(message), self.encrypt)
#                 num_bytes_to_send = bytes(str(len(message_data)), self.encrypt)
                label = self.formatting['default']
            num_bytes_to_send = bytes(str(len(message_data)), self.encrypt)
            prefix = label + num_bytes_to_send + self.formatting['separator']
            sock.sendall(prefix + message_data)
        else:
            print('WARNING: socket has not been initialised.')
    
    def receive(self, packet_size=None, sock=None):
        """
        Receive data from a socket channel.
        ---
        Optional Parameters
            packet_size : int
                Number of bytes to be read at a time.
            sock : socket.socket
                Socket from which data is read.
        ---
        Outputs
            message
                Recovered data that was transmitted.
        """
        if packet_size is None or packet_size > self.max_packet:
            packet_size = self.max_packet
        if sock is None:
            sock = self.connection
        if not sock is None:
            prefix_bytes = b''
            separator_index = -1
            while separator_index == -1:
                # Receive and append a packet of bytes data from the connected socket.
                prefix_bytes += sock.recv(packet_size)
                # Convert the bytes in to a raw string.
                prefix = prefix_bytes.decode(self.encrypt)
                # Search for the separator symbol to detect the end of prefix, if separator is...
                # ... not detected, -1 is returned by str.find().
                separator_index = prefix.find(self.formatting['separator'].decode(self.encrypt))
#             array_shape = np.array([-1])
            label = prefix_bytes[:1]
            # Recover the number of bytes that should be received.
#             if label == self.formatting[np.ndarray]:
#                 # Read the dimensions of the array to be received as a comma separated sequence...
#                 # ... of integers wrapped in brackets, formatted as a numpy array.
#                 dimensions = prefix_bytes[2:separator_index - 1].decode(self.encrypt)
#                 array_shape = np.array([int(d) for d in dimensions.split(',')])
#                 total_expected_bytes = array_shape.prod()
#             else:
            total_expected_bytes = int(prefix_bytes[1:separator_index])
            # Allocate excess bytes read when searching for prefix in to the first element in a...
            # ... list of bytes packets.
            raw_data_packets = [prefix_bytes[separator_index + 1:]]
            received_bytes = len(raw_data_packets[0])
            # Continue to read bytes until the full message has been read.
            while received_bytes < total_expected_bytes:
                print(received_bytes)
                num_bytes_to_read = min(total_expected_bytes - received_bytes, packet_size)
                packet_bytes = sock.recv(num_bytes_to_read)
#                 received_bytes += num_bytes_to_read
                received_bytes += len(packet_bytes)
                raw_data_packets.append(packet_bytes)
            # Recover the complete sequence of message bytes, then decode according to message...
            # ... type.
#             message_bytes = ft.reduce(lambda x,y:x+y, raw_data_packets)
            message_bytes = b''.join(raw_data_packets)
            if label == self.formatting[np.ndarray]:
#                 message = np.frombuffer(message_bytes,
#                                         dtype=self.np_equiv[self.encrypt]).reshape(array_shape)
                message = decode_array(message_bytes, self.encrypt)
            else:
                message = json.loads(message_bytes.decode(self.encrypt))
            return message
        else:
            print('WARNING: socket has not been initialised.')
            return None

def encode_array(array : np.ndarray, encryption : str):
    '''
    Store the shape and values of an array as a custom sequence of bytes.
    ---
    Parameters
        array : np.ndarray
            Array to be stored as a custom bytes sequence.
        encryption : str
            Method of encryption for generating the bytes sequence.
    ---
    Outputs
        array_bytes : bytes
            Custom sequence of bytes containing shape and values of the array.
    '''
    flat_array = array.flatten()
    array_values = [str(flat_array[i]) for i in range(0, len(flat_array))]
    array_bytes = b''.join([bytes(val + ',', encryption) for val in array_values])
    shape_bytes = bytes(str(array.shape), encryption)
    shape_array_bytes = shape_bytes + array_bytes
    return shape_array_bytes
    
def decode_array(shape_array_bytes : bytes, encryption : str):
    '''
    Retreive an array from a custom sequence of bytes.
    ---
    Parameters
        array_bytes : bytes
            Custom sequence of bytes containing the array values.
        encryption : str
            Method of encryption used to generate the bytes sequence.
    ---
    Outputs
        flat_array : np.ndarray
            Array with the shape and values encoded in the bytes sequence.
    '''
    [shape_data, array_values] = shape_array_bytes.decode(encryption).split(')')
    array_shape = [int(s) for s in shape_data[1:].split(',')]
    flat_array = np.array([int(val) for val in array_values.split(',')[:-1]])
    array = flat_array.reshape(array_shape)
    return array


def main(i_am : str, terminate = 'exit'):
    """
    Creates a socket channel to measure the latency of sending messages of different data types via
    a simple, text-based user interface.
    ---
    Parameters
        i_am : str
            String indicating whether the script will act as a "host" or as a "client".
    ---
    Optional Parameters
        terminate : str
            String that will be used to recognise when the socket communication should end.
    """
    if i_am == 'host':
        # Example setup for DOME camera Pi script.
        with NetworkNode() as camera_node:
            print('Listening for connection...')
            camera_node.accept_connection()
            print('... connection accepted!')
            while True:
                # Request an instruction from the user on each loop to perform associated actions.
                message = None
                instruction = input('Please specify a data type to send:\n')
                instruction_segments = instruction.split(' ')
                # Check for termination instruction.
                if instruction_segments[0] == terminate:
                    camera_node.transmit(terminate)
                    response = camera_node.receive()
                    print(f'Projector Pi said: {response}')
                    break
                # Example instructions to transmit different data types.
                elif instruction_segments[0] == 'integer':
                    message = 0
                    if len(instruction_segments) > 1:
                        message = int(instruction_segments[1])
                elif instruction_segments[0] == 'float':
                    message = 0.0
                    if len(instruction_segments) > 1:
                        message = float(instruction_segments[1])
                elif instruction_segments[0] == 'string':
                    num_repeats = 1
                    test_string = 'TestingTesting123...'
                    if len(instruction_segments) > 1:
                        num_repeats = int(instruction_segments[1])
                    if len(instruction_segments) > 2:
                        test_string = instruction_segments[2]
                    message = test_string * num_repeats
                elif instruction_segments[0] == 'list':
                    message = [0, 1, 2, 3, 4, 5]
                    if len(instruction_segments) > 1:
                        message = [n for n in range(0, instruction_segments[1])]
                elif instruction_segments[0] == 'dictionary':
                    message = {0: 1}
                    if len(instruction_segments) > 1:
                        for d in range(0, int(instruction_segments[1])):
                            message[d + 1] = d + 2
                elif instruction_segments[0] == 'array':
                    array_shape = np.array([10, 10])
                    if len(instruction_segments) > 1:
                        array_shape = np.array([int(s) for s in
                                                instruction_segments[1].split(',')])
                    message = np.ceil(255 * np.random.random(array_shape)).astype(np.uint8)
                else:
                    print('Examples have been prepared for the following data types:\n' \
                          'string, integer, float, list, dictionary or array.')
                    continue
                transmit_time = time.perf_counter()
                camera_node.transmit(message)
                response = camera_node.receive()
                respond_time = time.perf_counter()
                delay = respond_time - transmit_time
                print(f'Got confirmation in {delay} seconds')
                print(f'Projector Pi said: {response}')
    elif i_am == 'client':
        # Example setup for DOME projector Pi script.
        with NetworkNode() as projector_node:
            print('Requesting connection...')
            projector_node.establish_connection()
            print('... connection established!')
            while True:
                # Perform operations with received message.
                message = projector_node.receive()
                # Check for a termination command.
                if isinstance(message, str) and message == terminate:
                    print('Received exit command.')
                    projector_node.transmit('I\'m exiting now, bye!')
                    # Delay closing down connection to ensure confirmation is transmitted properly.
                    time.sleep(1)
                    break
                # Example instructions to receive different data types.
                elif isinstance(message, int):
                    print(f'Received integer: {message}')
                    projector_node.transmit('Thank\'s for the integer!')
                elif isinstance(message, float):
                    print(f'Received float: {message}')
                    projector_node.transmit('Thank\'s for the float!')
                elif isinstance(message, str):
                    print(f'Received string of length {len(message)}:\n')
                    print(message)
                    projector_node.transmit('Thank\'s for the string!')
                elif isinstance(message, list):
                    print(f'Received list of length {len(message)}:\n')
                    print(message)
                    projector_node.transmit('Thank\'s for the list!')
                elif isinstance(message, dict):
                    print(f'Received dictionary of length {len(message)}:\n')
                    print(message)
                    projector_node.transmit('Thank\'s for the dictionary!')
                elif isinstance(message, np.ndarray):
                    print('Received array of shape ' + str(message.shape))
                    print(message)
                    projector_node.transmit('Thank\'s for the array!')
                else:
                    projector_node.transmit('Unexpected data type: {type(message)}')


if __name__ == '__main__':
    identity = 'host'
    #identity = 'client'
    main(identity)
