# This code is provided to assist with creating affine matrices to describe transformations or
# poses of objects in the camera or projector frame of the DOME. The DOME (Dynamic Optical Micro
# Environment) was developed by Ana Rubio Denniss. To access the functions in custom scripts,
# ensure that a copy of this file is stored in the same directory. Then run the following command
# to import the code at the beginning of the custom files:
#     import DOME_transformation as DOMEtran
# Classes and functions can then be access via DOMEtran.CLASS or DOMEtran.FUNCTION, where "CLASS" /
# "FUNCTION" refers to the name of the relevant class or function respectively.
# #################################################################################################
# Authors = Matthew Uppington <mu15531@bristol.ac.uk>
# Affiliation = Farscope CDT, University of Bristol, University of West England
# #################################################################################################
# This work is licensed under a Creative Commons Attribution 4.0 International License.

import numpy as np
import cv2


class PoseManager:
    '''
    Class for storing and manipulating poses of shapes.
    '''
    
    def __init__(self, shape_groups=None):
        '''
        Initialise the stored groups of shapes.
        ---
        Optional Inputs
            shape_groups : dict[list[dict,],]
                Structured dictionary of shape groups.
        '''
        self.shape_groups = {}
        if not shape_groups is None:
            self.shape_groups = shape_groups
    
    def add_shape(self, label : str, shape_type : str, pose : np.array, colour=None):
        '''
        Store parameters of a new shape.
        ---
        Inputs
            label : str
                Dictionary key to identify the group of shapes to add to.
            shape_type : str
                String specifying the type of shape; 'circle', 'square' or 'triangle'.
            shape_pose : np.ndarray
                3x3 affine transformation matrix describing pose.
        Optional Inputs
            colour : list
                Shape colour represented as either a single gray scale value [int], pixel colour
                values listed in BGR order [int, int, int], or BGR pixel values plus an alpha value
                [int, int, int, float].
        '''
        bgra = format_colour(colour)
        shape_info = {'type': shape_type, 'pose': pose, 'bgra': bgra}
        if label in self.shape_groups.keys():
            self.shape_groups[label].append(shape_info)
        else:
            self.shape_groups[label] = [shape_info]
    
    def apply_transform(self, matrix : np.ndarray, labels=None):
        '''
        Applies an affine transformation to stored shapes.
        ---
        Inputs
            matrix : np.ndarray
                3x3 affine transformation matrix to apply.
        ---
        Optional Inputs
            labels : list
                List of dictionary keys identifying the shape groups to transform. By default all
                shape groups are transformed.
        '''
        if labels is None:
            labels = self.shape_groups.keys()
        for l in labels:
            shape_group = self.shape_groups[l]
            for s, shape in enumerate(shape_group):
                self.shape_groups[l][s] = matrix @ shape['pose']
        
    def set_colour(self, colour : list, label : str, indices=None):
        '''
        Set the colours of stored shapes.
        ---
        Inputs
            colour : list
                New colour represented as either a single gray scale value [int], pixel colour
                values listed in BGR order [int, int, int], or BGR pixel values plus an alpha value
                [int, int, int, float].
            label : str
                Dictionary key identifying the shape group to set colour for.
        ---
        Optional Inputs
            indices : list
                List of indices specifying the specific shapes in a shape group to set colour for.
                By default the colours of all shapes in a shape group will be set.
        '''
        bgra = format_colour(colour)
        if indices is None:
            indices = [i for i in range(0, len(self.shape_groups[label]))]
        for i in indices:
            self.shape_groups[label][i]['colour'] = bgra
    
    def draw_shapes(self, image : np.ndarray, axis_offset=None, labels=None):
        '''
        Draw stored shapes on to an image.
        ---
        Inputs
            image : np.ndarray
                Image to draw shapes on to.
            axis_offset : list[int, int]
                Relative position of image in the shape reference frame.
        ---
        Optional Inputs
            labels : list[str,]
                List of dictionary keys ientifying which shapes to draw. By default, all stored
                shapes are drawn.
        ---
        Outputs
            _ : np.ndarray
                Updated version of original image with drawn shapes overlayed.
        '''
        if axis_offset is None:
            axis_offset = [0, 0]
        if labels is None:
            labels = self.shape_groups.keys()
        v, h = np.indices((image.shape[0], image.shape[1]))
        v_offset = v + axis_offset[0]
        h_offset = h + axis_offset[1]
        new_image = image.copy().astype(np.uint8)
        for l in labels:
            shape_group = self.shape_groups[l]
            for shape in shape_group:
                inverse_map = np.linalg.inv(shape['pose'])
                v_i, h_i = map_coordinates(v_offset, h_offset, inverse_map)
                flags = check_shape(shape['type'], v_i, h_i)
                gray_scale = int(np.array(shape['bgra'][0:3]).sum() / 3)
                paint = np.ones(image.shape) * gray_scale
                if len(image.shape) == 3:
                    flags = np.tile(np.expand_dims(flags, 2), (1, 1, image.shape[2]))
                    channels = [np.ones((image.shape[0], image.shape[1], 1)) * \
                                shape['bgra'][c] for c in range(0, 3)]
                    paint = np.concatenate(channels, 2)
                new_layer = (1 - shape['bgra'][-1]) * image.astype(int) + \
                             shape['bgra'][-1] * paint
                new_image = np.where(flags, new_layer.astype(np.uint8), new_image)
        return new_image


def check_shape(shape_type : str, v_mat : np.ndarray, h_mat : np.ndarray):
    '''
    Check for coordinates in a given shape.
    ---
    Inputs
        shape_type : str
            String specifying the type of shape to check for; 'circle', 'square' or 'triangle'.
        v_mat : np.ndarray
            MxN array of vertical coordinates.
        h_mat : np.ndarray
            MxN array of horizontal coordinates.
    ---
    Outputs
        _ : np.ndarray
            MxN array of boolean flags indicating which coordinates are within the specified shape.
    '''
    if shape_type == 'circle':
        # Check for circle with unit radius centred on origin.
        return (v_mat ** 2) + (h_mat ** 2) <= 1
    elif shape_type == 'square':
        # Check for square with unit side length centred on origin.
        return (v_mat <= 0.5) & (v_mat > -0.5) & (h_mat <= 0.5) & (h_mat > -0.5)
    elif shape_type == 'triangle':
        # Check for right-angled triangle with orthogonal sides that have unit length, intersect...
        # ... at the origin and are aligned with the positive vertical (down) and positive...
        # ... horizontal (right) axes.
        return (v_mat >= 0) & (h_mat >= 0) & (v_mat + h_mat <= 1)
    return np.zeros(v_mat.shape).astype(bool)


# --- Made obsolete by numpy's .indices function ---
# def coordinate_matrices(vertical : int, horizontal : int):
#     '''
#     Get matrices containing integer vertical and horizontal coordinates.
#     ---
#     Inputs
#         vertical : int
#             Range of vertical axis.
#         horizontal : int
#             Range of horizontal axis.
#     ---
#     Outputs
#         v_matrix : np.ndarray
#             Matrix of integer vertical coordinates.
#         h_matrix : np.ndarray
#             Matrix of integer horizontal coordinates.
#     '''
#     v_indices = np.expand_dims(np.arange(vertical), 1)
#     h_indices = np.expand_dims(np.arange(horizontal), 0)
#     v_matrix = np.tile(v_indices, (1, horizontal))
#     h_matrix = np.tile(h_indices, (vertical, 1))
#     return v_matrix, h_matrix
            

def format_colour(colour : list):
    '''
    Convert colour information in to BGRA format.
    ---
    Inputs
        colour : list
            Shape colour represented as either a single gray scale value [int], pixel colour
            values listed in BGR order [int, int, int], or BGR pixel values plus an alpha value
            [int, int, int, float].
    ---
    Outputs
        _ : list[int, int, int, float]
            Colour formatted as a BGRA list. By default [255, 255, 255, 1.0] is returned if the
            format of the input colour is not recognised.
    '''
    alpha = 1.0
    if not isinstance(colour, list):
        return [255, 255, 255, alpha]
    elif len(colour) == 1:
        return colour * 3 + [alpha]
    elif len(colour) == 3:
        return [c for c in colour] + [alpha]
    elif len(colour) == 4:
        return [c for c in colour]


def format_affine(vectors_matrix : np.ndarray):
    '''
    Extend standard format vectors in to affine format.
    ---
    Inputs
        vector_matrix : np.ndarray
            2xM array containing M 2x1 vectors in [[x], [y]] format.
    ---
    Outputs
        formatted_vectors : np.ndarray
            3xM array containing M 3x1 affine vectors in [[x], [y], [1]] format.
    '''
    coefficient = np.eye(3, 2)
    constant = np.zeros((3, vectors_matrix.shape[1]))
    constant[2, :] = 1
    formatted_vectors = coefficient @ vectors_matrix + constant
    return formatted_vectors


def reflect_axes(affine_matrix : np.ndarray):
    '''
    Reverse the x and y axes in an affine transformation matrix.
    ---
    Inputs
        affine_matrix : np.ndarray
            Affine matrix with original axes.
    ---
    Outputs
        reflected_matrix : np.ndarray
            Affine matrix with reflected axes.
    '''
    reflected_matrix = np.array([[affine_matrix[1, 1], affine_matrix[1, 0], affine_matrix[1, 2]],
                                 [affine_matrix[0, 1], affine_matrix[0, 0], affine_matrix[0, 2]],
                                 [0, 0, 1]])
    return reflected_matrix


def calculate_mapping(src_points : np.ndarray, dst_points : np.ndarray):
    '''
    Calculate an affine transformation mapping from a set of four coordinate pairs. The values in
    the first index or each array represent the negative vertical component (traditionally "y" in
    cv2) and the second index represents the positive horizontal component (traditionally "x" in
    cv2), measured from an origin in the top left corner of an image.
    ---
    Inputs
        src_points : np.ndarray
            4x2 array containing coordinates from the source frame. 
        dst_points : np.ndarray
            4x2 array containing coordinates from the destination frame.
    ---
    Outputs
        average_mapping : np.ndarray
            3x3 affine transformation matrix that maps the specified points from the source frame
            to the corresponding points from the destination frame.
    '''
    src_points_compat = src_points[:, -1:-(src_points.shape[1] + 1):-1].astype(np.float32)
    print(src_points_compat)
    dst_points_compat = dst_points[:, -1:-(dst_points.shape[1] + 1):-1].astype(np.float32)
    print(dst_points_compat)
    index_sets = [[n for n in range(0, 4) if n != m] for m in range(0, 4)]
    print(index_sets)
    sum_mappings_compat = np.zeros((2, 3))
    for indices in index_sets:
        print(indices)
        sum_mappings_compat += cv2.getAffineTransform(src_points_compat[indices, :],
                                                      dst_points_compat[indices, :])
    average_mapping_compat = sum_mappings_compat / 4
    average_mapping = reflect_axes(average_mapping_compat)
    return average_mapping


def transform_image(image : np.ndarray, transform_matrix : np.ndarray, dimensions : tuple):
    '''
    Apply an affine transformation to an image using cv2 package.
    ---
    Inputs
        image : np.ndarray
            Image to be transformed.
        transform_matrix : np.ndarray
            3x3 affine transform matrix to apply to image.
        dimensions : tuple[int, int]
            Dimensions to use for transformed image.
    ---
    Outputs
        new_image : np.ndarray
            Transformed image with specified dimensions.
    '''
    compatible_transform = reflect_axes(transform_matrix)
    # Reflecting axes is necessary because cv2 package considers origin at top left corner of...
    # ...image with x axis along width and y axis down height. This accounts for the fact that...
    # ...rotations are also considered as positive clockwise by cv2 package.
    new_image = cv2.warpAffine(image, compatible_transform[0:2, :], (dimensions[1], dimensions[0]))
    return new_image


def linear_transform(scale=None, shear=None, shift=None):
    '''
    Create affine transformation matrix for stretches, shears and translations.
    ---
    Optional Inputs
        scale : tuple(float, float)
            Amounts of stretching to apply in the x and y axes.
        shear : tuple(float, float)
            Amounts of shearing to apply in the x and y axes.
        shift : tuple(float, float)
            Amounts of translation to aply in the x and y axes.
    ---
    Outputs
        affine_matrix : np.ndarray
            3x3 affine transformation matrix.
    '''
    affine_matrix = np.eye(3)
    if not scale is None:
        affine_matrix[0, 0] = scale[0]
        affine_matrix[1, 1] = scale[1]
    if not shear is None:
        affine_matrix[0, 1] = shear[0]
        affine_matrix[1, 0] = shear[1]
    if not shift is None:
        affine_matrix[0:2, 2] = shift
    return affine_matrix


def rotational_transform(angle=0, centre=None):
    '''
    Create affine transformation matrix to rotate about a point.
    ---
    Inputs
        angle : float
            Number of radians to rotate by.
    ---
    Optional Inputs
        centre : np.ndarray
            2x1 vector representing the centre of rotation. If not specified, rotation will be
            applied about the origin.
    ---
    Outputs
        rotation_matrix / affine_matrix : np.ndarray
            3x3 affine transformation matrix.
    '''
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])
    if centre is None:
        return rotation_matrix
    affine_matrix = transform_linear(shift=centre) @ rotation_matrix @ \
                    transform_linear(shift=-centre)
    return affine_matrix


def map_coordinates(x_coordinates : np.ndarray, y_coordinates : np.ndarray,
                    transform_matrix : np.ndarray):
    '''
    Applies an affine transformation to matrices of x and y coordinates.
    ---
    Inputs
        x_coordinates : np.ndarray
            NxM matrix of x coordinates.
        y_coordinates : np.ndarray
            NxM matrix of y coordinates.
        transform_matrix : np.ndarray
            3x3 (or 2x3) affine transformation matrix to apply to coordinates.
    ---
    Outputs
        new_x : np.ndarray
            NxM matrix of transformed x coordinates.
        new_y : np.ndarray
            NxM matrix of transformed y coordinates.
    '''
    new_x = transform_matrix[0, 0] * x_coordinates + transform_matrix[0, 1] * y_coordinates + \
            transform_matrix[0, 2]
    new_y = transform_matrix[1, 0] * x_coordinates + transform_matrix[1, 1] * y_coordinates + \
            transform_matrix[1, 2]
    return new_x, new_y


def main():
    return 0


if __name__ == '__main__':
    main()
