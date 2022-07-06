# This code is provided to assist with creating affine matrices to describe transformations or
# poses in the camera or projector frame of the DOME. The DOME (Dynamic Optical Micro Environment)
# was developed by Ana Rubio Denniss. To access the functions in custom scripts, ensure that a copy
# of this file is stored in the same directory. Then run the following command to import the code
# at the beginning of the custom files:
#     import DOME_transformation as DOMEtran
# #################################################################################################
# Authors = Matthew Uppington <mu15531@bristol.ac.uk>
# Affiliation = Farscope CDT, University of Bristol, University of West England
# #################################################################################################
# This work is licensed under a Creative Commons Attribution 4.0 International License.

import numpy as np
import cv2


def format_affine(vectors_matrix : np.ndarray):
    '''
    Extend standard format vectors in to affine format.
    ---
    Parameters
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
    Parameters
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
    Parameters
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
    Parameters
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
    new_image = cv2.warpAffine(image, compatible_transform[0:2, :], [dimensions[1], dimensions[0]])
    return new_image

def transform_matrix(scale=None, shear=None, shift=None):
    '''
    Create affine transformation matrix for stretches, shears and translations.
    ---
    Optional Parameters
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

def rotation_matrix(angle=0, centre=None):
    '''
    Create affine transformation matrix to rotate about a point.
    ---
    Parameters
        angle : float
            Number of radians to rotate by.
    ---
    Optional Parameters
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
    Parameters
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

#np.savetxt(file_name, matrix), np.loadtxt(filename)

#def transform_image(image : np.ndarray, trans_mat):
#    """
#    Transform an image {image} according to the affine transformation matrix, {trans_mat}.
#    """
#    src_points = np.array([[500, 300], [1000, 300], [500, 800]]).astype(np.float32)
#>>> dst_points = np.array([[420, 160], [750, 10], [450, 450]]).astype(np.float32)
#>>> mapping = cv2.getAffineTransform(src_points, dst_points)
#>>> warp_noise_image = cv2.warpAffine(noise_image, mapping, [854, 480])
#>>> cv2.imshow('warpped noise', warp_noise_image)
#>>> cv2.waitKey(0)
    