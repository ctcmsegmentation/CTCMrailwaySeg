
import cv2
import numpy as np
from scipy.ndimage import binary_dilation

from copy import deepcopy



class sem_cc:
    def __init__(self):
        pass

    def remove_co_occurrences(self, binary_matrix, skeleton, edges, epsilon):
        skeleton_dilated = binary_dilation(skeleton, iterations=epsilon)
        edges_dilated = binary_dilation(edges, iterations=epsilon)
        
        mask = (skeleton_dilated == 1) & (edges_dilated == 1)
        binary_matrix[mask] = 0
        
        return binary_matrix
    
    def morphology_operations(self, binary_matrix):
        binary_matrix = cv2.morphologyEx(binary_matrix, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        binary_matrix = cv2.morphologyEx(binary_matrix, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))
        binary_matrix = cv2.erode(binary_matrix, np.ones((10, 10), np.uint8), iterations=1)
        binary_matrix = cv2.dilate(binary_matrix, np.ones((10, 10), np.uint8), iterations=1)
        return binary_matrix
    
    def get_edges(self, binary_matrix):
        postprocess_matrix = cv2.erode(binary_matrix, np.ones((5, 5), np.uint8), iterations=1)
        edges = binary_matrix - postprocess_matrix
        return edges
    
    def get_skeleton(self, binary_matrix):
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        # Create an empty output image to hold values
        thin = np.zeros(binary_matrix.shape,dtype='uint8')
        
        # Loop until erosion leads to an empty set
        while (cv2.countNonZero(binary_matrix)!=0):
            # Erosion
            erode = cv2.erode(binary_matrix, kernel)
            # Opening on eroded image
            opening = cv2.morphologyEx(erode, cv2.MORPH_OPEN,kernel)
            # Subtract these two
            subset = erode - opening
            # Union of all previous sets
            thin = cv2.bitwise_or(subset,thin)
            # Set the eroded image for next iteration
            binary_matrix = erode.copy()

        return thin
    
    def smooth(self, binary_matrix, kernel_size=(5, 5), iterations=2):
        binary_matrix = cv2.erode(binary_matrix, np.ones(kernel_size, np.uint8), iterations=iterations)
        binary_matrix = cv2.dilate(binary_matrix, np.ones(kernel_size, np.uint8), iterations=iterations)
        return binary_matrix
    
    def process(self, binary_matrix, return_all=False): 

        result_mask = deepcopy(binary_matrix)
        input_binary_matrix_rails = (deepcopy(binary_matrix) == 0)

        binary_matrix[binary_matrix==0] = 1
        binary_matrix[binary_matrix==2] = 0
        
        skeleton = self.get_skeleton(binary_matrix)
        edges = self.get_edges(binary_matrix)
        result_co_occ_removed = self.remove_co_occurrences(binary_matrix, skeleton, edges, 10)
        result = self.morphology_operations(result_co_occ_removed)
        result = self.smooth(result)

        skeleton = np.array(skeleton, dtype=np.uint8)

        result_mask[result==0] = 2
        result_mask[result==1] = 1
        result_mask[input_binary_matrix_rails & (result_mask==1)] = 0

        best_component = self.filter_by_rails(result_mask)
        result_mask[best_component==0] = 2

        if return_all:
            return result_mask, skeleton, edges, result_co_occ_removed, result, best_component
        return result_mask
    
    def filter_by_rails(self, segmentation_matrix):
        binary_matrix = deepcopy(segmentation_matrix)
        binary_matrix[binary_matrix==0] = 1
        binary_matrix[binary_matrix==2] = 0
        binary_matrix = cv2.erode(binary_matrix, np.ones((10, 10), np.uint8), iterations=2)
        num_labels, labels = cv2.connectedComponents(binary_matrix)
        best_mask_score = [np.array([]), -1]
        for label in range(1, num_labels):
            component_mask = np.uint8(labels == label)
            actual_score = np.count_nonzero(segmentation_matrix[component_mask==1]==0)
            if best_mask_score[1] < actual_score:
                best_mask_score = [component_mask, actual_score]
        result_component = cv2.dilate(best_mask_score[0], np.ones((10, 10), np.uint8), iterations=4)
        result_component = cv2.dilate(result_component, np.ones((10, 10), np.uint8), iterations=2)
        return result_component
