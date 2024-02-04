import numpy as np

def SE2_from_R3(R3):    
    x = R3[0]
    y = R3[1]
    theta = R3[2]
    return np.array([
                    [np.cos(theta), -np.sin(theta), x],
                    [np.sin(theta), np.cos(theta), y],
                    [0, 0, 1]
                    ])


def compute_error(trajectory, gt):
    
    assert len(trajectory) == len(gt)

    rotation_errors = []
    translation_errors = []

    for i in range(len(trajectory) - 1):
        
        rel_t = np.dot(np.linalg.inv(SE2_from_R3(trajectory[i])) , SE2_from_R3(trajectory[i+1]))
        rel_gt = np.dot(np.linalg.inv(SE2_from_R3(gt[i])) , SE2_from_R3(gt[i+1]))
        error_t = np.dot(np.linalg.inv(rel_t), rel_gt)
        rotation_errors.append(np.arctan2(error_t[1,0], error_t[0, 0]))
        translation_errors.append(np.sqrt(error_t[0, 2]**2 + error_t[1,2]**2))

    return rotation_errors, translation_errors
