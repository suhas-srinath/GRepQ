"""
A subset of the LIVE FB dataset is used for training.
LIVE_FB_synthetic has 5k images, 4 distortion types of 2 levels each (8 distortions per image)

Directory structure for synthetic LIVE_FB:
    - image_xyz/image_xyz_distortion_level.bmp
"""


from pathlib import Path
import pandas as pd
import os

def get_dataset_list(base_dataset_path: Path, dataset: str):
    """
    Get names, paths, scores and frames for synthetic video datasets.
    :param base_dataset_path: Path to the base dataset folder.
    :param dataset: Name of the dataset.
    :return: A dictionary containing image names and paths.

    Paths indicates- path to the folder containing distorted versions of the same image. (See above for directory structure)
    """

    if dataset == 'LIVE_FB_synthetic':
        names = []
        paths = []

        curr_path = os.path.join(base_dataset_path, 'LIVE_FB_synthetic')
        images_list = os.path.join(curr_path, r'LIVEFB.csv')
        loaded_data = pd.read_csv(images_list)
        names_list = list(loaded_data['im_loc'])

        for curr_name in names_list:
            folder_name = curr_name.split('/')[-1].split('.')[0]
            folder_name = folder_name + '.bmp'
            folder_path = os.path.join(curr_path, folder_name)
            if os.path.exists(folder_path):
                names.append(folder_name)
                paths.append(folder_path)

    return {'names': names, 'image_paths': paths}