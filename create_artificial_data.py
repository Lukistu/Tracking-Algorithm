import os

import numpy as np
from scipy.ndimage import rotate
from collections import OrderedDict
from main import LesionAnalysisManager
import itertools
import csv
from datetime import datetime

"""uncomment the seed if you want to get the same random numbers"""


# seed = 42
# np.random.seed(seed)


def assign_lesion_volume():
    """Assign a default volume of 100 to each lesion"""
    return 100


class SyntheticSubjectGenerator:

    def __init__(self):
        self.results = OrderedDict()
        self.subject_index = -1
        self.subject_0 = None
        self.subject_1 = None
        self.pad_width = 3
        self.cube_dims = [70, 70, 70]
        self.conf = {
            'lesions': [10],
            'remove': [0],
            'add': [5],
            'translation': [0],  # translates the lesions along the center-lesion-vector [1,2,3] --> [2,3,4]
            'rotation': [0],  # confusion of the highest order
            'variance': [4],  # randomly subtracts value away from [x,y,z]
        }
        self.permutations = itertools.product(*self.conf.values())

        # Initialize the attributes for tracking prediction accuracy
        self.total_pred_val = 0
        self.evaluation_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        perm = next(self.permutations)
        self.subject_index += 1
        print(f'Generating subject {self.subject_index} with permutation {perm}')
        self.results[self.subject_index] = {}
        for method, value in zip(self.conf.keys(), perm):
            self.results[self.subject_index][method] = value
            if hasattr(self, method):
                if value > 0:
                    getattr(self, method)(value)

        return {
            'subject_0_gt': self.subject_0,
            'subject_0_binary': self.mask(self.subject_0),
            'subject_1_binary': self.mask(self.subject_1)
        }

    def __len__(self):
        return len(list(self.permutations))

    @staticmethod
    def mask(subject):
        """Mask the lesions in the subject to 1"""
        masked = np.zeros_like(subject)
        masked[subject > 0] = 1
        return masked

    def lesions(self, num_of_lesions):
        """Create a synthetic subject with lesions that do not touch each other"""
        if any([dim % 2 != 0 for dim in self.cube_dims]):
            raise ValueError('All dimensions must be divisible by 2')

        mask = np.zeros(np.divide(self.cube_dims, 2).astype(np.int8), dtype=int)

        # Assign lesions to random positions
        lesion_indices = np.arange(1, num_of_lesions + 1)
        all_pos = mask.size
        random_pos = np.random.choice(all_pos, num_of_lesions, replace=False)
        random_pos = np.unravel_index(random_pos, mask.shape)
        mask[random_pos] = lesion_indices

        # Resize the mask to the original size
        subject = np.zeros(self.cube_dims, dtype=int)
        x, y, z = np.indices(mask.shape)
        new_x, new_y, new_z = x * 2, y * 2, z * 2
        subject[new_x, new_y, new_z] = mask
        self.subject_0 = np.pad(subject, self.pad_width, mode='constant', constant_values=0)
        self.subject_1 = np.pad(subject, self.pad_width, mode='constant', constant_values=0)

    def remove(self, num_of_lesions):
        """Remove lesions from the subject"""
        sub_1_unique = np.unique(self.subject_1)[np.unique(self.subject_1) > 0]
        lesions_to_remove = min(num_of_lesions, len(sub_1_unique))
        remove_index = np.random.choice(sub_1_unique, lesions_to_remove, replace=False)
        self.subject_1[np.isin(self.subject_1, remove_index)] = 0
        print(remove_index)

    def add(self, num_of_lesions):
        """Add new lesions to the subject"""
        # Got to mask space
        mask = np.zeros(np.divide(self.cube_dims, 2).astype(np.int8), dtype=int)
        x, y, z = np.indices(mask.shape) * 2
        subject = self.subject_1[
                  self.pad_width:-self.pad_width,
                  self.pad_width:-self.pad_width,
                  self.pad_width:-self.pad_width
                  ]
        mask = subject[x, y, z]

        # Assign new lesions to random zero positions
        zero_indices = np.argwhere(mask == 0)
        highest_lesions_number = np.max(mask)
        lesions_index = np.arange(highest_lesions_number + 1, highest_lesions_number + num_of_lesions + 1)
        random_indices = np.random.choice(len(zero_indices), num_of_lesions, replace=False)
        indices = np.unravel_index(random_indices, mask.shape)
        mask[indices] = lesions_index

        # Resize the mask to the original size
        x, y, z = np.indices(mask.shape)
        new_x, new_y, new_z = x * 2, y * 2, z * 2
        subject[new_x, new_y, new_z] = mask
        self.subject_1 = np.pad(subject, self.pad_width, mode='constant', constant_values=0)

    def translation(self, translation):
        """Translate the lesions in the subject"""
        translation = np.array([translation, translation, translation])
        current_lesions = np.array(np.argwhere(self.subject_1 > 0))
        shape = self.subject_0.shape
        new_lesions = current_lesions + translation
        if np.any(new_lesions < 0) or np.any(new_lesions >= shape):
            print('Lesions out of bounds for translation, subject will be ignored')
            return None

        for current, new in zip(current_lesions, new_lesions):
            if np.not_equal(current, new).any():
                self.subject_1[tuple(new)] = self.subject_1[tuple(current)]
                self.subject_1[tuple(current)] = 0

    def rotation(self, angle):
        """Rotate the lesions in the subject"""
        self.subject_1 = rotate(self.subject_1, angle=angle, axes=(0, 1), reshape=False)

    def variance(self, shift_range):
        """Add variance to the lesions in the subject"""
        current_indices = np.argwhere(self.subject_1 > 0)
        for index in current_indices:
            within_bounds = False
            while not within_bounds:
                vibrations = np.array([
                    np.random.randint(-shift_range, shift_range),
                    np.random.randint(-shift_range, shift_range),
                    np.random.randint(-shift_range, shift_range)
                ])
                new_index = index + vibrations
                if np.all(new_index >= 0) and np.all(new_index < self.subject_1.shape):
                    within_bounds = True

            if np.not_equal(index, new_index).any():
                self.subject_1[tuple(new_index)] = self.subject_1[tuple(index)]
                self.subject_1[tuple(index)] = 0

    # ----------------------------------------------------------------------------------------------------------------

    def evaluation(self):
        """Evaluate the synthetic subjects"""
        # Convert coordinates to the format expected by MultiLesion
        gt_lesions = self.coordinates_to_lesions(self.subject_0)
        pred_lesions = self.coordinates_to_lesions(self.subject_1)

        # Create subject instances with the required store structure
        self.subject_instance_0 = self.create_subject_instance(gt_lesions)
        self.subject_instance_1 = self.create_subject_instance(pred_lesions)

        # check if we have a removal of lesions
        lesion_count_1 = len(self.subject_instance_1.store['lesions'])
        lesion_id_max = max(self.subject_instance_1.store['lesions'])
        if lesion_count_1 != lesion_id_max:
            self.reassign_lesion_number(self.subject_instance_1)

        # Use the LesionAnalysisManager class to process the lesions
        lesion_analysis_manager = LesionAnalysisManager(
            subjects=[self.subject_instance_0, self.subject_instance_1],
            sphere_radius=90,
            mass_factor=1.0,
            center_distance_and_mass_approach=False,
        )

        if visualization:
            lesion_analysis_manager.run()
        else:
            lesion_analysis_manager.process_lesions()

        result = self.check_prediction_per_run(lesion_analysis_manager)
        # Store pred_val in results dictionary under subject_index
        self.results[self.subject_index]['pred_val'] = result
        print(f"\nCorrectly matched lesions: {result}\n\n---------------------------------------------")

        return lesion_analysis_manager, result

    @staticmethod
    def reassign_lesion_number(subject_instance):
        """Reassign the lesion numbers in the subject_instance.store['store']['lesions'] dictionary"""
        lesions = subject_instance.store['lesions']
        new_lesions = {}
        lesion_counter = 1
        for lesion_id, lesion_data in lesions.items():
            new_lesions[lesion_counter] = {
                'center_point': lesion_data['center_point'],
                'num_voxels': lesion_data['num_voxels'],
                'lesion_id': int(lesion_id)
            }
            lesion_counter += 1
        subject_instance.store['lesions'] = new_lesions

    @staticmethod
    def coordinates_to_lesions(subject):
        """Convert coordinates to a lesion dictionary format"""
        lesions = {}
        unique = np.unique(subject[subject != 0])
        for i in unique:
            coord = np.argwhere(subject == i).astype(float)
            # Calculate the mean of the coordinates to get a single center point per lesion
            center_point = coord.mean(axis=0)
            lesions[i] = {
                'center_point': center_point,
                'num_voxels': assign_lesion_volume(),
                'lesion_id': i
            }
        return lesions

    def create_subject_instance(self, lesions):
        """Create a subject instance with the required store structure"""

        class SubjectInstance:
            def __init__(self, lesions, cube_dims, distance_matrices=None):
                self.store = {'lesions': lesions, 'meta': {'distance_matrices': distance_matrices or {}}}
                self.cube_dims = cube_dims

            def centroid_MRI(self):
                cube = self.cube_dims
                return np.array(cube) / 2

        return SubjectInstance(lesions, self.cube_dims)

    def check_prediction_per_run(self, lesion_analysis_manager):
        """Check the prediction for matched lesions"""
        lowest_values = lesion_analysis_manager.store2['analysis']['lowest_values']
        correctness = 0
        lesion_count = max(len(lesion_analysis_manager.store1['lesions']),
                           len(lesion_analysis_manager.store2['lesions']))

        # get matching lesions
        matches = {}
        for match in lowest_values:
            lesion_id_0, lesion_id_1, _, _ = match
            matches[lesion_id_0 + 1] = lesion_id_1 + 1

        # Calculate correctness of matched lesions
        for lesion_id_0, lesion_id_1 in matches.items():
            if lesion_id_0 in lesion_analysis_manager.store1['lesions'] and lesion_id_1 in \
                    lesion_analysis_manager.store2['lesions']:
                if lesion_analysis_manager.store1['lesions'][lesion_id_0]['lesion_id'] == \
                        lesion_analysis_manager.store2['lesions'][lesion_id_1]['lesion_id']:
                    correctness += 1

        # Calculate correctness of new lesions
        max_lesion_id_subject_0 = max(lesion_analysis_manager.store1['lesions'].keys(), default=0)
        for lesion_id_1 in lesion_analysis_manager.store2['lesions'].keys():
            if lesion_id_1 > max_lesion_id_subject_0:
                correctness += 1

        # Calculate correctness of lost lesions
        lost_lesions = set(lesion_analysis_manager.store1['lesions'].keys()) - set(matches.keys())
        lowest_values_sub_1 = [value[0] + 1 for value in lowest_values]
        for lost_lesion_id in lost_lesions:
            # Adjust indexing for comparison
            if lost_lesion_id not in lowest_values_sub_1:
                correctness += 1

        # Update the prediction value
        pred_val = correctness / lesion_count

        # Update the overall prediction accuracy
        self.prediction_accuracy_over_all(pred_val)

        return pred_val

    def prediction_accuracy_over_all(self, pred_val):
        self.evaluation_count += 1
        self.total_pred_val += pred_val

        overall_accuracy = self.total_pred_val / self.evaluation_count

        return overall_accuracy

        # ---------------------------------------------------------------------------------------------------------------

    def to_csv(self, iteration, path=None):
        """Save the results to a CSV file"""
        if path is None:
            path = x  # todo: set a path to where you want to save the csv files

        csv_data = []
        header_row = ['Iteration', 'Subject Index'] + list(self.conf.keys()) + ['pred_val']
        csv_data.append(header_row)

        for row_key, row_data in self.results.items():
            conf_values = [row_data.get(col, 0) for col in self.conf.keys()]
            row = [iteration, row_key] + conf_values + [row_data.get('pred_val', '')]
            csv_data.append(row)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = os.path.join(path, f'output_{timestamp}_iteration_{iteration}.csv')
        with open(csv_filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(csv_data)

        print(f'CSV file saved as {csv_filename}')


if __name__ == '__main__':
    visualization = True
    x = 1  # Set the number of iterations to run

    for i in range(x):
        gen = SyntheticSubjectGenerator()
        for sub in gen:
            lesion_analysis_manager, result = gen.evaluation()  # Capture both return values
            pred = gen.check_prediction_per_run(lesion_analysis_manager)
            accuracy = gen.prediction_accuracy_over_all(pred)
            print(f'---------------------------------------------\n Overall tracking accuracy: {accuracy:.2%}\n'
                  f'---------------------------------------------\n---------------------------------------------')
        gen.to_csv(i)
