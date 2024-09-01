import copy

import SimpleITK as sitk
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import heapq
import numpy as np
from loguru import logger


class Subject:

    def __init__(self, src):
        self.src = src
        self.store = {"lesions": {}, "meta": {}}
        self.seg_mask_sitk = sitk.ReadImage(src)
        self.size = self.seg_mask_sitk.GetSize()
        self.connected_componentes()
        self.centroids()

    def __iter__(self) -> tuple[int, dict]:
        for id, values in self.store.items():
            yield id, values

    def __len__(self):
        return len(self.store)

    def connected_componentes(self):
        """Get the connected components of the image"""

        # todo: decide if you want to use dilation and erosion
        """Single voxels within a cluster would count as stand-alone lesion if dilation and erosion = 0"""
        # Perform dilation
        dilation_filter = sitk.BinaryDilateImageFilter()
        dilation_filter.SetKernelRadius(0)
        tmp = copy.deepcopy(self.seg_mask_sitk)
        cc_dilated_image = dilation_filter.Execute(tmp)

        cc_filter = sitk.ConnectedComponentImageFilter()
        cc_filter.SetFullyConnected(True)  # Setting to True is less restrictive, gives fewer connected components

        binary_img_sitk = sitk.BinaryThreshold(image1=cc_dilated_image,
                                               lowerThreshold=0,
                                               upperThreshold=0.5,
                                               insideValue=0,
                                               outsideValue=1)
        lesions = cc_filter.Execute(binary_img_sitk)  # connected components

        rl_filter = sitk.RelabelComponentImageFilter()
        lesions = rl_filter.Execute(lesions)  # sort lesions by size

        if rl_filter.GetNumberOfObjects() == 0:
            logger.warning('No connected components found')

        for label, _ in enumerate(rl_filter.GetSizeOfObjectsInPixels(), start=1):
            component = sitk.BinaryThreshold(lesions, label, label, 1, 0)

            # Perform erosion (! Original segmentation got altered !)
            _filter = sitk.BinaryErodeImageFilter()
            _filter.SetKernelRadius(0)
            component = _filter.Execute(component)

            # Calculate number of voxels
            num_voxels = sitk.GetArrayFromImage(component).sum()

            # Store component and its size
            self.store["lesions"][label] = {'component_sitk': component, 'num_voxels': num_voxels}

    def centroids(self):
        """Calculate the center of mass of the image"""
        # id is the Lesion identification, values are the Voxels

        for label, values in self.store["lesions"].items():
            component = copy.deepcopy(values['component_sitk'])  # Copy the component to avoid changing the original
            origin = component.GetOrigin()
            direction = component.GetDirection()

            component.SetOrigin((0, 0, 0))  # Set the origin to 0,0,0, centroid is calculated from the origin
            component.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))  # Set orientation to identity matrix -> centroid

            statistics = sitk.LabelShapeStatisticsImageFilter()
            statistics.Execute(component)

            centroid_img_sitk = sitk.Image(component.GetSize(), sitk.sitkUInt8)
            centroid_img_sitk.CopyInformation(component)

            center_point = statistics.GetCentroid(label=1)
            center_point = tuple([int(point) for point in center_point])
            centroid_value = 1
            centroid_img_sitk.SetPixel(center_point[0],
                                       center_point[1],
                                       center_point[2],
                                       centroid_value)

            centroid_img_sitk.SetOrigin(origin)
            centroid_img_sitk.SetDirection(direction)

            self.store["lesions"][label].update({
                'centroid_sitk': centroid_img_sitk,
                'center_point': center_point
            })

    def centroid_MRI(self):
        """Calculate the center of mass of the brain relative to the origin (0,0,0)"""

        original_path = self.src
        modified_path = original_path.replace('seg_', 'img_')

        brain_sitk = sitk.ReadImage(modified_path)
        brain = copy.deepcopy(brain_sitk)  # Copy the component to avoid changing the original

        # Set origin and direction
        brain.SetOrigin((0, 0, 0))
        brain.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

        np_array = sitk.GetArrayFromImage(brain)
        indices = np.nonzero(np_array)
        brain_center = np.mean(indices, axis=1) * np.array(brain.GetSpacing())

        return brain_center


class MultiLesion:

    def __init__(self, subject_instance, sphere_radius):
        self.subject = subject_instance
        self.sphere_radius = sphere_radius
        self.store = {"lesions": {}, "meta": {}}  # Initialize a new dictionary for each instance
        self.store.update(self.subject.store)

    def calculate_vectors(self):
        direction_vectors = {}
        centroids = {label: values['center_point'] for label, values in self.store["lesions"].items()}
        for label1, centroid1 in centroids.items():
            for label2, centroid2 in centroids.items():
                if label1 < label2:
                    direction_vector = np.array(centroid2) - np.array(centroid1)
                    # Store the direction vector with a key indicating the pair of lesions
                    direction_vectors[(label1, label2)] = {
                        'connecting': f'{label1}-{label2}',
                        'vector': direction_vector
                    }
        self.store["direction_vectors"] = direction_vectors

    def get_intersect_points(self):
        intersect_points = {}
        brain_center = self.subject.centroid_MRI()
        self.store["meta"].update({'brain_center': brain_center})

        for key, value in self.store["direction_vectors"].items():
            vector = np.array(value['vector'])
            point_1 = self.store["lesions"][key[0]]['center_point']

            # Adjust vector origin by subtracting the sphere center
            vector_origin = point_1 - self.store["meta"]["brain_center"]

            # Calculate coefficients of the quadratic equation
            a = np.dot(vector, vector)
            b = 2 * np.dot(vector_origin, vector)
            c = np.dot(vector_origin, vector_origin) - self.sphere_radius ** 2

            # Solve quadratic equation
            discriminant = b ** 2 - 4 * a * c

            if discriminant < 0:
                continue  # Skip if there is no real intersection

            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2 * a)
            t2 = (-b + discriminant) / (2 * a)

            # Intersection points in the coordinate system of the sphere
            intersect_1 = vector_origin + t1 * vector + self.store["meta"]["brain_center"]
            intersect_2 = vector_origin + t2 * vector + self.store["meta"]["brain_center"]

            intersect_points[key] = (intersect_1, intersect_2)

            self.store["intersect_points"] = intersect_points

    def get_distances(self):
        centroids = {label: values['center_point'] for label, values in self.store["lesions"].items()}

        inter_centroid_distances = {}
        for label1, centroid1 in centroids.items():
            for label2, centroid2 in centroids.items():
                if label1 < label2:  # Ensure only one distance per pair
                    distance = np.linalg.norm(np.array(centroid2) - np.array(centroid1))
                    inter_centroid_distances[(label1, label2)] = distance

        distance_to_center = {}
        brain_center = np.array(self.store["meta"]["brain_center"])
        for label, centroid in centroids.items():
            distance = np.linalg.norm(brain_center - centroid)
            distance_to_center[(label, "center")] = distance

        # Calculate distances from lesions to sphere intersections along the vector
        sphere_intersection_distances = {}
        for key, (intersect_1, intersect_2) in self.store["intersect_points"].items():
            point_1 = self.store["lesions"][key[0]]['center_point']
            point_2 = self.store["lesions"][key[1]]['center_point']

            dist1_1 = np.linalg.norm(point_1 - intersect_1)
            dist1_2 = np.linalg.norm(point_1 - intersect_2)
            dist2_1 = np.linalg.norm(point_2 - intersect_1)
            dist2_2 = np.linalg.norm(point_2 - intersect_2)

            min_dist1 = min(dist1_1, dist1_2)
            min_dist2 = min(dist2_1, dist2_2)

            sphere_intersection_distances[(key[0], key[1])] = min(min_dist1, min_dist2)
            sphere_intersection_distances[(key[1], key[0])] = max(min_dist1, min_dist2)

        self.store["vectors_distances"] = {
            'direction_vectors': self.store["direction_vectors"],
            'inter_centroid_distances': inter_centroid_distances,
            'distance_to_center': distance_to_center,
            'sphere_intersection_distances': sphere_intersection_distances
        }

    def create_distance_matrices(self):

        global distance_matrix
        num_lesions = len(self.store["lesions"])

        # Initialize or update the distance_matrices key in the meta dictionary
        if "distance_matrices" not in self.store["meta"]:
            self.store["meta"]["distance_matrices"] = {}

        for centroid_index in range(1, num_lesions + 1):
            matrix_name = f"matrix of lesion {centroid_index}"
            distance_matrix = np.zeros((num_lesions - 1, 3))
            row_index = 0

            # Iterate over all lesions in the subject
            for lesion_index, data in self.store["lesions"].items():
                if lesion_index != centroid_index:
                    # Get the distance data between the centroid and the current lesion
                    i1 = centroid_index
                    i2 = lesion_index

                    key = (i1 if i1 < i2 else i2, i2 if i2 > i1 else i1)
                    inter_centroid_distance = self.store["vectors_distances"]["inter_centroid_distances"].get(key, 0)
                    distance_matrix[row_index, 0] = inter_centroid_distance

                    sphere_intersection_distance = self.store["vectors_distances"]["sphere_intersection_distances"].get(
                        key, 0)
                    distance_matrix[row_index, 1] = sphere_intersection_distance

                    key2 = (i1 if i1 > i2 else i2, i2 if i2 < i1 else i1)
                    reverse_sphere_intersection_distance = self.store["vectors_distances"][
                        "sphere_intersection_distances"].get(key2, 0)
                    distance_matrix[row_index, 2] = reverse_sphere_intersection_distance

                    row_index += 1

            self.store["meta"]["distance_matrices"][matrix_name] = distance_matrix

        # Append a row to each distance matrix with the distance to the center
        for centroid_index in range(1, num_lesions + 1):
            matrix_name = f"matrix of lesion {centroid_index}"
            matrix = self.store["meta"]["distance_matrices"][matrix_name]

            # Create a row with the distance to the center
            distance_to_center = self.store["vectors_distances"]["distance_to_center"][(centroid_index, "center")]

            """Use the distance to the center either once or three times in the matrix. There is no clear advantage 
            to either approach, though some results suggest that using the distance only once may be slightly better."""

            # new_row = np.array([[distance_to_center] * 3])
            new_row = np.array([0, 0, distance_to_center])

            # Append the new row to the matrix
            updated_matrix = np.vstack([matrix, new_row])
            self.store["meta"]["distance_matrices"][matrix_name] = updated_matrix


class SingleLesion:
    def __init__(self, subject_instance, sphere_radius):
        self.subject = subject_instance
        self.sphere_radius = sphere_radius
        self.store = {"lesions": {}, "meta": {}}
        self.store.update(self.subject.store)

    def get_distances(self):
        brain_center = self.subject.centroid_MRI()
        self.store["meta"].update({'brain_center': brain_center})

        center_lesion_distances = {}

        for label, lesion_data in self.store["lesions"].items():
            lesion_center = np.array(lesion_data['center_point'])
            center_lesion_distance = np.linalg.norm(lesion_center - brain_center)
            center_lesion_distances[label] = center_lesion_distance

        self.store["center_lesion_distances"] = center_lesion_distances

    def create_single_lesion_distance_matrix(self, mass_factor):

        num_lesions = len(self.store["lesions"])
        self.store["meta"]["distance_matrices"] = {}

        # 1 / 300 is the number that mostly sets the volume to a range of the average distance value giving it similar
        # importance in the matching process. If you want to improve the single lesion matching, find a more
        # sophisticated method for lesion mass
        lesion_mass_factor = (1 / 300) * mass_factor

        for centroid_index in range(1, num_lesions + 1):
            vector_name = f"single to multi matrix of lesion {centroid_index}"

            distance_vector = np.zeros((2, 3))

            lesion_data = self.store["lesions"][centroid_index]

            # Lesion mass (num_voxels)
            lesion_mass = lesion_data['num_voxels']
            distance_vector[0, 2] = lesion_mass * lesion_mass_factor

            # Center to lesion distance
            center_lesion_distance = self.store["center_lesion_distances"][centroid_index]
            distance_vector[1, :] = center_lesion_distance

            self.store["meta"]["distance_matrices"][vector_name] = distance_vector

        for matrix_name, matrix in self.store["meta"]["distance_matrices"].items():
            print(f"Distance matrix for {matrix_name}:\n{matrix}")


class AnalyzerforLesion:
    def __init__(self, subject_instance, lesion_analysis_manager):
        self.subject = subject_instance
        self.store1 = lesion_analysis_manager.store1
        self.store2 = lesion_analysis_manager.store2

    def find_lowest_values(self, subject1, subject2):
        """Iterate through the matrices. Compare each matrix of a subject with all matrices of the other subject
        to find the most similar matrices."""

        lowest_values = []

        # Determine the smaller subject
        if len(subject1) == len(subject2):
            subject_a, subject_b = subject1, subject2
            swap_indices = False
        elif len(subject1) > len(subject2):
            subject_a, subject_b = subject2, subject1
            swap_indices = True
        else:
            subject_a, subject_b = subject1, subject2
            swap_indices = False

        for idx1, matrix1 in enumerate(subject_a):
            smallest_distance = None
            smallest_comparisons = []

            for idx3, matrix2 in enumerate(subject_b):
                total_distance, relative_distance = self.compare_matrices(matrix1, matrix2)
                if smallest_distance is None or total_distance < smallest_distance:
                    smallest_distance = total_distance
                    smallest_comparisons = [(idx1, idx3, total_distance, relative_distance)]
                elif total_distance == smallest_distance:
                    smallest_comparisons.append((idx1, idx3, total_distance, relative_distance))

            lowest_values.extend(smallest_comparisons)

        # If subject1 is bigger, adjust the indices in the lowest_values list
        if swap_indices:
            lowest_values = [(idx3, idx1, total_distance, relative_distance) for
                             idx1, idx3, total_distance, relative_distance in lowest_values]

        # Check for double assignments
        if self.find_double_assignment(lowest_values):
            lowest_values = self.resolve_double_assignments(lowest_values, subject_a, subject_b)

        for subject_instance in [subject1, subject2]:
            if subject_instance == self.subject:
                self.store1.setdefault("analysis", {}).update({"lowest_values": lowest_values})
            else:
                self.store2.setdefault("analysis", {}).update({"lowest_values": lowest_values})

        return lowest_values

    @staticmethod
    def find_double_assignment(lowest_values):
        """Check for double assignments in the lowest_values list."""
        first_numbers = set()
        second_numbers = set()

        for value in lowest_values:
            first_num, second_num = value[0], value[1]
            if first_num in first_numbers or second_num in second_numbers:
                return True
            first_numbers.add(first_num)
            second_numbers.add(second_num)

        return False

    def resolve_double_assignments(self, lowest_values, subject_a, subject_b):
        """Resolve double assignments and find new correspondences."""
        subject1_counter = Counter()
        subject2_counter = Counter()

        # Count occurrences of each index in the lowest values
        for idx1, idx3, _, _ in lowest_values:
            subject1_counter[idx1] += 1
            subject2_counter[idx3] += 1

        # Identify lesions that are double assigned
        double_assigned_lesions_sub1 = [lesion for lesion, count in subject1_counter.items() if count > 1]
        double_assigned_lesions_sub2 = [lesion for lesion, count in subject2_counter.items() if count > 1]

        # Resolve double assignments for subject 1
        for lesion in double_assigned_lesions_sub1:
            matches = [(idx1, idx3, total_distance, relative_distance)
                       for idx1, idx3, total_distance, relative_distance in lowest_values
                       if idx1 == lesion]
            if matches:
                # Find the match with the minimum total distance
                best_match = matches[0]
                for match in matches[1:]:
                    if match[2] < best_match[2]:  # Compare by total_distance (third item)
                        best_match = match

                # Filter out non-best matches
                lowest_values = [match for match in lowest_values if not (match[0] == lesion and match != best_match)]

        # Resolve double assignments for subject 2
        for lesion in double_assigned_lesions_sub2:
            matches = [(idx1, idx3, total_distance, relative_distance)
                       for idx1, idx3, total_distance, relative_distance in lowest_values
                       if idx3 == lesion]
            if matches:
                # Find the match with the minimum total distance
                best_match = matches[0]
                for match in matches[1:]:
                    if match[2] < best_match[2]:  # Compare by total_distance (third item)
                        best_match = match

                # Filter out non-best matches
                lowest_values = [match for match in lowest_values if not (match[1] == lesion and match != best_match)]

        # Find new correspondences for lesions that lost their match
        if len(subject_a) > len(subject2_counter):
            available_lesions1 = set(range(len(subject_a))) - {idx1 for idx1, _, _, _ in lowest_values}
            available_lesions2 = set(range(len(subject_b))) - {idx3 for _, idx3, _, _ in lowest_values}
        else:
            available_lesions1 = set(range(len(subject_b))) - {idx1 for idx1, _, _, _ in lowest_values}
            available_lesions2 = set(range(len(subject_a))) - {idx3 for _, idx3, _, _ in lowest_values}

        while available_lesions1 and available_lesions2:
            distances = []

            if len(subject_a) > len(subject2_counter):
                for lesion in available_lesions1:
                    for idx3 in available_lesions2:
                        total_distance, relative_distance = self.compare_matrices(subject_a[lesion], subject_b[idx3])
                        distances.append((lesion, idx3, total_distance, relative_distance))
            else:
                for lesion in available_lesions2:
                    for idx3 in available_lesions1:
                        total_distance, relative_distance = self.compare_matrices(subject_a[lesion], subject_b[idx3])
                        distances.append((lesion, idx3, total_distance, relative_distance))

            if not distances:
                break

            # Find the minimum based on total distance
            best_match = distances[0]
            for match in distances[1:]:
                if match[2] < best_match[2]:  # Compare by total_distance (third item)
                    best_match = match

            if len(subject_a) > len(subject2_counter):
                if best_match[0] in available_lesions1 and best_match[1] in available_lesions2:
                    lowest_values.append(best_match)
                    available_lesions1.remove(best_match[0])
                    available_lesions2.remove(best_match[1])
            else:
                if best_match[0] in available_lesions2 and best_match[1] in available_lesions1:
                    best_match = (best_match[1], best_match[0], best_match[2], best_match[3])
                    lowest_values.append(best_match)
                    available_lesions1.remove(best_match[0])
                    available_lesions2.remove(best_match[1])

        return lowest_values

    @staticmethod
    def compare_matrices(matrix1, matrix2):
        """Compare two matrices and find the smallest distance for each row of the smaller matrix to any row in the
        larger matrix."""
        # Determine the smaller and larger matrices
        if len(matrix1) <= len(matrix2):
            smaller_matrix, larger_matrix = matrix1, matrix2
        else:
            smaller_matrix, larger_matrix = matrix2, matrix1

        row_distances = []
        distance_heap = []

        for idx1, row1 in enumerate(smaller_matrix):
            row1_array = np.array(row1)
            for idx2, row2 in enumerate(larger_matrix):
                row2_array = np.array(row2)
                distance = np.sqrt(np.sum((row1_array - row2_array) ** 2))
                heapq.heappush(distance_heap, (distance, idx1, idx2))

        assigned_rows_smaller = set()
        assigned_rows_larger = set()
        total_distance = 0

        while distance_heap and len(assigned_rows_smaller) < len(smaller_matrix):
            distance, idx1, idx2 = heapq.heappop(distance_heap)
            if idx1 not in assigned_rows_smaller and idx2 not in assigned_rows_larger:
                assigned_rows_smaller.add(idx1)
                assigned_rows_larger.add(idx2)
                total_distance += distance
                row_distances.append((idx1, idx2, distance))

        relative_distance = total_distance / len(smaller_matrix)

        return total_distance, relative_distance

    def calc_volumetric_change(self, subject_1, subject_2):
        """Calculate the volumetric change between two lesions."""
        initial_volume = self.store1["lesions"][subject_1 + 1].get("num_voxels")
        new_volume = self.store2["lesions"][subject_2 + 1].get("num_voxels")
        volumetric_change = new_volume / initial_volume if initial_volume != 0 else float('inf')
        return volumetric_change

    def matching_output(self, lowest_values):
        matches = []

        # Iterate over the lowest_values to format the output
        for subject_1, subject_2, _, relative_distance in lowest_values:
            pair = f"Sub.1 lesion {subject_1 + 1} -> Sub.2 lesion {subject_2 + 1}"
            volumetric_change = self.calc_volumetric_change(subject_1, subject_2)
            matches.append(f"{pair}  Relative Distance: {relative_distance:.2f}      Volumetric-change:"
                           f" {volumetric_change:.2f} times bigger")

        return matches

    def new_lesion_output(self, lowest_values):
        new_lesions = []
        second_values = [value[1] + 1 for value in lowest_values]

        for lesion in self.store2['lesions']:
            if lesion not in second_values:
                lesion_volume = self.store2['lesions'][lesion]['num_voxels']
                new_lesions.append(f"Sub.2 Lesion {lesion} - Volume: {lesion_volume:.2f}")

        return new_lesions

    def lost_lesion_output(self, lowest_values):
        lost_lesions = []
        second_values = [value[0] + 1 for value in lowest_values]

        for lesion in self.store1['lesions']:
            if lesion not in second_values:
                lesion_volume = self.store1['lesions'][lesion]['num_voxels']
                lost_lesions.append(f"Sub.1 Lesion {lesion} - Volume: {lesion_volume:.2f}")

        return lost_lesions

    def calc_centroid_movement(self, lowest_values):
        centroid_movement_tot, max_movement = 0, 0
        centroid_movement_x_tot, centroid_movement_y_tot, centroid_movement_z_tot = 0, 0, 0
        max_mv_x, max_mv_y, max_mv_z = 0, 0, 0
        lesion_counter = 0

        for lesion1, lesion2, _, _ in lowest_values:
            lesion_1 = self.store1['lesions'].get(lesion1 + 1)
            lesion_2 = self.store2['lesions'].get(lesion2 + 1)

            if lesion_1 is not None and lesion_2 is not None:
                lesion_1_center = np.array(lesion_1['center_point'])
                lesion_2_center = np.array(lesion_2['center_point'])

                # Calculate total movement
                centroid_movement = np.linalg.norm(lesion_2_center - lesion_1_center)
                centroid_movement_tot += centroid_movement
                if centroid_movement > max_movement:
                    max_movement = centroid_movement

                # Calculate movement in each direction
                centroid_movement_x = abs(lesion_2_center[0] - lesion_1_center[0])
                centroid_movement_y = abs(lesion_2_center[1] - lesion_1_center[1])
                centroid_movement_z = abs(lesion_2_center[2] - lesion_1_center[2])
                centroid_movement_x_tot += centroid_movement_x
                centroid_movement_y_tot += centroid_movement_y
                centroid_movement_z_tot += centroid_movement_z

                # Update maximum movement in each direction
                if centroid_movement_x > max_mv_x:
                    max_mv_x = centroid_movement_x
                if centroid_movement_y > max_mv_y:
                    max_mv_y = centroid_movement_y
                if centroid_movement_z > max_mv_z:
                    max_mv_z = centroid_movement_z

                lesion_counter += 1

        # Calculate average total movement
        average_movement = centroid_movement_tot / lesion_counter if lesion_counter > 0 else 0

        return average_movement, max_movement, max_mv_x, max_mv_y, max_mv_z


class Visualisation:
    def __init__(self, store1, store2, sphere_radius):
        self.store1 = store1
        self.store2 = store2
        self.sphere_radius = sphere_radius
        self.color_mapping = {}

    def _create_color_mapping(self, acceptance_level):
        colors = [
            '#1f77b4',  # blue
            '#ff7f0e',  # orange
            '#2ca02c',  # green
            '#d62728',  # red
            '#9467bd',  # purple
            '#8c564b',  # brown
            '#e377c2',  # pink
            '#000000',  # black
            '#bcbd22',  # yellow-green
            '#17becf',  # cyan
        ]

        color_mapping = {}
        color_idx = 0
        added_pairs = set()  # Track added pairs to handle symmetry

        lowest_values = self.store2['analysis']['lowest_values']

        for subject1, subject2, total_distance, relative_distance in lowest_values:
            if relative_distance <= acceptance_level:
                pair = (subject1 + 1, subject2 + 1)
                if pair not in added_pairs:
                    color = colors[color_idx % len(colors)]
                    color_mapping[pair] = color
                    added_pairs.add(pair)
                    color_idx += 1

        return color_mapping

    def _plot_subject(self, ax, store, brain_center, subject_label):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_sphere = self.sphere_radius * np.outer(np.cos(u), np.sin(v)) + brain_center[0]
        y_sphere = self.sphere_radius * np.outer(np.sin(u), np.sin(v)) + brain_center[1]
        z_sphere = self.sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + brain_center[2]
        ax.plot_surface(x_sphere, y_sphere, z_sphere, color='lightgray', alpha=0.2)

        legend_handles = []
        for label, values in store["lesions"].items():
            color = 'yellow'  # Default color
            for key, mapped_color in self.color_mapping.items():
                lesion_tp1, lesion_tp2 = key
                if label == lesion_tp1 and subject_label == 1:
                    color = mapped_color
                if label == lesion_tp2 and subject_label == 2:
                    color = mapped_color
                    break

            centroid = values['center_point']
            ax.scatter(centroid[0], centroid[1], centroid[2], color=color)
            legend_handles.append(mpatches.Patch(color=color, label=f'Lesion {label}'))

        ax.scatter(brain_center[0], brain_center[1], brain_center[2], color='r', marker='x', label='Brain Center')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        return legend_handles

    def plot(self, fig, store, brain_center, subject_label):
        ax = fig.add_subplot(111, projection='3d')
        legend_handles = self._plot_subject(ax, store, brain_center, subject_label)
        ax.legend(handles=legend_handles)
        ax.set_title(f'Subject {subject_label} Lesion Centroids')

        return ax

    def update_plot(self, acceptance_level, ax1, ax2):
        self.color_mapping = self._create_color_mapping(acceptance_level)

        ax1.clear()
        ax2.clear()

        brain_center1 = self.store1["meta"]["brain_center"]
        subject_label1 = 1
        legend_handles1 = self._plot_subject(ax1, self.store1, brain_center1, subject_label1)
        ax1.legend(handles=legend_handles1)
        ax1.set_title(f'Subject {subject_label1} Lesion Centroids')

        brain_center2 = self.store2["meta"]["brain_center"]
        subject_label2 = 2
        legend_handles2 = self._plot_subject(ax2, self.store2, brain_center2, subject_label2)
        ax2.legend(handles=legend_handles2)
        ax2.set_title(f'Subject {subject_label2} Lesion Centroids')

        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

    def interactive_plot(self):
        root = tk.Tk()
        root.title("Lesion Visualisation")

        def on_close():
            root.quit()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_close)

        mainframe = ttk.Frame(root, padding="3 3 12 12")
        mainframe.grid(column=0, row=0, sticky="nsew")
        mainframe.columnconfigure(0, weight=1)
        mainframe.rowconfigure(0, weight=1)

        acceptance_level = tk.DoubleVar()
        acceptance_level.set(0.0)

        fig1 = plt.figure(figsize=(7, 7))
        fig2 = plt.figure(figsize=(7, 7))

        ax1 = self.plot(fig1, self.store1, self.store1["meta"]["brain_center"], 1)
        ax2 = self.plot(fig2, self.store2, self.store2["meta"]["brain_center"], 2)

        canvas1 = FigureCanvasTkAgg(fig1, master=mainframe)
        canvas1.get_tk_widget().grid(column=1, row=1, rowspan=2)

        canvas2 = FigureCanvasTkAgg(fig2, master=mainframe)
        canvas2.get_tk_widget().grid(column=2, row=1, rowspan=2)

        def on_slider_move(event):
            current_value_label.config(text=f"Acceptance Level: {acceptance_level.get()}")
            self.update_plot(acceptance_level.get(), ax1, ax2)
            canvas1.draw()
            canvas2.draw()

        def on_mouse_move(event):
            # Update the viewing angle of the second plot to match the first plot
            ax2.view_init(elev=ax1.elev, azim=ax1.azim)
            canvas2.draw()

        fig1.canvas.mpl_connect('motion_notify_event', on_mouse_move)

        slider = ttk.Scale(mainframe, orient=tk.HORIZONTAL, length=10, from_=0.0, to=20.0, variable=acceptance_level,
                           command=on_slider_move)
        slider.grid(column=1, row=3, columnspan=2, sticky="ew")

        slider_label = ttk.Label(mainframe, text="Acceptance Level")
        slider_label.grid(column=0, row=3, sticky=tk.W)

        current_value_label = ttk.Label(mainframe, text=f"Acceptance Level: {acceptance_level.get()}")
        current_value_label.grid(column=1, row=4, columnspan=2, sticky=tk.W)

        root.mainloop()


class LesionAnalysisManager:
    def __init__(self, subjects, sphere_radius, mass_factor, center_distance_and_mass_approach):
        self.store1 = {}
        self.store2 = {}
        self.subjects = subjects
        self.sphere_radius = sphere_radius
        self.mass_factor = mass_factor
        self.center_distance_and_mass_approach = center_distance_and_mass_approach

    def process_lesions(self):
        if len(self.subjects) == 2:
            if all(len(subject.store["lesions"]) > 1 for subject in
                   self.subjects) and not self.center_distance_and_mass_approach:
                distance_matrices = [MultiLesion(subject, self.sphere_radius) for subject in self.subjects]

                for distance_matrix in distance_matrices:
                    distance_matrix.calculate_vectors()
                    distance_matrix.get_intersect_points()
                    distance_matrix.get_distances()
                    distance_matrix.create_distance_matrices()

                self.store1, self.store2 = [dm.store for dm in distance_matrices]

                subject1 = [matrix for matrix_name, matrix in
                            distance_matrices[0].store["meta"]["distance_matrices"].items()]
                subject2 = [matrix for matrix_name, matrix in
                            distance_matrices[1].store["meta"]["distance_matrices"].items()]

                self.run_analyzer(subject1, subject2)

                self.store1 = copy.deepcopy(self.store1)
                self.store2 = copy.deepcopy(self.store2)

            if any(len(subject.store["lesions"]) == 1 for subject in
                   self.subjects) or self.center_distance_and_mass_approach:
                distance_matrices = [SingleLesion(subject, self.sphere_radius) for subject in self.subjects]

                for distance_matrix in distance_matrices:
                    distance_matrix.get_distances()
                    distance_matrix.create_single_lesion_distance_matrix(self.mass_factor)

                self.store1, self.store2 = [dm.store for dm in distance_matrices]

                subject1 = [matrix for matrix_name, matrix in
                            distance_matrices[0].store["meta"]["distance_matrices"].items()]
                subject2 = [matrix for matrix_name, matrix in
                            distance_matrices[1].store["meta"]["distance_matrices"].items()]

                self.run_analyzer(subject1, subject2)

                self.store1 = copy.deepcopy(self.store1)
                self.store2 = copy.deepcopy(self.store2)

        else:
            print("Processing requires exactly two subjects.")

    def run_analyzer(self, subject1, subject2):
        analyzer = AnalyzerforLesion(self.store2, self)
        lowest_values2 = analyzer.find_lowest_values(subject1, subject2)

        print("Matching Output:")
        matches = analyzer.matching_output(lowest_values2)
        new_lesions = analyzer.new_lesion_output(lowest_values2)
        lost_lesions = analyzer.lost_lesion_output(lowest_values2)

        print("\nMatched Lesions:")
        for match in matches:
            print(match)

        print("\nNew Lesions:")
        if new_lesions is None or len(new_lesions) == 0:
            print('--')
        else:
            for lesion in new_lesions:
                print(lesion)

        print("\nLost Lesions:")
        if lost_lesions is None or len(lost_lesions) == 0:
            print('--')
        else:
            for lesion in lost_lesions:
                print(lesion)

        # Calculate and print centroid movement metrics
        average_movement, max_movement, max_mv_x, max_mv_y, max_mv_z = analyzer.calc_centroid_movement(lowest_values2)
        print(f"\nAverage Centroid Movement: {average_movement:.2f} units")
        print(f"Max Centroid Movement: {max_movement:.2f} units")
        print(f"Max Movement in X direction: {max_mv_x:.2f} units")
        print(f"Max Movement in Y direction: {max_mv_y:.2f} units")
        print(f"Max Movement in Z direction: {max_mv_z:.2f} units")

    def run(self):
        self.process_lesions()
        vis = Visualisation(self.store1, self.store2, self.sphere_radius)
        vis.interactive_plot()


if __name__ == '__main__':
    # todo In Subject() insert your directory with the segmented lesions (binary mask)
    subjects = [
        Subject(),
        Subject()
    ]

    """set mass_factor to 0 if mass should not be taken into consideration.
    mass_factor has only an impact when we have a single-to-single or a single-to-multi lesion situation"""

    center_distance_and_mass_approach = False
    artificial_data = True
    manager = LesionAnalysisManager(subjects, sphere_radius=90, mass_factor=10,
                                    center_distance_and_mass_approach=center_distance_and_mass_approach)

    manager.run()
