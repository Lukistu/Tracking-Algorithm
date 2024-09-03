
# Tracking-Algorithm

**High-Sensitivity Longitudinal Brain Metastases Tracking Algorithm**

This algorithm offers an alternative approach to tracking brain lesions across timepoints, based on the characteristic distances of each lesion.

## Features

- Tracks lesions using Nifti files (binary masks).
- Supports both single and multiple lesions per scan.
- Includes artificial data generation for performance testing.

## Installation

Clone the repository and install the necessary dependencies.

```bash
git clone https://github.com/yourusername/Tracking-Algorithm.git
cd Tracking-Algorithm
pip install -r requirements.txt
```

## Usage

### Tracking Lesions with Nifti Files (Binary Mask)

1. **Set the path to your Nifti files:**  
   Update the path at the end of the main script.

2. **If you have more than one lesion per scan:**

   ```python
   center_distance_and_mass_approach = False
   artificial_data = False
   manager = LesionAnalysisManager(
       subjects, 
       sphere_radius=90, 
       mass_factor=0, 
       center_distance_and_mass_approach=center_distance_and_mass_approach
   )
   ```

   - Set `sphere_radius` to a value that fully encompasses the brain.

3. **If you have a single lesion in one or both scans:**

   ```python
   center_distance_and_mass_approach = True  # Alternatively, leave it False; it will auto-update
   artificial_data = False
   manager = LesionAnalysisManager(
       subjects, 
       sphere_radius=90, 
       mass_factor=0, 
       center_distance_and_mass_approach=center_distance_and_mass_approach
   )
   ```

### Tracking Artificially Generated Lesions

1. Open `create_artificial_data.py`.
2. Set the path to save CSV files (tracking accuracy results).
3. Adjust parameters as needed. Example:

   ```python
   self.pad_width = 3
   self.cube_dims = [100, 100, 100]
   self.conf = {
       'lesions': [10],
       'remove': [3],
       'add': [0],
       'translation': [0],
       'rotation': [0],  
       'variance': [4]
   }
   ```

4. Set visualization and the number of iterations:

   ```python
   visualization = True
   x = 1  # Number of iterations
   ```

### Running Performance Tests Using Artificial Data

1. Open `create_artificial_data.py`.
2. Set the path to save the CSV files (tracking accuracy results).
3. Configure parameters. Example:

   ```python
   self.pad_width = 3
   self.cube_dims = [100, 100, 100]
   self.conf = {
       'lesions': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
       'remove': [0],
       'add': [1, 2, 3, 4, 5],
       'translation': [0],
       'rotation': [0],  
       'variance': [4]
   }
   ```

4. Set visualization and iterations:

   ```python
   visualization = False
   x = 3  # Number of iterations
   ```

5. Open `print_graphs_from_csv.py`.
6. Set `file_path_pattern` to the location of your saved CSV files.
7. Define the variable to plot.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License.

## Contact

For any questions, feel free to reach out via [your email/contact link].
