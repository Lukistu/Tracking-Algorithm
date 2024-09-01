import pandas as pd
import matplotlib.pyplot as plt
import glob


# Function to read and process each CSV file
def process_csv(file):
    df = pd.read_csv(file)
    df = df[['lesions', 'variance', 'remove', 'add', 'translation', 'rotation', 'pred_val']]
    df['pred_val'] = df['pred_val'] * 100  # Multiply pred_val by 100 to get percentage
    return df


file_path_pattern =  # todo: set path to where your csv files are

# Load all CSV files into a list of DataFrames
csv_files = glob.glob(file_path_pattern)
dfs = [process_csv(file) for file in csv_files]

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(dfs)

# Specify numbers to exclude
numbers_to_exclude = []  # Replace with your specific numbers


# Group by 'lesions' and the selected variable, and calculate the average 'pred_val'
def group_and_average(df, variable):
    df = df[~df[variable].isin(numbers_to_exclude)]  # Exclude specified numbers
    avg_df = df.groupby(['lesions', variable]).mean().reset_index()
    pivot_df = avg_df.pivot(index='lesions', columns=variable, values='pred_val')
    return pivot_df


# Choose the variable to plot (options: 'variance', 'remove', 'add', 'translation', 'rotation')
variable_to_plot = 'variance'

pivot_df = group_and_average(combined_df, variable_to_plot)

# Plotting the data
plt.figure(figsize=(12, 6))
for column in pivot_df.columns:
    plt.plot(pivot_df.index, pivot_df[column], label=f'{variable_to_plot.capitalize()} {column}')

plt.xlabel('Number of Lesions for both tp1 and tp2')
plt.ylabel('Average Prediction Value (%)')
plt.title(f'Percentage of Correctly Matched Lesions by Varying Amount of Variance')
plt.legend()

# Set y-axis range from 0 to 100 to reflect percentage values
plt.ylim(0, 100)

# Ensure x-axis has only full numbers
plt.xticks(ticks=pivot_df.index, labels=[int(x) for x in pivot_df.index])

plt.show()
