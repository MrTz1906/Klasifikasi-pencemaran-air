import random
import csv
import os

# Define the number of rows and columns in the CSV file
num_rows = 500
num_cols = 5

# Generate random numbers and create a list of rows
data = []
for _ in range(num_rows):
    row = [random.randint(0, 100) for _ in range(num_cols-1)]
    row.append(random.randint(1, 5))  # Restrict water_quality_rating to be between 1 and 5
    data.append(row)

# Define the header for the CSV file
header = ['conductivity', 'temperature', 'turbidity', 'total_dissolved_solids', 'water_quality_rating']

# Get the path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the file path
filename = os.path.join(current_dir, 'example_data.csv')

# Write the data to a CSV file
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write the header
    writer.writerow(header)

    # Write the data rows
    writer.writerows(data)

print(f"Example CSV file '{filename}' with {num_rows} rows has been created.")
