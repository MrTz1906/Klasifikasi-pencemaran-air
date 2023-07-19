import os
import nbformat as nbf
import glob

def convert_py_to_ipynb(py_file):
    # Load Python code from file
    with open(py_file, 'r') as file:
        python_code = file.read()

    # Create a new Jupyter Notebook
    notebook = nbf.v4.new_notebook()

    # Add a code cell to the notebook
    code_cell = nbf.v4.new_code_cell(python_code)
    notebook['cells'].append(code_cell)

    # Save the notebook with the same name but different extension
    notebook_file = os.path.splitext(py_file)[0] + '.ipynb'
    nbf.write(notebook, notebook_file)
    print(f"Converted {py_file} to {notebook_file}")

def batch_convert_py_to_ipynb(directory):
    # Get all Python files in the specified directory
    python_files = glob.glob(os.path.join(directory, '*.py'))

    # Exclude the current script's file
    current_script = os.path.abspath(__file__)
    python_files = [file for file in python_files if os.path.abspath(file) != current_script]

    # Convert each Python file to Jupyter Notebook
    for py_file in python_files:
        convert_py_to_ipynb(py_file)

# Get the current script's directory
script_directory = os.path.dirname(os.path.abspath(__file__))

# Create the subfolder "change" in the script's directory if it doesn't exist
subfolder_name = 'change'
subfolder_path = os.path.join(script_directory, subfolder_name)
if not os.path.exists(subfolder_path):
    os.makedirs(subfolder_path)

# Set the subfolder path as the directory for conversion
directory_path = subfolder_path

# Call the function to convert Python files to Jupyter Notebook files
batch_convert_py_to_ipynb(directory_path)