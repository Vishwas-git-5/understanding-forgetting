import os
from itertools import product
import pandas as pd

def concat_csvs(filenames, resulting_filename):
    location = '/content/'
    combined_csv = pd.concat([pd.read_csv(location + f) for f in filenames], ignore_index=True)
    combined_csv.to_csv(location + resulting_filename + '.csv', index=False)

task_list = [('capslock-math', 2), ('repeat-math', 2), ('capslock-startblank', 2), ('repeat-startblank', 2)]
lang_list = ['en']
model_list = ['alpaca']
instr_list = ['instr']
prompt_template_list = ['input']

print('Generate batch of datasets for generate.py...')
print('  Models: alpaca')

files = []
for (task, shot), lang, instr, prompt_template in product(task_list, lang_list, instr_list, prompt_template_list):
    files.append(f"{task}-{instr}-{prompt_template}-{lang}-{shot}shot.csv")

# Check if files exist
missing_files = [f for f in files if not os.path.isfile(f'/content/{f}')]
if missing_files:
    print(f"Error: The following files are missing in /content/: {', '} ".join(missing_files))
else:
    concatenated_filehandle = "batch"
    concat_csvs(files, concatenated_filehandle)
    print(f'Generated {concatenated_filehandle}.csv')

    complete_command = f"""
import subprocess

subprocess.run(["python3", "/content/understanding-forgetting/icl_vs_if/generate.py", "--model", "alpaca", "--batch", "{concatenated_filehandle}"])
    """

    with open("/content/understanding-forgetting/batch_generate.py", "w") as f:
        f.write(complete_command)

    print("Generated batch_generate.py")
