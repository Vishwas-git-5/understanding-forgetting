import numpy as np
from evaluate_accuracy import get_results_dict
import sys
import os

sys.stdout = open('latex_tables.txt', 'w')

# Only English language
lang_codes_to_name = {
    'en': 'English',
}

# File path
file_path = '/content/understanding-forgetting/icl_vs_if/out_csvs/batch-alpaca.csv'

# Check if the file exists
if not os.path.exists(file_path):
    print(f"Error: The file '{file_path}' does not exist.")
    sys.exit(1)

files, results = get_results_dict(file_path)

# Only the alpaca model
MODELS = ['alpaca']
TASK_SHOTS = [('capslock-math', 2), ('repeat-math', 2), 
              ('capslock-startblank', 2), ('repeat-startblank', 2)]
TEMPLATES = ['input']
LANGS = ['en']
INSTR = 'instr'

def format_mean_std(mean, std):
    return f" {mean:0.2f} \\% "

def get_mean_std(task, instr, templates, lang, shot, model):
    good_files = [f'{task}-{instr}-{template}-{lang}-{shot}shot' for template in templates]
    vals = list(map(lambda pair: pair[1], list(filter(lambda pair: pair[0] in good_files, list(zip(files, results[model]))))))
    mean, std = np.mean(vals), np.std(vals)
    return mean, std

def get_latex_cell(task, model, shot):
    ret = "& \\begin{tabular}{c} "
    for lang in LANGS:
        ret += format_mean_std(*get_mean_std(task, INSTR, TEMPLATES, lang, shot, model))
        ret += " \\\\ "
    ret = ret[:-3]
    ret += " \end{tabular} "
    return ret

def get_latex_row(task, models, shot):
    ret = ""
    for model in models:
        ret += " " + get_latex_cell(task, model, shot) + " "
    return ret

def get_langs():
    ret = ''
    ret += '& \\begin{tabular}{c}'
    for lang in LANGS:
        ret += f' {lang_codes_to_name[lang]} '
        ret += ' \\\\ '
    ret = ret[:-3]
    ret += ' \end{tabular} '
    return ret

for (task, shot) in TASK_SHOTS:
    icl_task, if_task = task.split('-')[0].capitalize(), task.split('-')[1].capitalize()
    print('\\midrule')
    print(f'\\makecell{{ {icl_task} \\\\ {if_task}}}')
    print(get_langs())
    print(get_latex_row(task, MODELS, shot))
    print('\\\\')

print('\n\n\n')

def get_mean_by_lang_model(lang, model):
    return np.mean([get_mean_std(task, INSTR, TEMPLATES, lang, shot, model)[0] for (task, shot) in TASK_SHOTS])

average_results = {
    lang: {
        model: get_mean_by_lang_model(lang, model)
        for model in MODELS
    } for lang in LANGS
}

def get_average_latex_cell(fine_tune):
    base, tuned = fine_tune
    ret = ""

    ret += "& \\begin{tabular}{c} "
    for lang in LANGS:
        ret += f" {average_results[lang][base]:0.2f} \\% "
        ret += " \\\\ "
    ret = ret[:-3]
    ret += " \end{tabular} "

    ret += "& \\begin{tabular}{c} "
    for lang in LANGS:
        ret += f" {average_results[lang][tuned]:0.2f} \\% "
        ret += " \\\\ "
    ret = ret[:-3]
    ret += " \end{tabular} "

    ret += "& \\begin{tabular}{c} "
    for lang in LANGS:
        ret += f" {average_results[lang][base] - average_results[lang][tuned]:0.2f} \\% "
        ret += " \\\\ "
    ret = ret[:-3]
    ret += " \end{tabular} "

    return ret
