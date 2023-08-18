import argparse
import os
import os.path as osp
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description='cls training results parsing')
    parser.add_argument('--root', type=str, default='./', help='path to root folder')
    args = parser.parse_args()

    log_files_list = []
    for subdir, dirs, files in os.walk(args.root):
        for file in files:
            if '.log' in file:
                log_files_list.append(os.path.join(subdir, file))

    models_results = {}
    for file_p in sorted(log_files_list):
        d_name = osp.dirname(file_p)
        m_name = Path(d_name.split("-")[0]).stem
        device = d_name.split("-")[1]
        if not m_name in models_results:
            models_results[m_name] = {}
        models_results[m_name][device] = {}

        with open(file_p) as f:
            data = f.read()
            matched_acc_lines = re.findall('Top-1 accuracy: \d+.\d+', data)
            final_acc = float(re.findall("\d+\.\d+", matched_acc_lines[-1])[0])
            models_results[m_name][device]["accuracy"] = final_acc

            matched_epoch_time_lines = re.findall('Epoch time compute: \d+.\d+', data)
            final_epoch_time = float(re.findall("\d+\.\d+", matched_epoch_time_lines[-1])[0])
            models_results[m_name][device]["epoch_compute_time"] = final_epoch_time

            matched_val_time_lines = re.findall('Val compute time: \d+.\d+', data)
            final_val_time = float(re.findall("\d+\.\d+", matched_val_time_lines[-1])[0])
            models_results[m_name][device]["val_compute_time"] = final_val_time

    for model in models_results:
        print(model + ":")
        for dev in models_results[model]:
            device_result = ""
            for name, val in models_results[model][dev].items():
                device_result += f"{val}; "
            print("\t" + dev + ": " + device_result)


if __name__ == '__main__':
    main()