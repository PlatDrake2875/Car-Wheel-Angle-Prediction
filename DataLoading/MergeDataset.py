import json
import os
import glob
from tqdm.contrib.concurrent import process_map


def read_json_files(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data


def get_json_files(data_dir):
    subdirs = [os.path.join(data_dir, f'HMB_{i}') for i in range(1, 7)]
    return [file for subdir in subdirs for file in glob.glob(os.path.join(subdir, '*.json'))]


def process_files_in_parallel(files):
    return process_map(read_json_files, files, max_workers=os.cpu_count(), chunksize=1)


def merge_json_files(data_dir, output_filename='dataset.json'):
    all_json_files = get_json_files(data_dir)
    if not all_json_files:
        print("No JSON files found.")
        return

    results = process_files_in_parallel(all_json_files)

    final_data = {}
    for result in results:
        final_data.update(result)

    output_path = os.path.join(data_dir, output_filename)
    with open(output_path, 'w') as file:
        json.dump(final_data, file, indent=4)
    print(f"JSON data merged to: {output_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    merge_json_files(data_dir)


if __name__ == '__main__':
    main()
