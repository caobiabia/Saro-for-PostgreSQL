from pathlib import Path


def list_files_in_directory(directory_path):
    directory = Path(directory_path)
    file_list = [str(file) for file in directory.rglob('*') if file.is_file()]
    return file_list


# # 示例用法
# directory_path = r'D:\Saro\datasets\JOB'
# all_files = list_files_in_directory(directory_path)
# print(all_files)
