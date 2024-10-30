import argparse


def get_args():
    # sql执行数据记录
    parser = argparse.ArgumentParser(description="config of Saro")
    parser.add_argument('--ARMS', type=int, default=49, help='number of arms')
    parser.add_argument('--fp', type=str, default=r'D:\Saro\datasets\train\JOB', help='directory of SQL path')
    parser.add_argument('--sql_dict',
                        type=str,
                        default=r'D:\records\JOB.pkl',
                        help='dictionary of dataset execute record')

    return parser.parse_args()
