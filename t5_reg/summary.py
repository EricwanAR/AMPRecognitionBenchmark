import os
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='./run')
parser.add_argument('--output', type=str, default='summary.csv')
args = parser.parse_args()

# 设置目录路径和输出CSV文件名
directory_path = args.dir
output_csv = args.output

# 创建CSV文件并写入数据
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # 写入第一个子表的标题和表头
    writer.writerow(['QLX'])
    writer.writerow(['Run', 'MAE', 'RSE', 'PCC', 'KCC'])

    # 遍历指定目录中的子目录
    for subdirectory in os.listdir(directory_path):
        subdirectory_path = os.path.join(directory_path, subdirectory)
        if os.path.isdir(subdirectory_path):
            result_file_path = os.path.join(subdirectory_path, 'result.txt')
            if os.path.isfile(result_file_path):
                with open(result_file_path, 'r') as result_file:
                    lines = result_file.readlines()
                    if len(lines) >= 6:
                        means_5_6 = lines[4].strip().split(',')
                        std_devs_5_6 = lines[5].strip().split(',')
                        if len(means_5_6) == 4 and len(std_devs_5_6) == 4:
                            values_5_6 = [f'{float(mean.strip()) * 100:.2f}$_{{\\pm{{{float(std_dev.strip()) * 100:.2f}}}}}$' 
                                          for mean, std_dev in zip(means_5_6, std_devs_5_6)]
                            writer.writerow([subdirectory] + values_5_6)

    # 加入空行以分隔两个子表
    writer.writerow([])

    # 写入第二个子表的标题和表头
    writer.writerow(['SAAP'])
    writer.writerow(['Run', 'MAE', 'RSE', 'PCC', 'KCC'])

    # 再次遍历指定目录中的子目录
    for subdirectory in os.listdir(directory_path):
        subdirectory_path = os.path.join(directory_path, subdirectory)
        if os.path.isdir(subdirectory_path):
            result_file_path = os.path.join(subdirectory_path, 'result.txt')
            if os.path.isfile(result_file_path):
                with open(result_file_path, 'r') as result_file:
                    lines = result_file.readlines()
                    if len(lines) >= 9:
                        means_8_9 = lines[7].strip().split(',')
                        std_devs_8_9 = lines[8].strip().split(',')
                        if len(means_8_9) == 4 and len(std_devs_8_9) == 4:
                            values_8_9 = [f'{float(mean.strip()) * 100:.2f}$_{{\\pm{{{float(std_dev.strip()) * 100:.2f}}}}}$' 
                                      for mean, std_dev in zip(means_8_9, std_devs_8_9)]
                            writer.writerow([subdirectory] + values_8_9)