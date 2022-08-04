import matplotlib.pyplot as plt
import argparse
import numpy as np
from collections import Counter
import tqdm
import xlwt
import os

def parse_single(log_path, log_target, orig, excel_pointer):
    filename = os.path.join(log_path, log_target)
    # filename = os.path.join(log_path)
    subfolder = os.listdir(filename)
    # print(filename)
    # exit()
    # subfolder = []
    # list = ['_iNaturalist', '_SUN', '_Places', '_Textures']
    # for i in range(4):
    #     name = 'test_'+log_target+list[i]
    #     subfolder.append(name)
    if 'ODIN' in log_target:
        subfolder = ['test_ODIN_iNaturalist', 'test_ODIN_SUN', 'test_ODIN_Places', 'test_ODIN_Textures']
    elif 'MSP' in log_target:
        subfolder = ['test_MSP_iNaturalist', 'test_MSP_SUN', 'test_MSP_Places', 'test_MSP_Textures']
    elif 'Energy' in log_target:
        subfolder = ['test_Energy_iNaturalist', 'test_Energy_SUN', 'test_Energy_Places', 'test_Energy_Textures']
    elif 'new' in log_target:
        subfolder = ['test_new_iNaturalist', 'test_new_SUN', 'test_new_Places', 'test_new_Textures']
    elif 'GradNorm' in log_target:
        subfolder = ['test_GradNorm_iNaturalist', 'test_GradNorm_SUN', 'test_GradNorm_Places', 'test_GradNorm_Textures']
    # if 'ODIN' in log_target:
    #     subfolder = ['test_ODIN_Textures']
    # elif 'MSP' in log_target:
    #     subfolder = ['test_MSP_Textures']
    # elif 'Energy' in log_target:
    #     subfolder = ['test_Energy_Textures']
    # elif 'new' in log_target:
    #     subfolder = ['test_new_Textures']
    # elif 'GradNorm' in log_target:
    #     subfolder = ['test_GradNorm_Textures']
    # subfolder = ['test_new_Textures']
    # filename = log_path
    # subfolder = [log_target]
    log_name = 'log.txt'
    for i in range(len(subfolder)):
        path = os.path.join(filename, subfolder[i], log_name)
        with open(path, 'r') as f:
            lines = f.readlines()
        for j in range(len(lines)):
            r_delta = -1
            if 'Head Results for ' in lines[j]:
                r_delta = 1
            if 'Mid Results for ' in lines[j]:
                r_delta = 2
            if 'Tail Results for ' in lines[j]:
                r_delta = 3
            if 'Overall Results for ' in lines[j]:
                r_delta = 0
            if r_delta != -1:
                quick_data = lines[j + 5].split("quick data: ")[-1].strip()
                quick_data = list(map(float, quick_data.split(",")))
                for k, data in enumerate(quick_data):
                    try:
                        excel_pointer.write(orig[0]+r_delta, orig[1]+i*4+k, data)
                    except:
                        print(filename)




if __name__ == '__main__':
    log_path = "./checkpoint0801/maha_a2"
    log_folder = os.listdir(log_path)
    # log_folder_ = []
    # for item in log_folder:
    #     if 'repeat' in item:
    #         log_folder_.append(item)
    # log_folder = log_folder_
    # log_folder.sort(key=lambda x: list(map(float, x.split("repeat")[-1])))
    # log_folder = ['gradnorm_a7_repeat3', 'gradnorm_a7_repeat6', 'gradnorm_a7_repeat9','gradnorm_a8_repeat3', 'gradnorm_a8_repeat6']
    wbk = xlwt.Workbook()
    sheet = wbk.add_sheet('sheet 1')
    ptr = 0
    for log_target in log_folder:
        if 'out_confs' in log_target:
            continue
        if 'collate' in log_target:
            continue
        if 'Mahalanobis' in log_target:
            continue
        try:
            a = log_target.split("_a")[0]
            b = log_target.split("_a")[-1].split("_")[-1]
        except:
            a = log_target
            b = 'None'
        sheet.write(ptr, 0, a)
        sheet.write(ptr, 0 + 1, b)
        sheet.write(ptr + 1, 2, "Head")
        sheet.write(ptr + 2, 2, "Mid")
        sheet.write(ptr + 3, 2, "Tail")
        sheet.write(ptr , 2, "Overall")
        parse_single(log_path, log_target, (ptr, 3), sheet)
        ptr += 4
    wbk.save(os.path.join(log_path, "collate.xls"))



