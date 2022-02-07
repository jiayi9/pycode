import cv2
import numpy
import pandas as pd
import os
import matplotlib.pyplot as plt

# list all files under a folder recursively
def list_files_recur(path, format = '.BMP'):
    file_paths = []
    file_names = []
    for r, d, f in os.walk(path):
        for file in f:
            if format in file or format.lower() in file:
                file_paths.append(os.path.join(r, file))
                file_names.append(file)
    return([file_paths, file_names])


# Not recursively
def list_files(path, FORMAT = '.xlsx'):
    file_paths = []
    file_names = []
    for r, d, f in os.walk(path):
        for file in f:
            if (FORMAT.upper() in file or FORMAT.lower() in file) and r == path:
                file_paths.append(os.path.join(r, file))
                file_names.append(file)
    return([file_paths, file_names])
