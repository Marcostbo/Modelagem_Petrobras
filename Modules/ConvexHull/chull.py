from numpy import array
import pandas as pd
import numpy as np
import os
import subprocess
import time


class ConvexHull(object):

    def create_input_file(self, points: array) -> None:

        with open(os.path.join(os.environ['CHULL_EXE_PATH'], 'Matriz.txt'), 'w') as file:
            file.write(f"{points.shape[0]}\n")
            file.write(f"{points.shape[1]}\n")
            for lin in range(points.shape[0]):
                for col in range(points.shape[1]):
                    file.write(f"{lin + 1}\n")
                    file.write(f"{col + 1}\n")
                    file.write(f"{points[lin, col]}\n")
        file.close()

    def create_output_dataframe(self, variables: list) -> pd.DataFrame:

        columns = variables + ['indep']
        path_file = os.path.join(os.environ['CHULL_EXE_PATH'], 'ParamPlanos.m')
        while True:
            if os.path.exists(path_file):
                aux = open(path_file, 'r').read()
                if aux[-2:] == '];':
                    file = open(path_file, 'r').readlines()
                    output = pd.DataFrame(data=[], columns=columns)
                    for i in range(1, len(file)-1):
                        values = file[i].split()
                        for j, val in enumerate(values):
                            output.loc[i-1, columns[j]] = float(val)
                    break
        output = output.astype(float).round(5)
        output.drop_duplicates(keep='first', inplace=True)

        # orig_output = output.copy()
        # orig_length = len(orig_output.index)
        if 'vol' in variables:
            output.drop(output[output.vol < 0.].index, inplace=True)
        if 'turb' in variables:
            output.drop(output[output.turb < 0.].index, inplace=True)
        if 'vert' in variables:
            output.drop(output[output.vert > 0.].index, inplace=True)
        output.drop(output[output.indep < 0.].index, inplace=True)

        # if orig_length != len(output.index):
        #     print('teste')

        if len(output.index) == 0:
            print('teste')

        os.remove(path_file)
        while True:
            if not os.path.exists(path_file):
                break

        return output

    def run(self, points: array, variables: list) -> pd.DataFrame:

        self.create_input_file(points=points)

        os.chdir(os.environ['CHULL_EXE_PATH'])
        # os.startfile('chull.exe')
        subprocess.Popen('chull.exe', creationflags=subprocess.SW_HIDE, shell=True)
        # time.sleep(1.5)

        output = self.create_output_dataframe(variables=variables)
        return output
