import os
from Modules.WindScenariosGenerator.main_long import main as wind
from Modules.EnergyPlanning.main_isol import main as single_system
from Modules.EnergyPlanning.main_multi import main as multi_system


path_nw = r'D:\Documents\Rotinas_Alexandre\UFJF\Projeto Petrobras\PythonProject\NW202001' + os.sep
path_wind = r'D:\Documents\Rotinas_Alexandre\UFJF\Projeto Petrobras\PythonProject\Dados_Vento_2010_2018' + os.sep


def main():

    # Configurações Gerais
    nr_months = 24
    nr_process = 8

    # Parâmetros para geraçao de séries de densidade de potência do vento
    altitude = 100
    nr_wind_series = 3
    method_wind_series = 'PARp-SSA'

    # Parâmetros para geraçao de séries sintéticas de ENA
    nr_ena_abert = 2
    nr_ena_series = 50
    submarket_codes = [1, 2, 3, 4]  # [Sudeste: 1, Sul: 2, Nordeste: 3, Norte: 4]

    multi_system(path_nw=path_nw, path_wind=path_wind, nr_months=nr_months, altitude=altitude, method_wind_series=method_wind_series,
                 nr_wind_series=nr_wind_series, nr_ena_abert=nr_ena_abert, nr_ena_series=nr_ena_series, submarket_codes=submarket_codes, nr_process=nr_process)

    print('teste')


if __name__ == '__main__':
    main()

