import os
from timeit import default_timer as timer
import numpy as np

from Modules.WindScenariosGenerator.main_short import main as wind
from Modules.EnergyPlanningShortTerm.main import main as planning
from Modules.PyHydroShortTerm.phst import phst
from Modules.PyHydroShortTerm.hidr import hidr

from Modules.NewaveFcf.main import main as fcf_nw


def main():

    print("Iniciando leitura do deck DESSEM e gerenciamento das variáveis do problema ...")
    t = timer()
    sistema = phst(diretorio=os.environ['DESSEM_DECK_PATH'])
    print("Tempo para leitura do deck DESSEM e gerenciamento das variáveis do problema: %s seg" % int(timer()-t))

    print("Iniciando leitura e gerenciamento dos cortes da Função de Custo Futuro do Newave ...")
    t = timer()
    last_stage_fcf = fcf_nw(hidr=sistema.hidr, month=sistema.conf.DataInicial.month+1)
    print("Tempo para decorrido: %s seg" % int(timer()-t))

    t = timer()
    sistema.hidr = hidr().all_hydroplants_fph_calculus(data=sistema.hidr)
    print(f"Tempo FPH: {int(timer() - t)} seg")

    # Geração de previsão horária de densidade de potência de vento
    vento = wind(altitude=100, ref_date=sistema.conf.DataInicial, nr_horas=24*len(set(sistema.conf.DiscTemporalT['DI'])))
    # vento = np.ones(len(sistema.conf.DataInicial.index))

    # Parâmetros para geraçao de séries sintéticas de ENA
    planning(sistema, vento, nr_wind_turbines=500, last_stage_fcf=last_stage_fcf)

    print('teste')


if __name__ == '__main__':
    main()

