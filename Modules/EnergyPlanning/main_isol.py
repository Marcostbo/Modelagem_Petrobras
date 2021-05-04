# coding=utf-8

from Modules.PyHydro.mddh import mddh
from Modules.EnergyPlanning.pdde_sistema_equivalente import *
from Modules.WindScenariosGenerator.main_long import main as wind
from Modules.EnergyPlanning.sim_oper import Sim_Oper
from Modules.EnergyPlanning.fci import CustoImediato_Isol

from timeit import default_timer as timer


def main(path_nw: str, path_wind: str, nr_months: int, altitude: int, nr_wind_series: int, method_wind_series: str, nr_ena_abert: int, nr_ena_series: int, submarket_codes: list, nr_process: int):

    # Modulo de configuração leitura do deck NEWAVE e modulo de calculo de energias
    sistema = mddh(path_nw)

    submarket_codes_all = [x.Codigo for x in sistema.submercado]
    submarket_index = [submarket_codes_all.index(i) for i in submarket_codes]

    #  Parâmetros de entrada
    mes_ini = sistema.dger.MesInicioEstudo  # mes inicial
    ano_ini = sistema.dger.AnoInicioEstudo  # ano inicial

    # Gera cenários de densidade de potência do vento
    # Gera cenários de energia eólica
    np.random.seed(1234)
    wind_scenarios = np.zeros((len(submarket_index), nr_wind_series, nr_months))
    for isist in range(len(submarket_index)):
        if submarket_codes[isist] == 3:
            wind_series = wind(path_wind=path_wind, altitude=altitude, nr_series=nr_wind_series,
                               method=method_wind_series, sistema=sistema)
            wind_scenarios[isist] = wind_series[:, mes_ini - 1:mes_ini - 1 + nr_months]

    # Gera aberturas e cenários de afluência
    aberturas = np.zeros((len(submarket_index), nr_ena_abert, nr_months))
    submercados = [sistema.submercado[i] for i in submarket_index]
    for idx, submercado in enumerate(submercados):
        submercado.parp(dados=submercado.ENA[:-2, :], ord_max=6)
        submercado.gera_series_sinteticas_sem_tendencia(dados=submercado.ENA[:-2, :], nr_ser=nr_ena_abert, plot=False)
        aberturas[idx] = submercado.series_sinteticas[:, mes_ini-1:mes_ini-1+nr_months]

    cenarios = np.zeros((nr_ena_series, nr_months))
    for icen in range(nr_ena_abert):
        cenarios[icen, :] = icen
    cont = nr_ena_abert
    while cont < nr_ena_series:
        values = np.random.randint(0, nr_ena_abert, nr_months)
        condition = True
        for row in cenarios[:cont, :]:
            if all(values == row):
                condition = False
                break
        if condition:
            cenarios[cont, :] = values
            cont += 1

    afluencias = np.zeros((len(submarket_index), nr_ena_series, nr_months))
    for isist in range(len(submarket_index)):
        for icen in range(nr_ena_series):
            for imes in range(nr_months):
                afluencias[isist, icen, imes] = aberturas[isist, int(cenarios[icen, imes]), imes]

    # Calcula os pesos das aberturas
    pesos_aber = np.zeros((nr_months, nr_ena_abert))
    for imes in range(nr_months):
        for iaber in range(nr_ena_abert):
            cen_afl = [cenarios[icen][imes] for icen in range(nr_ena_series)]
            rep = len([i for i in cen_afl if i == iaber])
            pesos_aber[imes, iaber] = rep/nr_ena_series
    pesos_eol = (1 / nr_wind_series) * np.ones((1, nr_wind_series))

    # # Algoritmo de levantamento da FCI: Sistema Isolado
    # t = timer()
    # FCI = CustoImediato_Isol(sistema, nr_months, wind_scenarios, pesos_eol, submarket_index, nr_process)
    # FCI.run()
    # print('Tempo-CustoImediato:', round(timer() - t, 2), 'seg')
    #
    # # Algoritmo PL-FCI: Sistema Isolado
    # t = timer()
    # teste_FCI = pdde_SistIsol_FCI(sistema, nr_months, aberturas, pesos_aber, afluencias, submarket_index, FCI.Cortes_FCI, nr_process)
    # teste_FCI.run_parallel()
    # print('Tempo-PDDE-SistIsol-FCI:', round(timer() - t, 2), 'seg')
    # time = round((timer() - t) / 60, 2)
    # teste_FCI.plot_convergencia(time=time)

    # Algoritmo PDDE-MC: Sistema Isolado
    aberturas_used = aberturas[0].reshape((1, aberturas.shape[1], aberturas.shape[2]))
    afluencias_used = afluencias[0].reshape((1, afluencias.shape[1], afluencias.shape[2]))
    t = timer()
    teste = pdde_SistIsol(sistema, nr_months, aberturas_used, pesos_aber, afluencias_used, wind_scenarios, pesos_eol, submarket_index, nr_process)
    teste.run_parallel()
    # teste.run()
    time = round((timer() - t) / 60, 2)
    teste.plot_convergencia(time=time, processos=nr_process)

    print('teste')

    # # Simulação da Operação: PDDE-MC-FCI
    # t = timer()
    # operacao = Sim_Oper(sistema)
    # operacao.run(nr_months, afluencias, demandas, teste.cf)
    # print('Tempo-SO-PDDE-SistIsol:', round(timer() - t, 2), 'seg')
