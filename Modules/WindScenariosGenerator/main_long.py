from Modules.WindScenariosGenerator.wind import wind
import matplotlib.pyplot as plt
import pandas as pd
from numpy import ndarray


def main(path_wind: str, altitude: float, nr_series: int, sistema, nr_months: int = 60, test: bool = False, method: str = 'PARp-SSA') -> ndarray:

    # Definição de parâmetros e dados para PARp
    ord_max = 6
    submercado = 2  # submercado NORDESTE

    # TODO: Metodologia
    print('\n')
    print('---------------------------------------------')
    print(f'              Metodologia - {method}            ')
    print('---------------------------------------------')
    print('Altitude = %d metros' % altitude)

    if method == 'PARp':

        # Leitura dos dados de vento
        dados_vento = wind(path_wind)

        # Correção da velocidade de vento (Hellman)
        dados_vento.correcao_Hellman(altitude)

        # Tratamento dos dados (eliminar outliers)
        dados_vento.tratamento()

        # Cálculo da densidade de potência do vento horário e mensal
        dados_vento.calculo_dp()

        # Completando os dados de densidade de potência de vento mensal desde 1931
        dados_vento.completar_historico(sistema, submercado)

        # Escolha dos dados a serem utilizados no PAR(p)
        if dados_vento.mesFin == 12:
            dados = dados_vento.wDPFullmes_alt
        else:
            dados = dados_vento.wDPFullmes_alt[:-1]  # elimina última série (caso esteja incompleta)

        # Aplicação do modelo PAR(p)
        dados_vento.parp(dados, ord_max)

        # Geração das séries sintéticas
        dados_vento.gera_series_sinteticas_sem_tendencia(dados, nr_series, nr_months)

        output_series = dados_vento.series_sinteticas

    elif method == 'PARp-SSA':

        # Leitura dos dados de vento
        dados_vento_ssa = wind(path_wind)

        # Correção da velocidade de vento (Hellman)
        dados_vento_ssa.correcao_Hellman(altitude)

        # Tratamento dos dados (eliminar outliers)
        dados_vento_ssa.tratamento()

        # Cálculo da densidade de potência do vento horário e mensal
        dados_vento_ssa.calculo_dp()

        # Aplicação de SSA (eliminar ruído)
        dados_vento_ssa.algoritmo_SSA()

        # Completando os dados de densidade de potência de vento mensal desde 1931
        dados_vento_ssa.completar_historico(sistema, submercado)

        # Escolha dos dados a serem utilizados no PAR(p)
        if dados_vento_ssa.mesFin == 12:
            dados_ssa = dados_vento_ssa.wDPFullmes_alt
        else:
            dados_ssa = dados_vento_ssa.wDPFullmes_alt[:-1]  # elimina última série (caso esteja incompleta)

        # Aplicação do modelo PAR(p)
        dados_vento_ssa.parp(dados_ssa, ord_max)

        # Geração das séries sintéticas
        dados_vento_ssa.gera_series_sinteticas_sem_tendencia(dados_ssa, nr_series, nr_months)

        output_series = dados_vento_ssa.series_sinteticas

    elif method == 'PARp-BAT':

        # Leitura dos dados de vento
        dados_vento_bat = wind(path_wind)

        # Correção da velocidade de vento (Hellman)
        dados_vento_bat.correcao_Hellman(altitude)

        # Tratamento dos dados (eliminar outliers)
        dados_vento_bat.tratamento()

        # Cálculo da densidade de potência do vento horário e mensal
        dados_vento_bat.calculo_dp()

        # Completando os dados de densidade de potência de vento mensal desde 1931
        dados_vento_bat.completar_historico(sistema, submercado)  # melhor até o momento

        # Escolha dos dados a serem utilizados no PAR(p)
        if dados_vento_bat.mesFin == 12:
            dados_bat = dados_vento_bat.wDPFullmes_alt
        else:
            dados_bat = dados_vento_bat.wDPFullmes_alt[:-1]  # elimina última série (caso esteja incompleta)

        # Aplicação do modelo PAR(p)-BAT
        dados_vento_bat.parp_bat(dados_bat, ord_max)

        # Geração das séries sintéticas
        dados_vento_bat.gera_series_sinteticas_sem_tendencia(dados_bat, nr_series, nr_months)

        output_series = dados_vento_bat.series_sinteticas

    if test:

        # Testes estatísticos
        mean = dados_vento.Teste_de_Media(dados)
        var = dados_vento.Teste_de_Variancia(dados)
        adr = dados_vento.Teste_de_Aderencia(dados)
        med = dados_vento.Teste_de_Mediana(dados)
        ass = dados_vento.Teste_de_Assimetria(dados)
        comp, soma, int = dados_vento.Teste_de_Sequencia_Negativa(dados)

        mean_bat = dados_vento_bat.Teste_de_Media(dados_bat)
        var_bat = dados_vento_bat.Teste_de_Variancia(dados_bat)
        adr_bat = dados_vento_bat.Teste_de_Aderencia(dados_bat)
        med_bat = dados_vento_bat.Teste_de_Mediana(dados_bat)
        ass_bat = dados_vento_bat.Teste_de_Assimetria(dados_bat)
        comp_bat, soma_bat, int_bat = dados_vento_bat.Teste_de_Sequencia_Negativa(dados_bat)

        mean_ssa = dados_vento_ssa.Teste_de_Media(dados_ssa)
        var_ssa = dados_vento_ssa.Teste_de_Variancia(dados_ssa)
        adr_ssa = dados_vento_ssa.Teste_de_Aderencia(dados_ssa)
        med_ssa = dados_vento_ssa.Teste_de_Mediana(dados_ssa)
        ass_ssa = dados_vento_ssa.Teste_de_Assimetria(dados_ssa)
        comp_ssa, soma_ssa, int_ssa = dados_vento_ssa.Teste_de_Sequencia_Negativa(dados_ssa)

        # TODO: Resultados dos testes estatíticos
        print('\n')
        print('--------------------------------------------------------')
        print('           Testes Estatísticos')
        print('--------------------------------------------------------')
        table_1 = pd.DataFrame(
            {'Testes': ['Média', 'Variância', 'Mediana', 'Aderência', 'Assimetria'],
             'PAR(p)': [str(mean)+'%', str(var)+'%', str(med)+'%', str(adr)+'%', str(ass)+'%'],
             'PAR(p)-BAT': [str(mean_bat) + '%', str(var_bat) + '%', str(med_bat) + '%', str(adr_bat)+'%', str(ass_bat)+'%'],
             'PAR(p)-SSA': [str(mean_ssa) + '%', str(var_ssa) + '%', str(med_ssa) + '%', str(adr_ssa)+'%', str(ass_ssa)+'%'],
            }
        )
        print(table_1)
        print('--------------------------------------------------------')
        print('\n')
        print('--------------------------------------------------------')
        print('           Análise de Sequência Negativa')
        print('--------------------------------------------------------')
        table_2 = pd.DataFrame(
            {'Testes': ['Comprimento (Qui2)', 'Soma (Kolmogorov-Smirnov)', 'Intensidade (Kolmogorov-Smirnov)'],
             'PAR(p)': [comp[0], soma[0], int[0]],
             'PAR(p)-BAT': [comp_bat[0], soma_bat[0], int_bat[0]],
             'PAR(p)-SSA': [comp_ssa[0], soma_ssa[0], int_ssa[0]],
            }
        )
        print(table_2)
        print('--------------------------------------------------------')

        # TODO: Plotagem dos gráficos
        plt.show()

    return output_series
