from Modules.WindScenariosGenerator.wind import wind
import numpy as np
# from windrose import WindroseAxes
from matplotlib import pyplot as plt
import os
from datetime import date


def plot_wind_rose(dados_vento):

    # WindRose Plot
    nanos = len(dados_vento.ano)
    anos_meses = np.zeros(12)  # quantidade de anos contabilizados no mês
    for iano in range(nanos):
        for imes in range(dados_vento.ano[iano].nr_meses):
            anos_meses[imes] += 1
    mes_nome = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho', 'Agosto', 'Setembro', 'Outubro',
                'Novembro', 'Dezembro']
    for k in range(12):
        ws = []
        wd = []
        tam = 0
        for iano in range(int(anos_meses[k])):
            ws.append(dados_vento.ano[iano].wSpeed_100[k])
            wd.append(dados_vento.ano[iano].wDir_100[k])
            tam += len(dados_vento.ano[iano].wSpeed_100[k])

            # ws = np.asarray(ws[0])
            # wd = np.asarray(wd[0])
            # ax = WindroseAxes.from_ax()
            # ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='white')
            # ax.set_legend()
            # plt.savefig(f'Teste.png')
            #
            # galo = 13

        WS = np.zeros(tam)
        WD = np.zeros(tam)
        pos = 0
        for i in range(len(ws)):
            for j in range(len(ws[i])):
                WS[pos] = ws[i][j]
                # WD[pos] = 180+wd[i][j]
                WD[pos] = wd[i][j]
                pos += 1
        plt.rcParams.update({'font.size': 22})
        ax = WindroseAxes.from_ax()
        ax.bar(WD, WS, normed=True, opening=1, edgecolor='white')
        ax.set_legend()
        plt.legend(loc="lower left", prop={'size': 18})
        plt.tight_layout()
        ax.set_title("%s" % mes_nome[k])
        plt.savefig(" %s.png" % mes_nome[k])


def main(altitude: int, ref_date: date, nr_horas: int):

    # Leitura dos dados de vento
    dados_vento = wind(os.environ['WIND_DATA_PATH'])

    # Correção da velocidade de vento
    dados_vento.correcao_Hellman(altitude)

    # Tratamento dos dados (eliminar outliers)
    dados_vento.tratamento()

    # Cálculo da densidade de potência de vento horário e mensal
    dados_vento.calculo_dp()

    # Alocação dos dados no formato (dias x horas) para cada mês
    dados_vento.curtoprazo()

    # Aplicação da LSTM
    mes = ref_date.month
    dados = dados_vento.wDPcurto_alt[mes][-(3*31):][:]  # dados de todos os Janeiros de 2010 a 2018
    dados = np.reshape(dados, (np.size(dados), 1))
    # hours = np.reshape(np.cumsum([[i % 24] for i in range(len(dados))]), (len(dados), 1))
    # dados = np.concatenate((dados, hours), axis=1)
    hourly_prediction = dados_vento.previsao_LSTM(dados, nr_horas)

    return hourly_prediction

    # hours_forecast = 1*24
    #
    # n_output = 12
    #
    # train, test = split_dataset(dados, hours_forecast, n_output)
    # scaler, train_scaled, test_scaled = scale(train, test)
    # # evaluate model and get scores
    # for i in range(7):
    #
    #     n_input = (i+1)*24
    #
    #     t = timer()
    #
    #     predictions = lstm_prediction(train_scaled, test_scaled, n_input, n_output)
    #     predictions = predictions.reshape((predictions.size, 1))
    #     predictions = invert_scale(scaler, predictions)
    #
    #     tempo = round(timer() - t, 2)
    #
    #     s = 0
    #     for k in range(hours_forecast):
    #         s += abs(predictions[k, 0] - dados[-hours_forecast + k])/dados[-hours_forecast + k]
    #     mape = 100 * s / hours_forecast
    #
    #     print(f'Tempo - {i+1} dias: {round(tempo,1)} seg   ->    MAPE = {round(mape[0],1)}')
    #
    #     plt.figure(i)
    #     plt.title(f'{i+1} dia(s) usado na previsão de {n_output} horas à frente \n'
    #               f'MAPE = {round(mape[0],1)} % \n'
    #               f'Tempo: {round(tempo,1)} seg')
    #     plt.plot(predictions[:, 0], label='Previsão')
    #     plt.plot(dados[-hours_forecast:], label='Real')
    #     plt.legend()
    #     plt.xlabel('$Horas$')
    #     plt.ylabel('$KW/m^{2}$')
    #
    #     plt.show()