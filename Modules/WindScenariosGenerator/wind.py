import pandas as pd
import glob
import numpy as np
from numpy import linalg as LA
import os
from scipy.stats import pearsonr
# from scipy.stats import chi2
import matplotlib.pyplot as plt
import math
from timeit import default_timer as timer
from pandas import DataFrame
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from pandas import concat
from keras.utils import plot_model


class read(object):

    def __init__(self, path):
        self.path = path
        self.wSpeed_10 = []
        self.wDir_10 = []
        self.wDPhor_10 = []
        self.wDPmes_10 = []
        self.wSpeed_100 = []
        self.wDir_100 = []
        self.wDPhor_100 = []
        self.wDPmes_100 = []
        self.wSpeed_alt = []
        self.wDir_alt = []
        self.wDPhor_alt = []
        self.wDPmes_alt = []
        self.WeibullParam_10 = []
        self.WeibullParam_100 = []
        self.nr_meses = None
        self.ler()  # leitura dos dados de vento e cálculo da densidade de potência do vento (em KW/m^2)

    def ler(self):

        allfiles = sorted(glob.glob(self.path + '/*.txt'))
        self.nr_meses = len(allfiles)

        for imes in range(self.nr_meses):

            # TODO: Leitura de velocidade (m/s) e orientação (graus) do vento em cada mês

            dados = pd.read_csv(allfiles[imes], header=None, delimiter='\s+')

            # Dados a 10 metros de altitude
            v1_10 = dados.iloc[11:, 2]
            v_10 = []
            for i in v1_10:
                v_10.append(float(i))
            self.wSpeed_10.append(v_10)

            o1_10 = dados.iloc[11:, 3]
            o_10 = []
            for i in o1_10:
                o_10.append(float(i))
            self.wDir_10.append(o_10)

            # Dados a 100 metros de altitude
            v1_100 = dados.iloc[11:, 6]
            v_100 = []
            for i in v1_100:
                v_100.append(float(i))
            self.wSpeed_100.append(v_100)

            o1_100 = dados.iloc[11:, 7]
            o_100 = []
            for i in o1_100:
                o_100.append(float(i))
            self.wDir_100.append(o_100)

            # plt.figure()
            # plt.plot(v_10, label='10m')
            # plt.plot(v_100, label='100m')
            # plt.title('Janeiro de 2010')
            # plt.xlabel('Horas')
            # plt.ylabel('Velocidade [m/s]')
            # plt.legend()
            # plt.show()
            #
            # galo = 13


class wind(object):

    """ Leitura dos dados de velocidade de vento e
        cálculo da densidade de potência horária e mensal.
    """

    def __init__(self, path):

        self.mesIni = 1
        self.anoIni = 2010
        self.mesFin = 8
        self.anoFin = 2018

        self.wDPFullhor_10 = []
        self.wDPFullmes_10 = []
        self.wDPFullhor_100 = []
        self.wDPFullmes_100 = []
        self.wDPFullhor_alt = []
        self.wDPFullmes_alt = []

        self.path = path
        # paths = sorted(os.listdir(path))
        paths = sorted([f.path for f in os.scandir(path) if f.is_dir()])

        # TODO: Leitura de cada ano
        self.ano = []
        for iano in range(len(paths)):
            path = paths[iano]
            self.ano.append(read(path))

        print('Leitura dos dados de vento: OK')

    def correcao_Hellman(self, alt):

        if alt == 10:
            nanos = len(self.ano)
            for iano in range(nanos):
                for imes in range(self.ano[iano].nr_meses):
                    self.ano[iano].wSpeed_alt.append(self.ano[iano].wSpeed_10[imes])

        elif alt == 100:
            nanos = len(self.ano)
            for iano in range(nanos):
                for imes in range(self.ano[iano].nr_meses):
                    self.ano[iano].wSpeed_alt.append(self.ano[iano].wSpeed_100[imes])
        else:

            nanos = len(self.ano)
            for iano in range(nanos):
                for imes in range(self.ano[iano].nr_meses):

                    a = self.ano[iano].wSpeed_100[imes]
                    b = self.ano[iano].wSpeed_10[imes]

                    alfa = np.log([x/y for x, y in zip(a, b)]) / np.log(100/10)
                    v_alt = b*((alt/10)**alfa)
                    self.ano[iano].wSpeed_alt.append(v_alt.tolist())

                    # self.ano[iano].wSpeed_100[imes] = self.ano[iano].wSpeed_alt[imes]

                    # plt.figure()
                    # plt.plot(b, 'b', label='$v_{10}$', lw=2)
                    # # plt.plot(a, 'r', label='$v_{100}$', lw=2)
                    # plt.plot(v_alt, 'g', label='$v_{120}$', lw=2)
                    # plt.ylabel('Velocidade [m/s]')
                    # plt.xlabel('Tempo [hrs]')
                    # plt.legend()
                    # plt.tight_layout()
                    # plt.show()

        print('Cálculo de vento para nova altitude: OK')

    def tratamento(self):

        nanos = len(self.ano)
        for iano in range(nanos):
            for imes in range(self.ano[iano].nr_meses):

                # Coloca os dados em ordem crescente
                dados_10 = np.sort(self.ano[iano].wSpeed_10[imes])
                dados_100 = np.sort(self.ano[iano].wSpeed_100[imes])
                dados_alt = np.sort(self.ano[iano].wSpeed_alt[imes])
                n_10 = len(dados_10)
                n_100 = len(dados_100)
                n_alt = len(dados_alt)

                # Calcular os quartis
                Q_10 = np.zeros(3)
                Q_100 = np.zeros(3)
                Q_alt = np.zeros(3)
                for iquartis in np.arange(1, 4):
                    k_10 = int(iquartis * (n_10 + 1) / 4)
                    k_100 = int(iquartis * (n_100 + 1) / 4)
                    k_alt = int(iquartis * (n_alt + 1) / 4)

                    aux_10 = (iquartis * (n_10 + 1) / 4) - k_10
                    aux_100 = (iquartis * (n_100 + 1) / 4) - k_100
                    aux_alt = (iquartis * (n_alt + 1) / 4) - k_alt

                    Q_10[iquartis-1] = dados_10[k_10-1] + aux_10 * (dados_10[k_10] - dados_10[k_10-1])
                    Q_100[iquartis-1] = dados_100[k_100-1] + aux_100 * (dados_10[k_100] - dados_10[k_100-1])
                    Q_alt[iquartis-1] = dados_alt[k_alt-1] + aux_alt * (dados_alt[k_alt] - dados_alt[k_alt-1])

                # Calcula os limites
                L_10 = np.zeros(2)
                L_10[0] = max(min(dados_10), Q_10[0] - 1.5 * (Q_10[2] - Q_10[0]))
                L_10[1] = min(max(dados_10), Q_10[2] + 1.5 * (Q_10[2] - Q_10[0]))

                L_100 = np.zeros(2)
                L_100[0] = max(min(dados_100), Q_100[0] - 1.5 * (Q_100[2] - Q_100[0]))
                L_100[1] = min(max(dados_100), Q_100[2] + 1.5 * (Q_100[2] - Q_100[0]))

                L_alt = np.zeros(2)
                L_alt[0] = max(min(dados_alt), Q_alt[0] - 1.5 * (Q_alt[2] - Q_alt[0]))
                L_alt[1] = min(max(dados_alt), Q_alt[2] + 1.5 * (Q_alt[2] - Q_alt[0]))

                # Substitui os outliers pelos limites (inferior e superior)
                outliers_inf_10 = [index for index, value in enumerate(self.ano[iano].wSpeed_10[imes]) if value < L_10[0]]
                for i in outliers_inf_10:
                    self.ano[iano].wSpeed_10[imes][i] = L_10[0]

                    # plt.figure()
                    # plt.plot(self.ano[iano].wSpeed_10[imes])
                    # plt.xlabel('Horas')
                    # plt.ylabel('Velocidade do vento ($m/s$)')
                    # plt.title('Fevereiro de 2010')
                    # plt.show()
                    #
                    # plt.figure()
                    # plt.boxplot(self.ano[iano].wSpeed_10[imes])
                    # plt.xticks([])
                    # plt.ylabel('Velocidade do vento ($m/s$)')
                    # plt.title('Fevereiro de 2010')
                    # plt.show()

                    # galo = 13

                outliers_sup_10 = [index for index, value in enumerate(self.ano[iano].wSpeed_10[imes]) if value > L_10[1]]
                for i in outliers_sup_10:
                    self.ano[iano].wSpeed_10[imes][i] = L_10[1]

                outliers_inf_100 = [index for index, value in enumerate(self.ano[iano].wSpeed_100[imes]) if value < L_100[0]]
                for i in outliers_inf_100:
                    self.ano[iano].wSpeed_100[imes][i] = L_100[0]
                outliers_sup_100 = [index for index, value in enumerate(self.ano[iano].wSpeed_100[imes]) if value > L_100[1]]
                for i in outliers_sup_100:
                    self.ano[iano].wSpeed_100[imes][i] = L_100[1]

                outliers_inf_alt = [index for index, value in enumerate(self.ano[iano].wSpeed_alt[imes]) if
                                    value < L_alt[0]]
                for i in outliers_inf_alt:
                    self.ano[iano].wSpeed_alt[imes][i] = L_alt[0]
                outliers_sup_alt = [index for index, value in enumerate(self.ano[iano].wSpeed_alt[imes]) if
                                    value > L_alt[1]]
                for i in outliers_sup_alt:
                    self.ano[iano].wSpeed_alt[imes][i] = L_alt[1]

        print('Tratamento de outliers: OK')

    def calculo_dp(self):

        # TODO: Cálculo da densidade de potência horária (DP = 0.5*p*v^3) e mensal (DPmes = soma(DP))
        p = 1.225  # [kg/m^3]

        nanos = len(self.ano)
        for iano in range(nanos):
            for imes in range(self.ano[iano].nr_meses):

                # dados a 10 metros de altitude
                v_10 = self.ano[iano].wSpeed_10[imes]
                dp_10 = (1 / 1e3) * 0.5 * p * (np.asarray(v_10) ** 3)  # dado em [KW/m^2]
                self.ano[iano].wDPhor_10.append(dp_10)
                self.ano[iano].wDPmes_10.append(np.sum(dp_10))

                # dados a 100 metros de altitude
                v_100 = self.ano[iano].wSpeed_100[imes]
                dp_100 = (1 / 1e3) * 0.5 * p * (np.asarray(v_100) ** 3)  # dado em [KW/m^2]
                self.ano[iano].wDPhor_100.append(dp_100)
                self.ano[iano].wDPmes_100.append(np.sum(dp_100))

                # dados a alt metros de altitude
                v_alt = self.ano[iano].wSpeed_alt[imes]
                dp_alt = (1 / 1e3) * 0.5 * p * (np.asarray(v_alt) ** 3)  # dado em [KW/m^2]
                self.ano[iano].wDPhor_alt.append(dp_alt)
                self.ano[iano].wDPmes_alt.append(np.sum(dp_alt))

        print('Cálculo de DP horário e mensal: OK')

    def curtoprazo(self):

        # Número de dias totais
        nanos = len(self.ano)
        anos_meses = np.zeros(12)  # quantidade de anos contabilizados no mês
        for iano in range(nanos):
            for imes in range(self.ano[iano].nr_meses):
                anos_meses[imes] += 1

        self.wDPcurto_10 = []
        self.wDPcurto_100 = []
        self.wDPcurto_alt = []
        for imes in range(12):
            aux_10 = []
            aux_100 = []
            aux_alt = []
            for iano in range(int(anos_meses[imes])):
                ndias = int(len(self.ano[iano].wDPhor_10[imes]) / 24)
                wDPhor_10 = np.reshape(self.ano[iano].wDPhor_10[imes], (ndias, 24))
                wDPhor_100 = np.reshape(self.ano[iano].wDPhor_100[imes], (ndias, 24))
                wDPhor_alt = np.reshape(self.ano[iano].wDPhor_alt[imes], (ndias, 24))
                for idias in range(ndias):
                    aux_10.append(wDPhor_10[idias][:].tolist())
                    aux_100.append(wDPhor_100[idias][:].tolist())
                    aux_alt.append(wDPhor_alt[idias][:].tolist())

            self.wDPcurto_10.append(aux_10)
            self.wDPcurto_100.append(aux_100)
            self.wDPcurto_alt.append(aux_alt)

        print('Reagrupamento dos dados de vento: OK')

    def completar_historico(self, sistema, isis):

        nanos = len(self.ano)
        anos_meses = np.zeros(12, 'i')  # quantidade de anos contabilizados no mês
        for iano in range(nanos):
            for imes in range(self.ano[iano].nr_meses):
                anos_meses[imes] += 1

        # Compara cada histórico de ENA do Nordeste com as séries de DPmês
        if self.mesFin == 12:
            tam_hist_vento = self.anoFin - self.anoIni + 1
        else:
            tam_hist_vento = self.anoFin - self.anoIni + 1 - 1  # retira a última série

        self.correlacao_10 = np.zeros(self.anoIni-1931)
        self.correlacao_100 = np.zeros(self.anoIni-1931)
        self.correlacao_alt = np.zeros(self.anoIni-1931)

        for iano in range(self.anoIni-1931):
            ena = sistema.submercado[isis].ENA[iano, :]
            corr_10 = np.zeros(tam_hist_vento)
            corr_100 = np.zeros(tam_hist_vento)
            corr_alt = np.zeros(tam_hist_vento)
            for jano in range(tam_hist_vento):
                dp_10 = self.ano[jano].wDPmes_10[:]
                corr_10[jano] = pearsonr(ena, dp_10)[0]
                dp_100 = self.ano[jano].wDPmes_100[:]
                corr_100[jano] = pearsonr(ena, dp_100)[0]
                dp_alt = self.ano[jano].wDPmes_alt[:]
                corr_alt[jano] = pearsonr(ena, dp_alt)[0]
            ind_corr_10 = int(np.argmin(corr_10))
            ind_corr_100 = int(np.argmin(corr_100))
            ind_corr_alt = int(np.argmin(corr_alt))

            # Obtém parâmetros do metodo dos mínimos quadrados
            X = ena
            Y_10 = self.ano[ind_corr_10].wDPmes_10[:]
            Y_100 = self.ano[ind_corr_100].wDPmes_100[:]
            Y_alt = self.ano[ind_corr_alt].wDPmes_alt[:]

            mediaX = np.mean(X)
            mediaY_10 = np.mean(Y_10)
            mediaY_100 = np.mean(Y_100)
            mediaY_alt = np.mean(Y_alt)

            aux1 = 0
            aux2 = 0
            for jano in range(len(X)):
                aux1 += X[jano] * Y_10[jano]
                aux2 += X[jano] ** 2
            a_10 = (aux1 - (len(X) * mediaX * mediaY_10)) / (aux2 - (len(X) * (mediaX ** 2)))
            b_10 = mediaY_10 - a_10 * mediaX

            aux1 = 0
            aux2 = 0
            for jano in range(len(X)):
                aux1 += X[jano] * Y_100[jano]
                aux2 += X[jano] ** 2
            a_100 = (aux1 - (len(X) * mediaX * mediaY_100)) / (aux2 - (len(X) * (mediaX ** 2)))
            b_100 = mediaY_100 - a_100 * mediaX

            aux1 = 0
            aux2 = 0
            for jano in range(len(X)):
                aux1 += X[jano] * Y_alt[jano]
                aux2 += X[jano] ** 2
            a_alt = (aux1 - (len(X) * mediaX * mediaY_alt)) / (aux2 - (len(X) * (mediaX ** 2)))
            b_alt = mediaY_alt - a_alt * mediaX

            # Aplicação da backforecasting
            # ena = sistema.submercado[isis].ENA[iano, :]
            DP_10 = a_10 * ena + b_10
            self.wDPFullmes_10.append(DP_10.tolist())
            DP_100 = a_100 * ena + b_100
            self.wDPFullmes_100.append(DP_100.tolist())
            DP_alt = a_alt * ena + b_alt
            self.wDPFullmes_alt.append(DP_alt.tolist())

            # # grafico
            self.correlacao_10[iano] = pearsonr(X, Y_10)[0]
            self.correlacao_100[iano] = pearsonr(X, Y_100)[0]
            self.correlacao_alt[iano] = pearsonr(X, Y_alt)[0]

        for iano in range(nanos):
            AUX_10 = self.ano[iano].wDPmes_10[:]
            self.wDPFullmes_10.append(AUX_10)
            AUX_100 = self.ano[iano].wDPmes_100[:]
            self.wDPFullmes_100.append(AUX_100)
            AUX_alt = self.ano[iano].wDPmes_alt[:]
            self.wDPFullmes_alt.append(AUX_alt)

        print('Preenchimento do histórico: OK')

    def algoritmo_SSA(self):

        nanos = len(self.ano)
        n = 0
        for iano in range(nanos):
            n += self.ano[iano].nr_meses
        dados_10 = np.zeros(n)
        dados_100 = np.zeros(n)
        dados_alt = np.zeros(n)
        ind = 0
        for iano in range(nanos):
            for imes in range(self.ano[iano].nr_meses):

                # Coloca os dados em ordem crescente
                dados_10[ind] = self.ano[iano].wDPmes_10[imes]
                dados_100[ind] = self.ano[iano].wDPmes_100[imes]
                dados_alt[ind] = self.ano[iano].wDPmes_alt[imes]

                ind += 1

        for iteste in range(3):

            if iteste == 0:
                L = 42
                tol = 0.0001
                dados_SSA = dados_10
            elif iteste == 1:
                L = 40
                tol = 0.00035
                dados_SSA = dados_100
            else:
                L = 40
                tol = 0.00035
                dados_SSA = dados_alt

            #TODO: ETAPA 1) Decomposição:

            # 1.1) Incorporação:
            T = n
            K = T - L + 1
            X = np.zeros((L, K))  # Matriz trajetória (Henkel)
            for ilin in range(L):
                X[ilin, :] = dados_SSA[ilin:ilin+K]

            # 1.2) Decomposição em Valores Singulares (SVD):
            S = np.dot(X, np.transpose(X))
            lamb, U = LA.eig(S)

            ind = np.argsort(lamb)[::-1]
            lamb = lamb[ind]
            U = U[:, ind]
            Xi = np.zeros((L, L, K))
            for i in range(L):
                Ui = np.reshape(U[:, i], (L, 1))
                V = (1/np.sqrt(lamb[i]))*np.dot(np.transpose(X), Ui)
                Vi = np.reshape(V, (K, 1))
                Xi[i] = np.sqrt(lamb[i]) * np.dot(Ui, np.transpose(Vi))

            # TODO: ETAPA 2) Reconstrução:

            L_ast = np.minimum(L, K)
            K_ast = np.maximum(L, K)
            Xi_new = np.zeros((L, L, K))
            y = np.zeros((L, T))
            for i in range(L):

                for k in np.arange(1, T+1):

                    if k < L_ast:
                        for m in np.arange(1, k+1):
                            y[i, k-1] += (1 / k) * Xi[i][m-1, k - m]

                    if (k >= L_ast and k <= K_ast):
                        for m in np.arange(1, L_ast+1):
                            y[i, k-1] += (1 / L_ast) * Xi[i][m-1, k - m]

                    if k > K_ast:
                        for m in np.arange(k-K_ast+1, T-K_ast+2):
                            y[i, k-1] += (1 / (T - k + 1)) * Xi[i][m-1, k - m]

                for ilin in range(L):
                    Xi_new[i][ilin, :] = y[i, ilin:ilin + K]

            # Agrupamento
            d = np.linalg.matrix_rank(X)
            contrib = np.zeros(d)
            for i in range(d):
                contrib[i] = lamb[i] / np.sum(lamb)
            ind_sinal = [i for i in range(d) if contrib[i] > tol]
            ind_ruido = [i for i in range(d) if contrib[i] <= tol]

            Y_sinal = np.sum(y[ind_sinal, :], axis=0)
            Y_ruido = np.sum(y[ind_ruido, :], axis=0)

            # TODO: Atualiza os valores de Wind Speed
            ind = 0
            for iano in range(nanos):
                for imes in range(self.ano[iano].nr_meses):

                    # Coloca os dados em ordem crescente
                    if iteste == 0:
                        self.ano[iano].wDPmes_10[imes] = Y_sinal[ind]
                    elif iteste == 1:
                        self.ano[iano].wDPmes_100[imes] = Y_sinal[ind]
                    else:
                        self.ano[iano].wDPmes_alt[imes] = Y_sinal[ind]

                    ind += 1

        print('Aplicação do SSA: OK')

    def previsao_LSTM(self, dados, nr_horas) -> list:

        # create a differenced series
        def difference(dataset, interval=1):
            diff = list()
            for i in range(interval, len(dataset)):
                value = dataset[i] - dataset[i - interval]
                diff.append(value)
            return Series(diff)

        # frame a sequence as a supervised learning problem
        def timeseries_to_supervised(data, lag=1):
            df = DataFrame(data)
            columns = [df.shift(i) for i in range(1, lag + 1)]
            columns.append(df)
            df = concat(columns, axis=1)
            df.fillna(0, inplace=True)
            return df

        # scale train and test data to [-1, 1]
        def scale(train, test):
            # fit scaler
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler = scaler.fit(train)
            # transform train
            train = train.reshape(train.shape[0], train.shape[1])
            train_scaled = scaler.transform(train)
            # transform test
            test = test.reshape(test.shape[0], test.shape[1])
            test_scaled = scaler.transform(test)
            return scaler, train_scaled, test_scaled

        # fit an LSTM network to training data
        def fit_lstm(train, batch_size, nb_epoch, neurons):
            X, y = train[:, 0:-1], train[:, -1]
            X = X.reshape(X.shape[0], 1, X.shape[1])
            model = Sequential()
            model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
            model.add(Dense(1))
            # print(model.summary())
            model.compile(loss='mean_squared_error', optimizer='adam')
            # for i in range(nb_epoch):
            model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=0, shuffle=False)
            # model.evaluate(X, y)
            # print(model.summary())
                # model.reset_states()
                # print('teste')
            # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

            return model

        # make a one-step forecast
        def forecast_lstm(model, batch_size, X):
            X = X.reshape(1, 1, len(X))
            yhat = model.predict(X, batch_size=batch_size)
            return yhat[0, 0]

        # inverse scaling for a forecasted value
        def invert_scale(scaler, X, value):
            new_row = [x for x in X] + [value]
            array = np.array(new_row)
            array = array.reshape(1, len(array))
            inverted = scaler.inverse_transform(array)
            return inverted[0, -1]

        # invert differenced value
        def inverse_difference(history, yhat, interval=1):
            return yhat + history[-interval]

        # load dataset
        # series = Series(dados)
        series = pd.DataFrame(dados)

        # plt.figure()
        # plt.title('Série Original')
        # plt.plot(series.values, 'b')
        # plt.xlabel('Horas')
        # plt.ylabel('$KW/m^{2}$')
        # plt.legend('')
        # plt.show()

        # transform data to be stationary
        raw_values = series.values
        diff_values = difference(raw_values, 1)

        # plt.figure()
        # plt.title('Diferença')
        # plt.plot(diff_values, 'b')
        # plt.xlabel('Horas')
        # plt.ylabel('$KW/m^{2}$')
        # # plt.legend('')
        # plt.show()

        # transform data to be supervised learning
        supervised = timeseries_to_supervised(diff_values, 1)
        supervised_values = supervised.values

        # plt.figure()
        # plt.title('Aprendizado Supervisionado')
        # plt.plot(supervised_values[:, 0],  'b', label='Entrada')
        # plt.plot(supervised_values[:, 1],  'r', label='Saída')
        # plt.xlabel('Horas')
        # plt.ylabel('$KW/m^{2}$')
        # plt.legend()
        # plt.show()

        # split data into train and test-sets
        train, test = supervised_values[0:-nr_horas], supervised_values[-nr_horas:]

        # plt.figure()
        # plt.title('Dados para Treinamento do Modelo')
        # plt.plot(train[:, 0], 'b-', label='Treinamento - Entrada')
        # plt.plot(train[:, 1], 'r-', label='Treinamento - Saída')
        # plt.plot(np.arange(train.shape[0]-1, train.shape[0]+test.shape[0]-1), test[:, 0], 'b--', label='Previsão - Entrada')
        # plt.plot(np.arange(train.shape[0]-1, train.shape[0]+test.shape[0]-1), test[:, 1], 'r--', label='Previsão - Saída')
        # plt.xlabel('Horas')
        # plt.ylabel('$KW/m^{2}$')
        # plt.legend()
        # plt.show()

        # transform the scale of the data
        scaler, train_scaled, test_scaled = scale(train, test)

        # plt.figure()
        # plt.title('Dados para Treinamento do Modelo - Normalizado')
        # plt.plot(train_scaled[:, 0], 'b-', label='Treinamento - Entrada')
        # plt.plot(train_scaled[:, 1], 'r-', label='Treinamento - Saída')
        # plt.plot(np.arange(train_scaled.shape[0]-1, train_scaled.shape[0] + test_scaled.shape[0]-1), test_scaled[:, 0], 'b--', label='Previsão - Entrada')
        # plt.plot(np.arange(train_scaled.shape[0]-1, train_scaled.shape[0] + test_scaled.shape[0]-1), test_scaled[:, 1], 'r--', label='Previsão - Saída')
        # plt.xlabel('Horas')
        # plt.ylabel('$KW/m^{2}$')
        # plt.legend()
        # plt.show()

        # repeat experiment
        repeats = 1
        error_scores = list()
        for r in range(repeats):
            # fit the model
            lstm_model = fit_lstm(train_scaled, batch_size=1, nb_epoch=10, neurons=5)
            # forecast the entire training dataset to build up state for forecasting
            train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
            lstm_model.predict(train_reshaped, batch_size=1)
            # walk-forward validation on the test data
            predictions = list()
            X = train_scaled[-1, 0:-1]  # test_scaled[i, 0:-1]
            for i in range(nr_horas):
                # make one-step forecast
                yhat_1 = forecast_lstm(lstm_model, 1, X)
                # invert scaling
                yhat_2 = invert_scale(scaler, X, yhat_1)
                # invert differencing
                # if i == 0:
                yhat_3 = inverse_difference(raw_values, yhat_2, nr_horas + 1 - i)   #
                # else:
                #     yhat_3 = inverse_difference(predictions, yhat_2, 1)
                # store forecast
                predictions.append(yhat_3[0])
                X[0] = yhat_1
                # print('Valor previsto: ', yhat_3)

            # report performance
            mape = np.mean(np.abs((raw_values[-nr_horas:] - predictions) / raw_values[-nr_horas:])) * 100
            print(f' {r+1}) Test MAPE: {mape}')
            # error_scores.append(mape)

            # plt.figure()
            # plt.plot(predictions, label='Previsão')
            # plt.plot(raw_values[-nr_horas:], label='Real')
            # plt.legend()
            # plt.xlabel('$Horas$')
            # plt.ylabel('$KW/m^{2}$')
            #
            # plt.show()
            #
            # plt.figure()
            # plt.title('Previsão')
            # plt.plot(series.values, 'b', label='Original')
            # plt.plot(np.arange(train_scaled.shape[0]-1, train_scaled.shape[0] + test_scaled.shape[0]-1), predictions, 'ro', label='Previsão')
            # plt.xlabel('Horas')
            # plt.ylabel('$KW/m^{2}$')
            # plt.legend()
            # plt.show()

            # # summarize results
            # results = DataFrame()
            # results['mape'] = error_scores
            # print(results.describe())
            # results.boxplot()
            # plt.show()

        return np.array(predictions)

        print('Aplicação do LSTM: OK')

    def parp(self, dados, ord_max):

        nanos = len(dados)

        media = np.mean(dados, axis=0)
        desvio = np.std(dados, axis=0)

        # Calcula funcao de auto-correlacao (uma para cada mes)
        self.FAC = np.zeros((12, ord_max), 'd')
        for ilag in range(ord_max):
            for imes in range(12):
                for iano in np.arange(1, nanos):
                    ano_ant = iano
                    mes_ant = imes - ilag - 1
                    if mes_ant < 0:
                        ano_ant -= 1
                        mes_ant += 12
                    self.FAC[imes][ilag] += (dados[iano][imes] - media[imes]) * (dados[ano_ant][mes_ant] - media[mes_ant])
                self.FAC[imes][ilag] /= ((nanos-1) * desvio[imes] * desvio[mes_ant])

        # Calcula funcao de auto-correlacao parcial (uma para cada mes)
        self.FACP = np.zeros((12, ord_max), 'd')
        for ilag in np.arange(1, ord_max+1):
            for imes in range(12):
                A = np.eye(ilag)
                B = np.zeros(ilag)
                # Preenche matriz triangular superior
                for ilin in range(len(A)):
                    for icol in range(len(A)):           # TODO: Aqui poderia ser np.arange(ilin+1,len(A)): Testar depois
                        if icol > ilin:
                            mes = imes - ilin - 1
                            if mes < 0:
                               mes = mes + 12
                            A[ilin][icol] = self.FAC[mes][icol-ilin-1]
                    B[ilin] = self.FAC[imes][ilin]
                # Preenche matriz triangular inferior
                for ilin in range(len(A)):
                    for icol in range(len(A)):          # TODO: Aqui poderia ser np.arange(0, ilin): Testar depois
                        if icol < ilin:
                            A[ilin][icol] = A[icol][ilin]
                phi = np.linalg.solve(A, B)
                self.FACP[imes][ilag-1] = phi[-1]

        # Identificacao da ordem
        IC = 1.96/math.sqrt(nanos-1)
        self.Ordem = np.zeros(12, 'i')
        for imes in range(12):
            self.Ordem[imes] = 0
            for ilag in range(ord_max):
                if self.FACP[imes][ilag] > IC or self.FACP[imes][ilag] < -IC:
                    self.Ordem[imes] = ilag+1

        # ############## GRAFICO AUTOCORRELACAO #############
        # meses = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
        # for imes in range(12):
        #
        #     cores = []
        #     limitesup = []
        #     limiteinf = []
        #     for elemento in self.FACP[imes]:
        #         limitesup.append(IC)
        #         limiteinf.append(-IC)
        #         if elemento > IC or elemento < -IC:
        #             cores.append('r')
        #         else:
        #             cores.append('b')
        #
        #     f, ax2 = plt.subplots(1, 1, sharey=True)
        #     barWidth = 0.40
        #
        #     # titulo = 'Função de Autocorrelação Parcial - ' + meses[imes]
        #     # f.canvas.set_window_title(titulo)
        #
        #     # ax1.bar(np.arange(1, ord_max + 1), self.FAC[imes], barWidth, align='center')
        #     ax2.bar(np.arange(1, ord_max + 1), self.FACP[imes], barWidth, align='center', color=cores)
        #     ax2.plot(np.arange(1, ord_max + 1), limitesup, 'm--', lw=1, label='$IC=\dfrac{1,96}{\sqrt{N}}$')
        #     ax2.plot(np.arange(1, ord_max + 1), limiteinf, 'm--', lw=1)
        #
        #     # ax1.set_xticks(np.arange(1, ord_max + 1))
        #     ax2.set_xticks(np.arange(1, ord_max + 1))
        #     # tituloFAC = 'Função Autocorrelação'
        #     tituloFACP = 'Função Autocorrelação Parcial - ' + meses[imes]
        #     # # ax1.set_title(tituloFAC, fontsize=13)
        #     ax2.set_title(tituloFACP, fontsize=13)
        #     # ax1.set_xlabel('Lag')
        #     ax2.set_xlabel('Lag')
        #     plt.legend()
        #     plt.tight_layout()
        #     # ax1.ylabel('Autocorrelacao e Autocorrelacao Parcial')
        #
        #     plt.show()
        #
        #     galo = 13

        ###################################################

        # Calculo dos coeficientes
        self.CoefParp = np.zeros((12, ord_max), 'd')
        for imes in range(12):
            ilag = self.Ordem[imes]
            A = np.eye(ilag)
            B = np.zeros(ilag)
            # Preenche matriz triangular superior
            for ilin in range(len(A)):
                for icol in range(len(A)):             # TODO: Aqui poderia ser np.arange(ilin+1,len(A)): Testar depois
                    if icol > ilin:
                        mes = imes - ilin - 1
                        if mes < 0:
                           mes = mes + 12
                        A[ilin][icol] = self.FAC[mes][icol-ilin-1]
                B[ilin] = self.FAC[imes][ilin]
            # Preenche matriz triangular inferior
            for ilin in range(len(A)):
                for icol in range(len(A)):             # TODO: Aqui poderia ser np.arange(0, ilin): Testar depois
                    if icol < ilin:
                        A[ilin][icol] = A[icol][ilin]
            phi = np.linalg.solve(A, B)
            for iord in range(len(phi)):
                self.CoefParp[imes][iord] = phi[iord]

        print('Aplicação do PAR(p): OK')

    def parp_bat(self, dados, ord_max):

        np.random.seed(1234)

        # Funcao objetivo minimizar erro quatratico medio (o residuo eh o erro)
        def bat(dados, D, imes, Np, MaxIter, Amp, Pul, lb, ub):

            Q = np.zeros(Np)  # frequency
            Amplitude = np.ones(Np)  # vetor de loudness
            Pulso = np.zeros(Np)  # vetor de rate

            v = np.zeros((Np, D))  # velocity
            Sol = np.zeros((Np, D))  # population of solutions
            Fitness = np.zeros(Np)  # fitness

            # Incializacao das particulas, Amplitude, Pulso e Fitness
            for i in range(Np):
                Sol[i] = lb + (ub - lb) * np.random.uniform(0, 1, D)
                Fitness[i] = Fun(Sol[i], dados, D, imes)

            # Encontrar melhor solucao
            idx = np.argmin(Fitness)
            best = Sol[idx]

            # Inicio do Bat Algorithm
            iter = 0
            fmin = 99999999
            while iter < MaxIter and fmin > 0.001:

                for w in range(Np):

                    Q[w] = np.random.uniform(0, 1)
                    v[w] = v[w] + (Sol[w] - best) * Q[w]
                    SolTemp = Sol[w] + v[w]

                    if np.random.uniform(0, 1) > Pulso[w]:

                        for j in range(D):
                            SolTemp[j] = best[j] + (-1 + 2 * np.random.random_sample()) * np.mean(Amplitude)
                            SolTemp[j] = verifica_limites(SolTemp[j], lb[j], ub[j])

                    for j in range(D):
                        SolTemp[j] = verifica_limites(SolTemp[j], lb[j], ub[j])

                    Fnew = Fun(SolTemp, dados, D, imes)

                    if Fnew <= Fitness[w] or np.random.uniform(0, 1) < Amplitude[w]:
                        Sol[w] = SolTemp
                        Fitness[w] = Fnew
                        Pulso[w] = 1 - np.exp(-Pul * (iter + 1))
                        Amplitude[w] = Amp * Amplitude[w]

                    if Fnew < fmin:
                        fmin = Fnew
                        best = SolTemp

                iter += 1

            return fmin, best

        def verifica_limites(sol, lb, ub):

            if sol > ub:
                sol = ub

            if sol < lb:
                sol = lb

            return sol

        def Fun(x, dados, ord_max, imes):

            nanos = len(dados)

            media = np.mean(dados, axis=0)
            desv_pad = np.std(dados, axis=0)

            # Calcula funcao de auto-correlacao (uma para cada mes)
            FAC = np.zeros(ord_max, 'd')
            for ilag in range(ord_max):
                for iano in np.arange(1, nanos):
                    ano_ant = iano
                    mes_ant = imes - ilag - 1
                    if mes_ant < 0:
                        ano_ant -= 1
                        mes_ant += 12
                    FAC[ilag] += (dados[iano][imes] - media[imes]) * (dados[ano_ant][mes_ant] - media[mes_ant])
                FAC[ilag] /= ((nanos - 1) * desv_pad[imes] * desv_pad[mes_ant])

            # # Calcula a variância do ruido
            variancia_ruido = 1
            for icoef in range(ord_max):
                variancia_ruido -= x[icoef] * FAC[icoef]
            # variancia_ruido = (desv_pad[imes] ** 2) * (1 + np.sum(FAC))  # outra fórmula
            if variancia_ruido <= 0 or variancia_ruido > 0.001*(desv_pad[imes]**2):
                penal = 99999   # variancia_ruido = 1e-6
            else:
                penal = 0

            # Calcula a variância do ruido (método de Ricardo Reis, doutorado USP)
            # variancia_ruido = (desv_pad[imes] **2) * (1 + np.sum(FAC))

            residuos = np.zeros(nanos - 1)
            for iano in np.arange(1, nanos):
                somatorio = (dados[iano][imes] - media[imes]) / desv_pad[imes]
                for ilag in range(ord_max):
                    mes_ant = imes - ilag - 1
                    ano_ant = iano
                    if mes_ant < 0:
                        mes_ant += 12
                        ano_ant -= 1
                    somatorio -= x[ilag] * ((dados[ano_ant][mes_ant] - media[mes_ant]) / desv_pad[mes_ant])
                residuos[iano - 1] = somatorio

            # Autocorrelação
            autocorrelacao = 0
            # ord_max = int(nanos/2)
            # ord_max = nanos-2
            ord_max = 6
            for ilag in range(ord_max):
                somatorio = 0
                for iano in np.arange(ilag, nanos-1):
                    somatorio += residuos[iano]*residuos[iano-ilag-1]
                somatorio /= (nanos-1-ilag)
                autocorrelacao += np.abs(somatorio)

            var_ruido_norm = (desv_pad / np.max(desv_pad)) ** 2

            fob = np.abs(np.var(residuos)) + np.abs(np.mean(residuos)) + autocorrelacao  # - variancia_ruido
            fob = fob + penal

            return fob

        nanos = len(dados)

        media = np.mean(dados, axis=0)
        desvio = np.std(dados, axis=0)

        # Calcula funcao de auto-correlacao (uma para cada mes)
        self.FAC = np.zeros((12, ord_max), 'd')
        for ilag in range(ord_max):
            for imes in range(12):
                for iano in np.arange(1, nanos):
                    ano_ant = iano
                    mes_ant = imes - ilag - 1
                    if mes_ant < 0:
                        ano_ant -= 1
                        mes_ant += 12
                    self.FAC[imes][ilag] += (dados[iano][imes] - media[imes]) * (
                                dados[ano_ant][mes_ant] - media[mes_ant])
                self.FAC[imes][ilag] /= ((nanos - 1) * desvio[imes] * desvio[mes_ant])

        self.CoefParp = np.zeros((12, ord_max), 'd')
        self.Ordem = np.zeros(12, 'i')  # comentar se for usar outro metodo
        self.FOB = np.zeros(12, 'd')
        print('Modelo PAR(p)-BAT em execução....')
        for imes in range(12):
            # print('*******', imes + 1)
            best = 99999999
            t = timer()
            # for iord in np.arange(self.Ordem[imes], self.Ordem[imes]+1):
            for iord in np.arange(1, ord_max + 1):

                # Define limites e condicao inicial
                lb = np.zeros(iord)
                ub = np.zeros(iord)
                for i in range(iord):
                    lb[i] = -2
                    ub[i] = 2

                fitness, coef = bat(dados, iord, imes, 50, 20, 0.5, 0.1, lb, ub)

                if fitness < best:
                    best = fitness
                    ordem = iord
                    COEF = coef

            for iord in range(ordem):
                self.CoefParp[imes, iord] = COEF[iord]
            self.Ordem[imes] = ordem
            self.FOB[imes] = best

            print('Mês ' + str(imes+1) + ' - Tempo: ', round(timer() - t, 2), 'seg')

        print('Aplicação do PAR(p)-BAT: OK')

    def gera_series_sinteticas(self, dados, nr_ser, nr_meses):

        # np.random.seed(1234)

        media = np.mean(dados, axis=0)
        desvio = np.std(dados, axis=0)

        desvio_ruido = np.ones(12)
        for imes in range(12):
            for icoef in range(int(self.Ordem[imes])):
                desvio_ruido[imes] -= self.CoefParp[imes][icoef] * self.FAC[imes][icoef]
            desvio_ruido[imes] = np.sqrt(desvio_ruido[imes])

        # Gera series sinteticas
        sintetica_adit = np.zeros((nr_ser, nr_meses), 'd')
        for iser in range(nr_ser):
            contador = -1
            for iano in range(int(nr_meses/12)):
                for imes in range(12):
                    contador += 1
                    delta = - media[imes] / desvio[imes]
                    valor = media[imes]
                    for ilag in range(int(self.Ordem[imes])):
                        mes_ant = imes - ilag - 1
                        ano_ant = iano
                        if mes_ant < 0:
                            mes_ant += 12
                            ano_ant -= 1
                        if ano_ant < 0:
                            ventoant = media[mes_ant]
                        else:
                            ventoant = sintetica_adit[iser][contador-ilag-1]
                        delta -= self.CoefParp[imes][ilag]*(ventoant-media[mes_ant])/desvio[mes_ant]
                        valor += desvio[imes]*self.CoefParp[imes][ilag]*(ventoant-media[mes_ant])/desvio[mes_ant]
                    teta = 1 + ((desvio_ruido[imes] ** 2) / ((-delta) ** 2))
                    mu = (1 / 2) * np.log((desvio_ruido[imes]**2)/(teta*(teta-1)))
                    sigma = np.sqrt(np.log(teta))
                    epsilon = np.random.normal(mu, sigma, 1)
                    ruido = np.exp(epsilon) + delta    #  desvio_ruido[imes] * np.random.normal(0, 1, 1)
                    valor += desvio[imes] * ruido
                    sintetica_adit[iser][contador] = valor

        x_axis = np.arange(1, nr_meses + 1)
        plt.figure()
        plt.plot(x_axis, sintetica_adit.transpose(), color='silver', linestyle='-')
        plt.plot(x_axis, np.mean(sintetica_adit, 0), 'k-', lw=3, label='Média - Séries Sintéticas')
        plt.plot(x_axis, np.mean(sintetica_adit, 0) + np.nanstd(sintetica_adit, axis=0), 'k--', lw=2,
                 label='Desvio Padrão - Séries Sintéticas')
        plt.plot(x_axis, np.mean(sintetica_adit, 0) - np.nanstd(sintetica_adit, axis=0), 'k--', lw=2)
        m = media
        d = desvio
        for iano in range(int(nr_meses/12)-1):
            m = np.concatenate([m, media])
            d = np.concatenate([d, desvio])
        plt.plot(x_axis, m, 'ro', lw=3, label='Média - Série Histórica')
        plt.plot(x_axis, m + d, 'bo', lw=2, label='Desvio Padrão - Série Histórica')
        plt.plot(x_axis, m - d, 'bo', lw=2)
        titulo = "Séries Sintéticas de Densidade de Potência do Vento"
        plt.title(titulo, fontsize=12)
        plt.xlabel('Meses', fontsize=10)
        plt.ylabel('$KW/m^{2}$', fontsize=10)
        plt.legend(fontsize=10)
        plt.tight_layout()
        # plt.show()

        self.series_sinteticas = sintetica_adit

        print('Geração de séries sintéticas: OK')

    def gera_series_sinteticas_sem_tendencia(self, dados, nr_ser, nr_meses, plot: bool = False):

        np.random.seed(1234)

        nanos = len(dados)
        media = np.mean(dados, axis=0)
        desvio = np.std(dados, axis=0)

        # Calculo dos residuos
        self.residuos = np.zeros((nanos-1, 12))
        for iano in np.arange(1, nanos):
            for imes in range(12):
                self.residuos[iano-1][imes] = (dados[iano][imes] - media[imes]) / desvio[imes]
                somatorio = 0
                for ilag in np.arange(1, self.Ordem[imes]):
                    ano_ant = iano
                    mes_ant = imes - ilag
                    if mes_ant < 0:
                        ano_ant -= 1
                        mes_ant += 12
                    somatorio += self.CoefParp[imes][ilag] * (dados[ano_ant][mes_ant] - media[mes_ant]) / desvio[mes_ant]
                self.residuos[iano - 1][imes] = self.residuos[iano - 1][imes] - somatorio

        desvio_ruido = np.ones(12)
        for imes in range(12):
            for icoef in range(int(self.Ordem[imes])):
                desvio_ruido[imes] -= self.CoefParp[imes][icoef] * self.FAC[imes][icoef]
            desvio_ruido[imes] = np.sqrt(desvio_ruido[imes])

        # Gera series sinteticas
        sintetica_adit = np.zeros((nr_ser, nr_meses), 'd')
        for irod in range(2):
            for iser in range(nr_ser):
                contador = -1
                for iano in range(int(nr_meses/12)):
                    for imes in range(12):
                        contador += 1
                        delta = - media[imes] / desvio[imes]
                        valor = media[imes]
                        somatorio = 0
                        for ilag in range(int(self.Ordem[imes])):
                            mes_ant = imes - ilag - 1
                            ano_ant = iano
                            if mes_ant < 0:
                                mes_ant += 12
                                ano_ant -= 1
                            if (ano_ant < 0) and (irod == 0):
                                ventoant = media[mes_ant]
                            else:
                                ventoant = sintetica_adit[iser][contador - ilag - 1]
                            delta -= self.CoefParp[imes][ilag]*(ventoant-media[mes_ant])/desvio[mes_ant]
                            somatorio += self.CoefParp[imes][ilag]*(ventoant-media[mes_ant])/desvio[mes_ant]
                        valor += desvio[imes]*somatorio
                        teta = 1 + ((desvio_ruido[imes] ** 2) / ((-delta) ** 2))
                        mu = (1 / 2) * np.log((desvio_ruido[imes]**2)/(teta*(teta-1)))
                        # mu = 0
                        sigma = np.sqrt(np.log(teta)) #np.sqrt()
                        epsilon = np.random.normal(mu, sigma, 1)
                        ruido = np.exp(epsilon) + delta
                        valor += desvio[imes] * ruido
                        sintetica_adit[iser][contador] = valor

        if plot:

            x_axis = np.arange(1, nr_meses + 1)
            plt.figure()
            plt.plot(x_axis, sintetica_adit.transpose(), color='silver', linestyle='-')
            plt.plot(x_axis, np.mean(sintetica_adit, 0), 'k-', lw=3, label='Média - Séries Sintéticas')
            plt.plot(x_axis, np.mean(sintetica_adit, 0) + np.nanstd(sintetica_adit, axis=0), 'k--', lw=2,
                     label='Desvio Padrão - Séries Sintéticas')
            plt.plot(x_axis, np.mean(sintetica_adit, 0) - np.nanstd(sintetica_adit, axis=0), 'k--', lw=2)
            m = media
            d = desvio
            for iano in range(int(nr_meses / 12) - 1):
                m = np.concatenate([m, media])
                d = np.concatenate([d, desvio])
            plt.plot(x_axis, m, 'ro', lw=3, label='Média - Série Histórica')
            plt.plot(x_axis, m + d, 'bo', lw=2, label='Desvio Padrão - Série Histórica')
            plt.plot(x_axis, m - d, 'bo', lw=2)
            titulo = "Séries Sintéticas de Densidade de Potência do Vento"
            plt.title(titulo, fontsize=12)
            plt.xlabel('Meses', fontsize=10)
            plt.ylabel('$KW/m^{2}$', fontsize=10)
            # plt.legend(loc='upper center', fontsize=10, ncol=2)
            plt.tight_layout()
            # plt.show()

        self.series_sinteticas = sintetica_adit

        print('Geração de séries sintéticas sem tendência: OK')

    def Teste_de_Media(self, dados):

        # Vetorizacao da media mensal historica de vazoes, tendo em vista o numero de estagios de analise
        nestagios = self.series_sinteticas.shape[1]
        aprovados = 0

        # Realizacao do Teste T
        Teste = np.zeros((1, nestagios))
        cont = 0
        for iteste in range(nestagios):
            aux = [dados[i][cont] for i in range(len(dados))]
            a = aux
            b = self.series_sinteticas[:, iteste]
            t_valor, p_valor = stats.ttest_ind(a, b, equal_var=True)
            Teste[0, iteste] = p_valor*100

            # Verificacao da quantidade de valores aprovados
            if p_valor >= float(0.05):
                aprovados += 1

            if (cont < 11):
                cont = cont + 1
            else:
                cont = 0

        # Aprovação total da serie
        porcentagem = int((aprovados/nestagios)*100)
        # print("Resultado do Teste de Média das Séries Sintéticas Geradas: ", porcentagem,"% aprovados.")

        # # Grafico analitico
        # y_axis = Teste[0, :]
        # x_axis = np.arange(1, nestagios+1)
        # k_axis = np.zeros((nestagios, 1))
        # for iplot in range(nestagios):
        #     k_axis[iplot, 0] = int(5)
        # width_n = 0.9
        # bar_color = 'gray'
        # plt_color = 'red'
        # plt.bar(x_axis, y_axis, width=width_n, color=bar_color, label=str(porcentagem)+"% Aprovados")
        # plt.plot(x_axis, k_axis, color=plt_color)
        # titulo = "TESTE DE MÉDIA (Teste t)"
        # plt.title(titulo, fontsize=16)
        # plt.xlabel('Meses', fontsize=16)
        # plt.ylabel('p-valor (%)', fontsize=16)
        # plt.ylim(0,100)
        # plt.xlim(0,nestagios+1)
        # plt.legend(fontsize=12)
        # plt.show()

        # Retorno dos resultados obtidos pelo teste
        return porcentagem

    def Teste_de_Variancia(self, dados):

        # Vetorizacao da media mensal historica de vazoes, tendo em vista o numero de estagios de analise
        nestagios = self.series_sinteticas.shape[1]
        aprovados = 0

        # Realizacao do Teste de Levene
        Teste= np.zeros((1,nestagios))
        cont = 0
        for iteste in range(nestagios):
            aux = [dados[i][cont] for i in range(len(dados))]
            a = aux
            b = self.series_sinteticas[:, iteste]
            t_valor, p_valor = stats.bartlett(a, b)
            Teste[0, iteste] = p_valor*100

            # Verificacao da quantidade de valores aprovados
            if p_valor >= float(0.05):
                aprovados += 1

            if (cont < 11):
                cont = cont + 1
            else:
                cont = 0

        # Aprovação total da serie
        porcentagem = int((aprovados/nestagios)*100)
        # print("Resultado do Teste de Variância das Séries Sintéticas Geradas: ", porcentagem,"% aprovados.")

        # Grafico analitico
        # plt.figure()
        # y_axis = Teste[0, :]
        # x_axis = np.arange(1, nestagios+1)
        # k_axis = np.zeros((nestagios, 1))
        # for iplot in range(nestagios):
        #     k_axis[iplot,0] = int(5)
        # width_n = 0.9
        # bar_color = 'gray'
        # plt_color = 'red'
        # plt.bar(x_axis, y_axis, width=width_n, color=bar_color, label=str(porcentagem)+"% Aprovados")
        # plt.plot(x_axis, k_axis, color=plt_color)
        # titulo = "TESTE DE VARIÂNCIA (Bartlett)"
        # plt.title(titulo, fontsize=16)
        # plt.xlabel('Meses', fontsize=16)
        # plt.ylabel('p-valor (%)', fontsize=16)
        # plt.ylim(0, 100)
        # plt.xlim(0, nestagios+1)
        # plt.legend(fontsize=12)
        # plt.show()
        return porcentagem

    def Teste_de_Aderencia(self, dados):

        # Vetorizacao da media mensal historica de vazoes, tendo em vista o numero de estagios de analise
        nestagios = self.series_sinteticas.shape[1]
        aprovados = 0

        # Realizacao do Teste T
        Teste = np.zeros((1, nestagios))
        cont = 0
        for iteste in range(nestagios):
            aux = [dados[i][cont] for i in range(len(dados))]
            a = aux
            b = self.series_sinteticas[:, iteste]
            t_valor, p_valor = stats.ks_2samp(a, b)
            Teste[0, iteste] = p_valor*100

            # Verificacao da quantidade de valores aprovados
            if p_valor >= float(0.05):
                aprovados += 1

            if (cont < 11):
                cont = cont + 1
            else:
                cont = 0

        # Aprovação total da serie
        porcentagem = int((aprovados/nestagios)*100)
        # print("Resultado do Teste de Aderência das Séries Sintéticas Geradas: ", porcentagem,"% aprovados.")

        # # Grafico analitico
        # y_axis = Teste[0, :]
        # x_axis = np.arange(1, nestagios+1)
        # k_axis = np.zeros((nestagios,1))
        # for iplot in range(nestagios):
        #     k_axis[iplot,0] = int(5)
        # width_n = 0.9
        # bar_color = 'gray'
        # plt_color = 'red'
        # plt.bar(x_axis, y_axis, width=width_n, color=bar_color, label=str(porcentagem)+"% Aprovados")
        # plt.plot(x_axis, k_axis, color=plt_color)
        # titulo = "TESTE DE ADERÊNCIA (Kolmogorov-Smirnov)"
        # plt.title(titulo, fontsize=16)
        # plt.xlabel('Meses', fontsize=16)
        # plt.ylabel('p-valor (%)', fontsize=16)
        # plt.ylim(0,100)
        # plt.xlim(0,nestagios+1)
        # plt.legend(fontsize=12)
        # plt.show()
        return porcentagem

    def Teste_de_Mediana(self, dados):

        # Vetorizacao da media mensal historica de vazoes, tendo em vista o numero de estagios de analise
        nestagios = self.series_sinteticas.shape[1]

        # Realizacao do Teste de Wilcoxon
        aprovados = 0
        Teste = np.zeros((1, nestagios))
        cont = 0
        for iteste in range(nestagios):
            aux = [dados[i][cont] for i in range(len(dados))]
            a = aux
            b = self.series_sinteticas[:, iteste]
            t_valor, p_valor = stats.ranksums(a,b)
            Teste[0, iteste] = p_valor * 100

            # Verificacao da quantidade de valores aprovados
            if p_valor >= float(0.05):
                aprovados += 1

            if (cont < 11):
                cont = cont + 1
            else:
                cont = 0

        # Aprovação total da serie
        porcentagem = int((aprovados / nestagios) * 100)
        # print("Resultado do Teste de Mediana das Séries Sintéticas Geradas: ", porcentagem,"% aprovados.")

        # # Grafico analitico
        # y_axis = Teste[0, :]
        # x_axis = np.arange(1, nestagios+1)
        # k_axis = np.zeros((nestagios, 1))
        # for iplot in range(nestagios):
        #     k_axis[iplot, 0] = int(5)
        # width_n = 0.9
        # bar_color = 'gray'
        # plt_color = 'red'
        # plt.bar(x_axis, y_axis, width=width_n, color=bar_color, label=str(porcentagem) + "% Aprovados")
        # plt.plot(x_axis, k_axis, color=plt_color)
        # titulo = "TESTE DE MEDIANA (Wilcoxon)"
        # plt.title(titulo, fontsize=16)
        # plt.xlabel('Meses', fontsize=16)
        # plt.ylabel('p-valor (%)', fontsize=16)
        # plt.ylim(0, 100)
        # plt.xlim(0, nestagios+1)
        # plt.legend(fontsize=12)
        # plt.show()

        # Retorno dos resultados obtidos pelo teste
        #return Teste_Mediana
        return porcentagem

    def Teste_de_Assimetria(self, dados):

        # Vetorizacao da media mensal historica de vazoes, tendo em vista o numero de estagios de analise
        nestagios = self.series_sinteticas.shape[1]

        # Realizacao do Teste de Assimetria
        aprovados = 0
        Teste = np.zeros((1, nestagios))
        cont = 0
        for iteste in range(nestagios):
            aux = [dados[i][cont] for i in range(len(dados))]
            a = aux
            b = self.series_sinteticas[:, iteste]
            z_valor1, p_valor1 = stats.skewtest(a)
            z_valor2, p_valor2 = stats.skewtest(b)
            aux = p_valor1 - p_valor2
            max_v = max(p_valor1, p_valor2)
            Teste[0, iteste] = abs(aux / max_v) * 100

            # Verificacao da quantidade de valores aprovados
            if Teste[0, iteste] >= float(40.):
                aprovados += 1

            if (cont < 11):
                cont = cont + 1
            else:
                cont = 0

        # Aprovação total da serie
        porcentagem = int((aprovados / nestagios) * 100)
        # print("Resultado do Teste de Assimetria das Séries Sintéticas Geradas: ", porcentagem, "% aprovados.")

        # Grafico analitico
        # y_axis = Teste[0, :]
        # x_axis = np.arange(1, nestagios+1)
        # k_axis = np.zeros((nestagios, 1))
        # for iplot in range(nestagios):
        #     k_axis[iplot, 0] = int(5)
        # width_n = 0.9
        # bar_color = 'gray'
        # plt_color = 'red'
        # plt.bar(x_axis, y_axis, width=width_n, color=bar_color, label=str(porcentagem) + "% Aprovados")
        # plt.plot(x_axis, k_axis, color=plt_color)
        # titulo = "TESTE DE ASSIMETRIA"
        # plt.title(titulo, fontsize=16)
        # plt.xlabel('Meses', fontsize=16)
        # plt.ylabel('p-valor (%)', fontsize=16)
        # plt.ylim(0, 100)
        # plt.xlim(0, nestagios+1)
        # plt.legend(fontsize=12)
        # plt.show()

        # Retorno dos resultados obtidos pelo teste
        # return Teste_Assimetria
        return porcentagem

    def Teste_de_Sequencia_Negativa(self, dados):

        # Determinação dos parâmetros de sequêcia negativa do histórico
        dados = np.asanyarray(dados)
        dados_historico = np.reshape(dados, np.size(dados))
        media_historico = np.tile(np.mean(dados, axis=0), len(dados))
        aux = []
        for k in range(len(dados_historico)):
            if dados_historico[k] < media_historico[k]:  # alterado <=
                aux.append(k)
        aux = np.asarray(aux, 'i')
        seq_negativa = np.split(aux, np.where(np.diff(aux) != 1)[0] + 1)

        comprimento_historico = np.zeros(len(seq_negativa))
        soma_historico = np.zeros(len(seq_negativa))
        intensidade_historico = np.zeros(len(seq_negativa))
        for k in range(len(seq_negativa)):
            comprimento_historico[k] = seq_negativa[k].size
            soma_historico[k] = np.sum(dados_historico[seq_negativa[k]] - media_historico[seq_negativa[k]])
            intensidade_historico[k] = soma_historico[k] / comprimento_historico[k]

        # Determinação dos parâmetros de sequêcia negativa das séries geradas
        dados_series = np.reshape(self.series_sinteticas, np.size(self.series_sinteticas))
        media_historico = np.tile(np.mean(dados, axis=0), int(self.series_sinteticas.shape[0]*self.series_sinteticas.shape[1]/12))
        aux = []
        for k in range(len(dados_series)):
            if dados_series[k] < media_historico[k]: # alterado <=
                aux.append(k)
        aux = np.asarray(aux, 'i')
        seq_negativa = np.split(aux, np.where(np.diff(aux) != 1)[0] + 1)

        comprimento_series = np.zeros(len(seq_negativa))
        soma_series = np.zeros(len(seq_negativa))
        intensidade_series = np.zeros(len(seq_negativa))
        for k in range(len(seq_negativa)):
            comprimento_series[k] = seq_negativa[k].size
            soma_series[k] = np.sum(dados_series[seq_negativa[k]] - media_historico[seq_negativa[k]])
            intensidade_series[k] = soma_series[k] / comprimento_series[k]

        # Preparação dos dados para o teste Qui^2 (aplicado ao comprimento de sequencia negativa)
        max_comp = np.maximum(np.max(comprimento_series), np.max(comprimento_historico))
        min_comp = np.minimum(np.min(comprimento_series), np.min(comprimento_historico))
        num_classes = int((max_comp-min_comp)/3)  # Alterado: correto 3
        data = np.zeros((2, int(num_classes)))
        classes = np.zeros((num_classes, 2))
        aux = min_comp-1
        for icl in np.arange(1, num_classes+1):
            if (icl/num_classes) <= 0.6:
                classes[icl-1, :] = [aux, aux+1]
                aux += 1
            elif (icl/num_classes > 0.6) and (icl/num_classes <= 0.9):
                classes[icl - 1, :] = [aux, aux + 2]
                aux += 2
            if icl == num_classes:
                classes[icl - 1, :] = [aux, max_comp]

        for k in range(num_classes):
            quant_historico = np.size(np.where(np.logical_and(comprimento_historico > classes[k, 0], comprimento_historico <= classes[k, 1])))
            quant_series = np.size(np.where(np.logical_and(comprimento_series > classes[k, 0], comprimento_series <= classes[k, 1])))
            data[:, k] = [quant_historico, quant_series]
        # data[:, -1] = np.sum(data[:, :-1], axis=1)
        # Deletar colunas com zeros em ambas distribuições
        ind_del = [i for i in range(data.shape[1]) if ((data[0, i] == 0) and (data[1, i] == 0))]
        data = np.delete(data, ind_del, axis=1)

        # Aplicação do teste de Qui^2
        qui2, p, dof, ex = stats.chi2_contingency(data, correction=False)
        # Valor crítico
        valor_critico = chi2.ppf(0.95, dof)
        # print('-------------------------------------------------------------')
        # print('Teste de sequência negativa - Comprimento (Chi Quadrado)')
        if qui2 <= valor_critico:
            # print('Aprovado no teste!!!!')
            # print('chi2 = %4.3f <= %4.3f = valor crítico' % (qui2, valor_critico))
            cond = 'chi2 = %4.3f <= %4.3f = valor crítico' % (qui2, valor_critico)
            res_comp =['Aprovado', cond]
        else:
            cond = 'chi2 = %4.3f > %4.3f = valor crítico' % (qui2, valor_critico)
            res_comp = ['Reprovado', cond]

            # print('Reprovado no teste!!!!')
        #     print('chi2 = %4.3f > %4.3f = valor crítico' % (qui2, valor_critico))
        # print('-------------------------------------------------------------')

        # Aplicação do teste de Kolmogorov-Smirnov (Soma)
        ks, p = stats.ks_2samp(soma_historico, soma_series)
        # Valor crítico
        valor_critico = 1.36*np.sqrt((soma_historico.size + soma_series.size) / (soma_historico.size * soma_series.size))
        # print('Teste de sequência negativa - Soma (Kolmogorov-Smirnov)')
        if ks <= valor_critico:
            # print('Aprovado no teste!!!!')
            # print('ks = %4.3f <= %4.3f = valor crítico' % (ks, valor_critico))
            cond = 'ks = %4.3f <= %4.3f = valor crítico' % (ks, valor_critico)
            res_soma = ['Aprovado', cond]
        else:
            cond = 'ks = %4.3f > %4.3f = valor crítico' % (ks, valor_critico)
            res_soma = ['Reprovado', cond]
        #     print('Reprovado no teste!!!!')
        #     print('ks = %4.3f > %4.3f = valor crítico' % (ks, valor_critico))
        # print('-------------------------------------------------------------')

        # Aplicação do teste de Kolmogorov-Smirnov (Intensidade)
        ks, p = stats.ks_2samp(intensidade_historico, intensidade_series)
        # Valor crítico
        valor_critico = 1.36*np.sqrt((intensidade_historico.size + intensidade_series.size) / (intensidade_historico.size * intensidade_series.size))
        # print('Teste de sequência negativa - Intensidade (Kolmogorov-Smirnov)')
        if ks <= valor_critico:
            cond = 'ks = %4.3f <= %4.3f = valor crítico' % (ks, valor_critico)
            res_int = ['Aprovado', cond]
            # print('Aprovado no teste!!!!')
            # print('ks = %4.3f <= %4.3f = valor crítico' % (ks, valor_critico))
        else:
            cond = 'ks = %4.3f > %4.3f = valor crítico' % (ks, valor_critico)
            res_int = ['Reprovado', cond]
        #     print('Reprovado no teste!!!!')
        #     print('ks = %4.3f > %4.3f = valor crítico' % (ks, valor_critico))
        # print('-------------------------------------------------------------')

        return res_comp, res_soma, res_int
