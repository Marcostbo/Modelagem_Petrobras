# coding=utf-8

from Modules.EnergyPlanning.fcf import fcf
from Modules.EnergyPlanning.MosekOptimization import MosekProcess

from numpy import ndarray
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer
import mosek
import sys
import multiprocessing
# from itertools import product


const_vert = 0.00001
const_ecx = 0.00001


class pdde_SistIsol(object):

    sist = None
    cf = None
    ZINF = list()
    ZSUP = list()
    iter = 0
    sigma = list()
    CI_serie = None

    def __init__(self, sistema, nr_months, aberturas, pesos_aber, afluencias, eolicas, pesos_eol, submarket_index, nr_process: int):
        self.sist = sistema
        self.nr_months = nr_months
        self.aberturas = aberturas
        self.pesos_aber = pesos_aber
        self.afluencias = afluencias
        self.eolicas = eolicas
        self.pesos_eol = pesos_eol
        self.submarket_index = submarket_index
        self.nr_process = nr_process
        tx_mensal = (1 + self.sist.dger.TaxaDesconto) ** (1 / 12) - 1
        # self.tx_desc_mensal = 1 / (1 + tx_mensal)
        self.tx_desc_mensal = 1

    def forward_solve(self, iser, Aeq, Beq, A, B, A_GH, B_GH, c, lb, ub, EAIni, imes, ano, mes) -> tuple:

        nr_sist = len(self.submarket_index)
        nr_cen_eol = self.eolicas.shape[1]

        # Preenche Beq e B
        col_1 = nr_sist
        col_2 = 0
        for ieol in range(nr_cen_eol):
            for isist in range(nr_sist):
                submercado = self.sist.submercado[self.submarket_index[isist]]
                EA_vetor = np.array([EAIni[isist, iser, imes] ** 2, EAIni[isist, iser, imes], 1])
                if ieol == 0:
                    Beq[isist, 0] = EAIni[isist, iser, imes] + np.dot(submercado.ParamFC[:, mes], EA_vetor) * \
                                    submercado.FatorSeparacao * self.afluencias[isist, iser, imes] - np.dot(submercado.ParamEVMin[:, mes], EA_vetor) - \
                                    np.dot(submercado.ParamEVP[:, mes], EA_vetor) - submercado.EVM[ano, mes] - \
                                    submercado.EDESVC[ano, mes]

                Beq[col_1, 0] = submercado.Mercado[ano, mes] - submercado.NaoSimuladas[ano, mes] - submercado.GTMIN[
                    ano, mes] - \
                                (1 - submercado.FatorSeparacao) * self.afluencias[isist, iser, imes] - np.dot(
                    submercado.ParamEVMin[:, mes], EA_vetor) \
                                - self.eolicas[isist, ieol, imes] - submercado.EDESVF[ano, mes]
                B_GH[col_2, 0] = np.dot(submercado.ParamGHMAX, EA_vetor) - (1 - submercado.FatorSeparacao) * \
                                 self.afluencias[isist, iser, imes] - np.dot(submercado.ParamEVMin[:, mes], EA_vetor)
                col_1 += 1
                col_2 += 1

        if A is not None and B is not None:
            A_full = np.concatenate((A_GH, A), axis=0)
            B_full = np.concatenate((B_GH, B), axis=0)
        else:
            A_full = A_GH
            B_full = B_GH

        # tempo = timer()
        [X, _, FVAL] = MosekProcess(Aeq, Beq, A_full, B_full, c, lb, ub).run()
        # print(f'Iteração {self.iter} ->  Tempo: {round(timer() - tempo, 3)} seg (FORWARD)')
        # # print('mes: ', imes, '    ser: ', iser,)

        # Atualiza armazenamento das hidrelétricas
        EAFin = np.array(X[:nr_sist])

        CustoImed = FVAL - const_vert * sum(X[nr_sist:2 * nr_sist]) - const_ecx * sum(
            X[(2 + nr_cen_eol) * nr_sist:(2 + 2 * nr_cen_eol) * nr_sist]) - self.tx_desc_mensal*X[-1]
        CustoFut = self.tx_desc_mensal*X[-1]

        return iser, EAFin, CustoImed, CustoFut

    def backward_solve(self, iser, Aeq, Beq, A, B, A_GH, B_GH, c, lb, ub, EAIni, imes, ano, mes) -> tuple:

        nr_aber = self.aberturas.shape[1]
        nr_sist = len(self.submarket_index)
        nr_cen_eol = self.eolicas.shape[1]

        LAMBDA = np.zeros((nr_aber, nr_sist))
        CUSTO = np.zeros((nr_aber, 1))
        for iaber in range(nr_aber):

            # Preenche Beq e B
            col_1 = nr_sist
            col_2 = 0
            for ieol in range(nr_cen_eol):

                for isist in range(nr_sist):
                    submercado = self.sist.submercado[self.submarket_index[isist]]
                    EA_vetor = np.array([EAIni[isist, iser, imes] ** 2, EAIni[isist, iser, imes], 1])
                    if ieol == 0:
                        Beq[isist, 0] = EAIni[isist, iser, imes] + np.dot(submercado.ParamFC[:, mes], EA_vetor) * \
                                        submercado.FatorSeparacao * self.aberturas[isist, iaber, imes] - np.dot(
                            submercado.ParamEVMin[:, mes], EA_vetor) - \
                                        np.dot(submercado.ParamEVP[:, mes], EA_vetor) - submercado.EVM[ano, mes] - \
                                        submercado.EDESVC[ano, mes]

                    Beq[col_1, 0] = submercado.Mercado[ano, mes] - submercado.NaoSimuladas[ano, mes] - submercado.GTMIN[
                        ano, mes] - (1 - submercado.FatorSeparacao) * self.aberturas[isist, iaber, imes] - np.dot(submercado.ParamEVMin[:, mes], EA_vetor) - \
                                    self.eolicas[isist, ieol, imes] - submercado.EDESVF[ano, mes]
                    B_GH[col_2, 0] = np.dot(submercado.ParamGHMAX, EA_vetor) - (1 - submercado.FatorSeparacao) * \
                                     self.aberturas[isist, iaber, imes] - np.dot(submercado.ParamEVMin[:, mes], EA_vetor)
                    col_1 += 1
                    col_2 += 1

            if A is not None and B is not None:
                A_full = np.concatenate((A_GH, A), axis=0)
                B_full = np.concatenate((B_GH, B), axis=0)
            else:
                A_full = A_GH
                B_full = B_GH

            # tempo = timer()
            [X, DUAL, FVAL] = MosekProcess(Aeq, Beq, A_full, B_full, c, lb, ub).run()
            # print(f'Iteração {self.iter} ->  Tempo: {round(timer() - tempo, 3)} seg (BACWARD)')
            # print('mes: ', imes, '    ser: ', iser, '     aber: ', iaber)

            CUSTO[iaber, 0] = FVAL - const_vert * sum(X[nr_sist:2 * nr_sist]) - const_ecx * sum(
                X[(2 + nr_cen_eol) * nr_sist:(2 + 2 * nr_cen_eol) * nr_sist])
            for isist in range(nr_sist):
                if DUAL[isist] > 0.:
                    LAMBDA[iaber, isist] = 0.
                else:
                    LAMBDA[iaber, isist] = DUAL[isist]

        lambda_mean = np.matmul(self.pesos_aber[imes, :], LAMBDA)
        custo_mean = np.matmul(self.pesos_aber[imes, :], CUSTO)

        return iser, lambda_mean, custo_mean

    def run_parallel(self):

        # Parâmetros iniciais
        nr_aber = self.aberturas.shape[1]
        nr_serfor = self.afluencias.shape[1]

        nr_cen_eol = self.eolicas.shape[1]
        nr_sist = self.afluencias.shape[0]
        submarket_codes = [self.sist.submercado[i].Codigo for i in self.submarket_index]
        nr_ute = len([x for x in self.sist.conf_ut if x.Sist in submarket_codes])
        mes_ini = self.sist.dger.MesInicioEstudo

        # Incializa matrizes da otimizacao
        Aeq = np.zeros(((1 + nr_cen_eol) * nr_sist, (2 + 3 * nr_cen_eol) * nr_sist + nr_ute * nr_cen_eol + 1))
        Beq = np.zeros(((1 + nr_cen_eol) * nr_sist, 1))
        A_GH = np.zeros((nr_cen_eol * nr_sist, (2 + 3 * nr_cen_eol) * nr_sist + nr_ute * nr_cen_eol + 1))
        B_GH = np.zeros((nr_cen_eol * nr_sist, 1))

        # Balanço Hídrico
        BH = np.zeros((nr_sist, (2 + 3 * nr_cen_eol) * nr_sist + nr_ute * nr_cen_eol + 1))
        for isist in range(nr_sist):
            BH[isist, isist:2 * nr_sist:nr_sist] = 1  # contante de ea e evert
            BH[isist, isist + 2 * nr_sist:(2 + nr_cen_eol) * nr_sist:nr_sist] = self.pesos_eol  # contante de gh

        # Atendimento Demanda
        AD = np.zeros((nr_sist, 3 * nr_cen_eol * nr_sist + nr_ute))
        for isist in range(nr_sist):
            AD[isist, isist:nr_sist * (3 * nr_cen_eol):nr_sist * nr_cen_eol] = [1, -1, 1]  # constante de gh, exc e def
            for iusi in range(nr_ute):
                AD[isist, (3 * nr_cen_eol) * nr_sist + iusi] = 1  # constante de gt

        # Insere BH e AD na Aeq e Insere GH na matriz A
        Aeq[:nr_sist, :] = BH
        lin_1 = nr_sist
        lin_2 = 0
        col_1 = 2 * nr_sist
        col_2 = 2 * nr_sist + 3 * nr_cen_eol * nr_sist
        for ieol in range(nr_cen_eol):
            Aeq[lin_1:lin_1 + nr_sist, col_1:col_1 + 3 * nr_cen_eol * nr_sist] = AD[:, :3 * nr_cen_eol * nr_sist]
            Aeq[lin_1:lin_1 + nr_sist, col_2:col_2 + nr_ute] = AD[:, 3 * nr_cen_eol * nr_sist:]
            A_GH[lin_2:lin_2 + nr_sist, col_1:col_1 + 2 * nr_cen_eol * nr_sist] = AD[:, :2 * nr_cen_eol * nr_sist]  # Verificar GH - EXC

            lin_1 += nr_sist
            lin_2 += nr_sist
            col_1 += nr_sist
            col_2 += nr_ute

        # Inicializa os volumes dos submercados
        EAIni = np.zeros((nr_sist, nr_serfor, self.nr_months + 1))
        for isist in range(nr_sist):
            EAIni[isist, :, 0] = self.sist.submercado[self.submarket_index[isist]].EAIni

        self.cf = list()
        for imes in range(self.nr_months):
            self.cf.append(fcf(imes))

        process = multiprocessing.Pool(processes=self.nr_process)

        # Processo iterativo
        self.iter = 0
        erro = 9e10
        tol = 0.1
        tol_IC = 0.1
        tol_perc = 0.1
        estab = 9e10
        while ((abs(erro) > tol_IC) or (abs(estab) > tol_perc)) and (
                abs(erro) > tol_perc):  # and self.iter < 10:   # && (abs(estab) > tol) %&& (abs(erro) > tol_IC)

            t_pdde = timer()

            # FORWARD
            CustoImed = np.zeros((self.nr_months, nr_serfor))
            CustoFut = np.zeros((self.nr_months, nr_serfor))
            for imes in range(self.nr_months):

                # Definindo ano e mes
                ano = int((mes_ini + imes) / 12)
                mes = (mes_ini + imes - 1) % 12

                # Determina a canalização das variáveis
                [c, lb, ub] = self.canalizacao(nr_ute, nr_sist, nr_cen_eol, ano, mes)

                if imes < self.nr_months - 1:
                    [A, B] = self.cf[imes].calcAB(Aeq)
                else:
                    A = None
                    B = None

                # Multiprocessamento aplicado na resoluçao dos PLs em cada série de ENA
                # t = timer()
                arguments_list = []
                for iser in range(nr_serfor):
                    arguments_list.append([iser, Aeq, Beq, A, B, A_GH, B_GH, c, lb, ub, EAIni, imes, ano, mes])
                # p = multiprocessing.Pool(processes=self.nr_process)
                result_list = process.starmap(self.forward_solve, arguments_list)
                # p.close()
                # tempo = (timer() - t)
                # print(f'Iteração {self.iter}: Tempo={round(tempo, 4)}s={round(tempo / nr_serfor, 5)}s/processo   ->   mês: {imes}    (FORWARD)')

                for value in result_list:
                    EAIni[:, value[0], imes+1] = value[1]
                    CustoImed[imes, value[0]] = value[2]
                    CustoFut[imes, value[0]] = value[3]

                # print('teste')

            CustoTot = CustoImed + CustoFut

            ZINF = np.mean(CustoTot[0, :])
            ZSUP = np.sum(np.mean(CustoImed, axis=1))
            self.ZINF.append(ZINF)
            self.ZSUP.append(ZSUP)

            self.CI_serie = np.sum(CustoImed, axis=0)

            sigma = (1 / nr_serfor) * np.sqrt(np.sum((self.CI_serie - np.mean(self.CI_serie)) ** 2))
            self.sigma.append(sigma)

            erro = self.ZSUP[self.iter] - self.ZINF[self.iter]
            tol_IC = 1.96 * self.sigma[self.iter]
            tol_perc = 0.005 * self.ZSUP[self.iter]

            if self.iter >= 2:
                media = np.mean(self.ZINF[-3:])
                estab = np.abs(media - self.ZINF[-1])

            # BACKWARD
            for imes in range(self.nr_months - 1, 0, -1):

                # Definindo iano e imes
                ano = int((mes_ini + imes) / 12)
                mes = (mes_ini + imes - 1) % 12

                # Determina a canalização das variáveis
                [c, lb, ub] = self.canalizacao(nr_ute, nr_sist, nr_cen_eol, ano, mes)

                if imes < self.nr_months - 1:
                    [A, B] = self.cf[imes].calcAB(Aeq)
                else:
                    A = None
                    B = None

                # Multiprocessamento aplicado na resoluçao dos PLs em cada série e abertura de ENA
                # t = timer()
                arguments_list = []
                for iser in range(nr_serfor):
                    arguments_list.append([iser, Aeq, Beq, A, B, A_GH, B_GH, c, lb, ub, EAIni, imes, ano, mes])
                # p = multiprocessing.Pool(processes=self.nr_process)
                result_list = process.starmap(self.backward_solve, arguments_list)
                # p.terminate()
                # tempo = (timer() - t)
                # print(f'Iteração {self.iter}: Tempo={round(tempo, 4)}s={round(tempo / (nr_serfor*nr_aber), 5)}s/processo   ->   mês: {imes}    (BACKWARD)')

                # Geração dos cortes da FCF
                for value in result_list:
                    ea_ini = EAIni[:nr_sist, value[0], imes]
                    lambda_mean = value[1]
                    custo_mean = value[2]

                    self.cf[imes - 1].geraFCF(ea_ini, custo_mean, lambda_mean)

            print('Iteração:', self.iter + 1,
                  ' - ZINF:', self.ZINF[self.iter],
                  ' - ZSUP:', self.ZSUP[self.iter],
                  ' - Erro:', erro,
                  ' - Tempo:', round(timer() - t_pdde, 2), 'seg')

            self.iter += 1

    def run(self):

        # Parâmetros iniciais
        nr_serfor = self.afluencias.shape[1]
        nr_aber = self.aberturas.shape[1]
        nr_cen_eol = self.eolicas.shape[1]
        nr_sist = self.afluencias.shape[0]
        submarket_codes = [self.sist.submercado[i].Codigo for i in self.submarket_index]
        nr_ute = len([x for x in self.sist.conf_ut if x.Sist in submarket_codes])
        mes_ini = self.sist.dger.MesInicioEstudo

        # Incializa matrizes da otimizacao
        Aeq = np.zeros(((1 + nr_cen_eol) * nr_sist, (2 + 3*nr_cen_eol)*nr_sist + nr_ute*nr_cen_eol + 1))
        Beq = np.zeros(((1 + nr_cen_eol) * nr_sist, 1))
        A_GH = np.zeros((nr_cen_eol * nr_sist, (2 + 3*nr_cen_eol)*nr_sist + nr_ute*nr_cen_eol + 1))
        B_GH = np.zeros((nr_cen_eol * nr_sist, 1))

        # Balanço Hídrico
        BH = np.zeros((nr_sist, (2 + 3*nr_cen_eol)*nr_sist + nr_ute*nr_cen_eol + 1))
        for isist in range(nr_sist):
            BH[isist, isist:2*nr_sist:nr_sist] = 1  # contante de ea e evert
            BH[isist, isist+2*nr_sist:(2+nr_cen_eol)*nr_sist:nr_sist] = self.pesos_eol  # contante de gh

        # Atendimento Demanda
        AD = np.zeros((nr_sist, 3*nr_cen_eol * nr_sist + nr_ute))
        for isist in range(nr_sist):
            AD[isist, isist:nr_sist*(3*nr_cen_eol):nr_sist*nr_cen_eol] = [1, -1, 1]   # constante de gh, exc e def
            for iusi in range(nr_ute):
                AD[isist, (3*nr_cen_eol)*nr_sist+iusi] = 1  # constante de gt

        # Insere BH e AD na Aeq e Insere GH na matriz A
        Aeq[:nr_sist, :] = BH
        lin_1 = nr_sist
        lin_2 = 0
        col_1 = 2*nr_sist
        col_2 = 2*nr_sist+3*nr_cen_eol*nr_sist
        for ieol in range(nr_cen_eol):
            Aeq[lin_1:lin_1+nr_sist, col_1:col_1+3*nr_cen_eol*nr_sist] = AD[:, :3*nr_cen_eol*nr_sist]
            Aeq[lin_1:lin_1+nr_sist, col_2:col_2+nr_ute] = AD[:, 3*nr_cen_eol*nr_sist:]
            A_GH[lin_2:lin_2+nr_sist, col_1:col_1+2*nr_cen_eol*nr_sist] = AD[:, :2*nr_cen_eol*nr_sist]   # Verificar GH - EXC

            lin_1 += nr_sist
            lin_2 += nr_sist
            col_1 += nr_sist
            col_2 += nr_ute

        # Inicializa os volumes das usinas
        EAIni = np.zeros((nr_sist, nr_serfor, self.nr_months + 1))
        for isist in range(nr_sist):
            EAIni[isist, :, 0] = self.sist.submercado[self.submarket_index[isist]].EAIni

        self.cf = list()
        for imes in range(self.nr_months):
            self.cf.append(fcf(imes))

        # Processo iterativo
        self.iter = 0
        erro = 9e10
        tol = 0.1
        tol_IC = 0.1
        tol_perc = 0.1
        estab = 9e10
        while ((abs(erro) > tol_IC) or (abs(estab) > tol_perc)) and (abs(erro) > tol_perc): # and self.iter < 10:   # && (abs(estab) > tol) %&& (abs(erro) > tol_IC)
            t_pdde = timer()

            # FORWARD
            ZINF = 0.
            ZSUP = 0.

            CustoImed = np.zeros((self.nr_months, nr_serfor))
            CustoFut = np.zeros((self.nr_months, nr_serfor))
            CustoTot = np.zeros((self.nr_months, nr_serfor))
            for imes in range(self.nr_months):

                # Definindo iano e imes
                ano = int((mes_ini + imes)/12)
                mes = (mes_ini + imes - 1) % 12

                # Determina a canalização das variáveis
                [c, lb, ub] = self.canalizacao(nr_ute, nr_sist, nr_cen_eol, ano, mes)

                if imes < self.nr_months-1:
                    [A, B] = self.cf[imes].calcAB(Aeq)
                else:
                    A = None
                    B = None

                for iser in range(nr_serfor):

                    # Preenche Beq e B
                    col_1 = nr_sist
                    col_2 = 0
                    for ieol in range(nr_cen_eol):
                        for isist in range(nr_sist):
                            submercado = self.sist.submercado[self.submarket_index[isist]]
                            EA_vetor = np.array([EAIni[isist, iser, imes] ** 2, EAIni[isist, iser, imes], 1])
                            if ieol == 0:
                                Beq[isist, 0] = EAIni[isist, iser, imes] + np.dot(submercado.ParamFC[:, mes], EA_vetor) * \
                                             submercado.FatorSeparacao * self.afluencias[isist, iser, imes] - np.dot(submercado.ParamEVMin[:, mes], EA_vetor) - \
                                             np.dot(submercado.ParamEVP[:, mes], EA_vetor) - submercado.EVM[ano, mes] - submercado.EDESVC[ano, mes]

                            Beq[col_1, 0] = submercado.Mercado[ano, mes] - submercado.NaoSimuladas[ano, mes] - submercado.GTMIN[ano, mes] - \
                                       (1 - submercado.FatorSeparacao)*self.afluencias[isist, iser, imes] - np.dot(submercado.ParamEVMin[:, mes], EA_vetor) \
                                            - self.eolicas[isist, ieol, imes] - submercado.EDESVF[ano, mes]
                            B_GH[col_2, 0] = np.dot(submercado.ParamGHMAX, EA_vetor) - (1 - submercado.FatorSeparacao) * \
                                       self.afluencias[isist, iser, imes] - np.dot(submercado.ParamEVMin[:, mes], EA_vetor)
                            col_1 += 1
                            col_2 += 1

                    if A is not None and B is not None:
                        A_full = np.concatenate((A_GH, A), axis=0)
                        B_full = np.concatenate((B_GH, B), axis=0)
                    else:
                        A_full = A_GH
                        B_full = B_GH

                    # tempo = timer()
                    [X, _, FVAL] = MosekProcess(Aeq, Beq, A_full, B_full, c, lb, ub).run()
                    # print(f'Iteração {self.iter} ->  Tempo: {round(timer() - tempo, 5)} seg (FORWARD)')
                    # print('mes: ', imes, '    ser: ', iser,)

                    # Atualiza armazenamento das hidrelétricas
                    for isist in range(nr_sist):
                        EAIni[isist, iser, imes+1] = X[isist]

                    CustoImed[imes, iser] = FVAL - const_vert * sum(X[nr_sist:2*nr_sist]) - const_ecx * sum(X[(2+nr_cen_eol)*nr_sist:(2+2*nr_cen_eol)*nr_sist]) - self.tx_desc_mensal*X[-1]
                    CustoFut[imes, iser] = self.tx_desc_mensal*X[-1]
                    CustoTot[imes, iser] = FVAL - const_vert * sum(X[nr_sist:2*nr_sist]) - const_ecx * sum(X[(2+nr_cen_eol)*nr_sist:(2+2*nr_cen_eol)*nr_sist])

                if imes == 0:
                    custototal_mean = np.mean(CustoTot[imes, :])
                    ZINF += custototal_mean

                custoimediato_mean = np.mean(CustoImed[imes, :])
                ZSUP += custoimediato_mean

            self.ZINF.append(ZINF)
            self.ZSUP.append(ZSUP)

            self.CI_serie = np.sum(CustoImed, axis=0)

            sigma = (1/nr_serfor) * np.sqrt(np.sum((self.CI_serie - np.mean(self.CI_serie))**2))
            self.sigma.append(sigma)

            erro = self.ZSUP[self.iter] - self.ZINF[self.iter]
            tol_IC = 1.96 * self.sigma[self.iter]
            tol_perc = 0.005 * self.ZSUP[self.iter]

            if self.iter >= 2:
                media = np.mean(self.ZINF[-3:])
                estab = np.abs(media - self.ZINF[-1])

            # BACKWARD
            for imes in range(self.nr_months-1, 0, -1):

                # Definindo iano e imes
                ano = int((mes_ini + imes) / 12)
                mes = (mes_ini + imes - 1) % 12

                # Determina a canalização das variáveis
                [c, lb, ub] = self.canalizacao(nr_ute, nr_sist, nr_cen_eol, ano, mes)
                
                if imes < self.nr_months-1:
                    [A, B] = self.cf[imes].calcAB(Aeq)
                else:
                    A = None
                    B = None

                for iser in range(nr_serfor):

                    LAMBDA = np.zeros((nr_aber, nr_sist))
                    CUSTO = np.zeros((nr_aber, 1))
                    for iaber in range(nr_aber):

                        # Preenche Beq e B
                        col_1 = nr_sist
                        col_2 = 0
                        for ieol in range(nr_cen_eol):

                            for isist in range(nr_sist):
                                submercado = self.sist.submercado[self.submarket_index[isist]]
                                EA_vetor = np.array([EAIni[isist, iser, imes] ** 2, EAIni[isist, iser, imes], 1])
                                if ieol == 0:
                                    Beq[isist, 0] = EAIni[isist, iser, imes] + np.dot(submercado.ParamFC[:, mes], EA_vetor) * \
                                                 submercado.FatorSeparacao * self.aberturas[isist, iaber, imes] - np.dot(submercado.ParamEVMin[:, mes], EA_vetor) - \
                                                 np.dot(submercado.ParamEVP[:, mes], EA_vetor) - submercado.EVM[ano, mes] - submercado.EDESVC[ano, mes]

                                Beq[col_1, 0] = submercado.Mercado[ano, mes] - submercado.NaoSimuladas[ano, mes] - submercado.GTMIN[ano, mes] - \
                                             (1 - submercado.FatorSeparacao) * self.aberturas[isist, iaber, imes] - np.dot(submercado.ParamEVMin[:, mes], EA_vetor) -\
                                                self.eolicas[isist, ieol, imes] - submercado.EDESVF[ano, mes]
                                B_GH[col_2, 0] = np.dot(submercado.ParamGHMAX, EA_vetor) - (1 - submercado.FatorSeparacao) * \
                                              self.aberturas[isist, iaber, imes] - np.dot(submercado.ParamEVMin[:, mes], EA_vetor)
                                col_1 += 1
                                col_2 += 1

                        if A is not None and B is not None:
                            A_full = np.concatenate((A_GH, A), axis=0)
                            B_full = np.concatenate((B_GH, B), axis=0)
                        else:
                            A_full = A_GH
                            B_full = B_GH

                        # tempo = timer()
                        [X, DUAL, FVAL] = MosekProcess(Aeq, Beq, A_full, B_full, c, lb, ub).run()
                        # print(f'Iteração {self.iter} ->  Tempo: {round(timer() - tempo, 5)} seg (BACWARD)')
                        # print('mes: ', imes, '    ser: ', iser, '     aber: ', iaber)

                        CUSTO[iaber, 0] = FVAL - const_vert * sum(X[nr_sist:2*nr_sist]) - const_ecx * sum(X[(2+nr_cen_eol)*nr_sist:(2+2*nr_cen_eol)*nr_sist])
                        for isist in range(nr_sist):
                            if DUAL[isist] > 0.:
                                LAMBDA[iaber, isist] = 0.
                            else:
                                LAMBDA[iaber, isist] = DUAL[isist]

                    lambda_mean = np.matmul(self.pesos_aber[imes, :], LAMBDA)
                    custo_mean = np.matmul(self.pesos_aber[imes, :], CUSTO)

                    ea_ini = np.zeros((nr_sist, 1))
                    for isist in range(nr_sist):
                        ea_ini[isist, 0] = EAIni[isist, iser, imes]

                    # Função que gera o corte da FCF
                    self.cf[imes-1].geraFCF(ea_ini, custo_mean, lambda_mean)

            print('Iteração:', self.iter + 1,
                  ' - ZINF:', self.ZINF[self.iter],
                  ' - ZSUP:', self.ZSUP[self.iter],
                  ' - Erro:', erro,
                  ' - Tempo:', round(timer() - t_pdde, 2), 'seg')

            self.iter += 1

        print('teste')

    def canalizacao(self, nr_ute, nr_sist, nr_cen_eol, ano, mes):

        submarket_codes = [self.sist.submercado[i].Codigo for i in self.submarket_index]

        c = np.zeros((2 + 3 * nr_cen_eol) * nr_sist + nr_ute * nr_cen_eol + 1)
        lb = np.zeros((2 + 3 * nr_cen_eol) * nr_sist + nr_ute * nr_cen_eol + 1)
        ub = np.zeros((2 + 3 * nr_cen_eol) * nr_sist + nr_ute * nr_cen_eol + 1)

        # Energia Armazenada
        for isist in range(nr_sist):
            c[isist] = 0.
            lb[isist] = 0. # self.sist.submercado[self.submarket_index[isist]].EAMIN[ano, mes]
            ub[isist] = self.sist.submercado[self.submarket_index[isist]].EAMAX[ano, mes]

        # Energia Vertida
        for isist in range(nr_sist):
            c[isist + nr_sist] = const_vert
            lb[isist + nr_sist] = 0.
            ub[isist + nr_sist] = None

        # Geração Hidráulica
        col = 2 * nr_sist
        for ieol in range(nr_cen_eol):
            for isist in range(nr_sist):
                c[col] = 0.
                lb[col] = 0.
                ub[col] = None
                col += 1

        # Excesso de Energia
        col = 2 * nr_sist + nr_sist * nr_cen_eol
        for ieol in range(nr_cen_eol):
            for isist in range(nr_sist):
                c[col] = const_ecx
                lb[col] = 0.
                ub[col] = None
                col += 1

        # Dados de Déficit
        col = 2 * nr_sist + 2 * (nr_sist * nr_cen_eol)
        for ieol in range(nr_cen_eol):
            for isist in range(nr_sist):
                c[col] = self.pesos_eol[0, ieol] * self.sist.submercado[self.submarket_index[isist]].CustoDeficit[0]
                lb[col] = 0.
                ub[col] = None  # infinito floa('inf')
                col += 1

        # Geração Térmica
        col = 2 * nr_sist + 3 * (nr_sist * nr_cen_eol)
        for ieol in range(nr_cen_eol):
            for iusi in self.sist.conf_ut:
                if iusi.Sist in submarket_codes:
                    c[col] = self.pesos_eol[0, ieol] * iusi.Custo[0]
                    lb[col] = 0.  # self.sist.conf_ut[isist].GTMIN[ano, mes]
                    ub[col] = iusi.GTMAX[ano, mes] - iusi.GTMin[ano, mes]
                    col += 1

        # Custo Futuro
        c[-1] = self.tx_desc_mensal
        lb[-1] = 0.
        ub[-1] = None  # infinito float('inf')  #

        return c, lb, ub

    def plot_convergencia(self, time: float, processos: int = 1):
        plt.figure()
        plt.plot(np.arange(1, self.iter+1), self.ZINF, marker='o', label='ZINF')
        plt.errorbar(np.arange(1, self.iter+1), self.ZSUP, 1.96*np.array(self.sigma), marker='o', label='ZSUP')
        plt.xlabel('Iteração')
        plt.ylabel('Custo [R$]')
        plt.title(f'Tempo de processamento: {time} min   Processos: {processos}')
        plt.legend()
        plt.show()


class pdde_SistIsol_FCI(object):
    sist = None
    cf = None
    ZINF = list()
    ZSUP = list()
    iter = 0
    sigma = list()
    CI_serie = None

    def __init__(self, sistema, nr_months, aberturas, pesos_aber, afluencias, submarket_index, cortes_FCI, nr_process: int):
        self.sist = sistema
        self.nr_months = nr_months
        self.aberturas = aberturas
        self.pesos_aber = pesos_aber
        self.afluencias = afluencias
        self.submarket_index = submarket_index
        self.nr_process = nr_process
        tx_mensal = (1 + self.sist.dger.TaxaDesconto) ** (1 / 12) - 1
        # self.tx_desc_mensal = 1 / (1 + tx_mensal)
        self.tx_desc_mensal = 1
        self.cortes_FCI = cortes_FCI

    def forward_solve(self, iser, Aeq, Beq, A, B, A_GH_FCI, B_GH_FCI, c, lb, ub, EAIni, imes, ano, mes) -> tuple:

        nr_sist = len(self.submarket_index)

        # Preenche Beq e B
        for isist in range(nr_sist):
            submercado = self.sist.submercado[self.submarket_index[isist]]
            EA_vetor = np.array([EAIni[isist, iser, imes] ** 2, EAIni[isist, iser, imes], 1])
            Beq[0, 0] = EAIni[isist, iser, imes] + np.dot(submercado.ParamFC[:, mes], EA_vetor) * \
                            submercado.FatorSeparacao * self.afluencias[isist, iser, imes] - np.dot(
                submercado.ParamEVMin[:, mes], EA_vetor) - \
                            np.dot(submercado.ParamEVP[:, mes], EA_vetor) - submercado.EVM[ano, mes] - \
                            submercado.EDESVC[ano, mes]
            B_GH_FCI[0, 0] = np.dot(submercado.ParamGHMAX, EA_vetor) - (1 - submercado.FatorSeparacao) * \
                             self.afluencias[isist, iser, imes] - np.dot(submercado.ParamEVMin[:, mes], EA_vetor)

        if A is not None and B is not None:
            A_full = np.concatenate((A_GH_FCI, A), axis=0)
            B_full = np.concatenate((B_GH_FCI, B), axis=0)
        else:
            A_full = A_GH_FCI
            B_full = B_GH_FCI

        # tempo = timer()
        [X, _, FVAL] = MosekProcess(Aeq, Beq, A_full, B_full, c, lb, ub).run()
        # print(f'Iteração {self.iter} ->  Tempo: {round(timer() - tempo, 3)} seg (FORWARD)')
        # # print('mes: ', imes, '    ser: ', iser,)

        # Atualiza armazenamento das hidrelétricas
        EAFin = np.array(X[:nr_sist])

        CustoImed = FVAL - const_vert * X[1] - const_ecx * X[3] - self.tx_desc_mensal * X[-1]
        CustoFut = self.tx_desc_mensal * X[-1]

        return iser, EAFin, CustoImed, CustoFut

    def backward_solve(self, iser, Aeq, Beq, A, B, A_GH, B_GH, c, lb, ub, EAIni, imes, ano, mes) -> tuple:

        nr_aber = self.aberturas.shape[1]
        nr_sist = len(self.submarket_index)

        LAMBDA = np.zeros((nr_aber, nr_sist))
        CUSTO = np.zeros((nr_aber, 1))
        for iaber in range(nr_aber):

            # Preenche Beq e B
            submercado = self.sist.submercado[self.submarket_index[0]]
            isist = 0
            EA_vetor = np.array([EAIni[isist, iser, imes] ** 2, EAIni[isist, iser, imes], 1])
            Beq[0, 0] = EAIni[isist, iser, imes] + np.dot(submercado.ParamFC[:, mes], EA_vetor) * \
                            submercado.FatorSeparacao * self.aberturas[isist, iaber, imes] - np.dot(
                submercado.ParamEVMin[:, mes], EA_vetor) - \
                            np.dot(submercado.ParamEVP[:, mes], EA_vetor) - submercado.EVM[ano, mes] - \
                            submercado.EDESVC[ano, mes]
            B_GH[0, 0] = np.dot(submercado.ParamGHMAX, EA_vetor) - (1 - submercado.FatorSeparacao) * \
                             self.aberturas[isist, iaber, imes] - np.dot(submercado.ParamEVMin[:, mes], EA_vetor)
            if A is not None and B is not None:
                A_full = np.concatenate((A_GH, A), axis=0)
                B_full = np.concatenate((B_GH, B), axis=0)
            else:
                A_full = A_GH
                B_full = B_GH

            # tempo = timer()
            [X, DUAL, FVAL] = MosekProcess(Aeq, Beq, A_full, B_full, c, lb, ub).run()
            # print(f'Iteração {self.iter} ->  Tempo: {round(timer() - tempo, 3)} seg (BACWARD)')
            # print('mes: ', imes, '    ser: ', iser, '     aber: ', iaber)

            CUSTO[iaber, 0] = FVAL - const_vert * X[1] - const_ecx * X[3]
            for isist in range(nr_sist):
                if DUAL[isist] > 0.:
                    LAMBDA[iaber, isist] = 0.
                else:
                    LAMBDA[iaber, isist] = DUAL[isist]

        lambda_mean = np.matmul(self.pesos_aber[imes, :], LAMBDA)
        custo_mean = np.matmul(self.pesos_aber[imes, :], CUSTO)

        return iser, lambda_mean, custo_mean

    def run_parallel(self):

        # Parâmetros iniciais
        nr_serfor = self.afluencias.shape[1]

        nr_sist = self.afluencias.shape[0]
        submarket_codes = [self.sist.submercado[i].Codigo for i in self.submarket_index]
        nr_ute = len([x for x in self.sist.conf_ut if x.Sist in submarket_codes])
        mes_ini = self.sist.dger.MesInicioEstudo

        # Incializa matrizes da otimizacao
        Aeq = np.zeros((nr_sist, 4 * nr_sist + 2))
        Beq = np.zeros((nr_sist, 1))
        A_GH_FCI = np.zeros((nr_sist + self.cortes_FCI[0].indepe.size, 4 * nr_sist + 2))
        B_GH_FCI = np.zeros((nr_sist + self.cortes_FCI[0].indepe.size, 1))

        # Balanço Hídrico
        BH = np.zeros((nr_sist, 4 * nr_sist + 2))
        for isist in range(nr_sist):
            BH[isist, isist:4:4*nr_sist] = 1  # contante de ea
            BH[isist, isist+1:4:4*nr_sist] = 1  # contante de evert
            BH[isist, isist+2:4:4*nr_sist] = 1  # contante de gh

        # Insere BH na Aeq e Insere GH na matriz A
        Aeq[:nr_sist, :] = BH
        A_GH_FCI[:nr_sist, 2:4:4*nr_sist] = 1  # Verificar GH - EXC
        A_GH_FCI[:nr_sist, 3:4:4*nr_sist] = -1  # Verificar GH - EXC

        # Inicializa os volumes dos submercados
        EAIni = np.zeros((nr_sist, nr_serfor, self.nr_months + 1))
        for isist in range(nr_sist):
            EAIni[isist, :, 0] = self.sist.submercado[self.submarket_index[isist]].EAIni

        self.cf = list()
        for imes in range(self.nr_months):
            self.cf.append(fcf(imes))

        process = multiprocessing.Pool(processes=self.nr_process)

        # Processo iterativo
        self.iter = 0
        erro = 9e10
        tol = 0.1
        tol_IC = 0.1
        tol_perc = 0.1
        estab = 9e10
        while ((abs(erro) > tol_IC) or (abs(estab) > tol_perc)) and (
                abs(erro) > tol_perc):  # and self.iter < 10:   # && (abs(estab) > tol) %&& (abs(erro) > tol_IC)

            t_pdde = timer()

            # FORWARD
            CustoImed = np.zeros((self.nr_months, nr_serfor))
            CustoFut = np.zeros((self.nr_months, nr_serfor))
            for imes in range(self.nr_months):

                # Definindo ano e mes
                ano = int((mes_ini + imes) / 12)
                mes = (mes_ini + imes - 1) % 12

                # Determina a canalização das variáveis
                [c, lb, ub] = self.canalizacao(nr_sist, ano, mes)

                if imes < self.nr_months - 1:
                    [A, B] = self.cf[imes].calcAB(Aeq)
                else:
                    A = None
                    B = None

                # Função de Custo Imediato
                for icor in range(self.cortes_FCI[imes].indepe.size):
                    A_GH_FCI[nr_sist + icor, 2] = self.cortes_FCI[imes].inc[icor]
                    B_GH_FCI[nr_sist + icor, 0] = -self.cortes_FCI[imes].indepe[icor]

                # Multiprocessamento aplicado na resoluçao dos PLs em cada série de ENA
                # t = timer()
                arguments_list = []
                for iser in range(nr_serfor):
                    arguments_list.append([iser, Aeq, Beq, A, B, A_GH_FCI, B_GH_FCI, c, lb, ub, EAIni, imes, ano, mes])
                # p = multiprocessing.Pool(processes=self.nr_process)
                result_list = process.starmap(self.forward_solve, arguments_list)
                # p.close()
                # tempo = (timer() - t)
                # print(f'Iteração {self.iter}: Tempo={round(tempo, 4)}s={round(tempo / nr_serfor, 5)}s/processo   ->   mês: {imes}    (FORWARD)')

                for value in result_list:
                    EAIni[:, value[0], imes + 1] = value[1]
                    CustoImed[imes, value[0]] = value[2]
                    CustoFut[imes, value[0]] = value[3]

                # print('teste')

            CustoTot = CustoImed + CustoFut

            ZINF = np.mean(CustoTot[0, :])
            ZSUP = np.sum(np.mean(CustoImed, axis=1))
            self.ZINF.append(ZINF)
            self.ZSUP.append(ZSUP)

            self.CI_serie = np.sum(CustoImed, axis=0)

            sigma = (1 / nr_serfor) * np.sqrt(np.sum((self.CI_serie - np.mean(self.CI_serie)) ** 2))
            self.sigma.append(sigma)

            erro = self.ZSUP[self.iter] - self.ZINF[self.iter]
            tol_IC = 1.96 * self.sigma[self.iter]
            tol_perc = 0.005 * self.ZSUP[self.iter]

            if self.iter >= 2:
                media = np.mean(self.ZINF[-3:])
                estab = np.abs(media - self.ZINF[-1])

            # BACKWARD
            for imes in range(self.nr_months - 1, 0, -1):

                # Definindo iano e imes
                ano = int((mes_ini + imes) / 12)
                mes = (mes_ini + imes - 1) % 12

                # Determina a canalização das variáveis
                [c, lb, ub] = self.canalizacao(nr_sist, ano, mes)

                if imes < self.nr_months - 1:
                    [A, B] = self.cf[imes].calcAB(Aeq)
                else:
                    A = None
                    B = None

                # Função de Custo Imediato
                for icor in range(self.cortes_FCI[imes].indepe.size):
                    A_GH_FCI[nr_sist + icor, 2] = self.cortes_FCI[imes].inc[icor]
                    B_GH_FCI[nr_sist + icor, 0] = -self.cortes_FCI[imes].indepe[icor]

                # Multiprocessamento aplicado na resoluçao dos PLs em cada série e abertura de ENA
                # t = timer()
                arguments_list = []
                for iser in range(nr_serfor):
                    arguments_list.append([iser, Aeq, Beq, A, B, A_GH_FCI, B_GH_FCI, c, lb, ub, EAIni, imes, ano, mes])
                # p = multiprocessing.Pool(processes=self.nr_process)
                result_list = process.starmap(self.backward_solve, arguments_list)
                # p.terminate()
                # tempo = (timer() - t)
                # print(f'Iteração {self.iter}: Tempo={round(tempo, 4)}s={round(tempo / (nr_serfor*nr_aber), 5)}s/processo   ->   mês: {imes}    (BACKWARD)')

                # Geração dos cortes da FCF
                for value in result_list:
                    ea_ini = EAIni[:nr_sist, value[0], imes]
                    lambda_mean = value[1]
                    custo_mean = value[2]

                    self.cf[imes - 1].geraFCF(ea_ini, custo_mean, lambda_mean)

            print('Iteração:', self.iter + 1,
                  ' - ZINF:', self.ZINF[self.iter],
                  ' - ZSUP:', self.ZSUP[self.iter],
                  ' - Erro:', erro,
                  ' - Tempo:', round(timer() - t_pdde, 2), 'seg')

            self.iter += 1

    def canalizacao(self, nr_sist, ano, mes):

        c = np.zeros(4 * nr_sist + 2)
        lb = np.zeros(4 * nr_sist + 2)
        ub = np.zeros(4 * nr_sist + 2)

        # Energia Armazenada
        for isist in range(nr_sist):
            c[isist] = 0.
            lb[isist] = 0.  # self.sist.submercado[self.submarket_index[isist]].EAMIN[ano, mes]
            ub[isist] = self.sist.submercado[self.submarket_index[isist]].EAMAX[ano, mes]

        # Energia Vertida
        for isist in range(nr_sist):
            c[isist + nr_sist] = const_vert
            lb[isist + nr_sist] = 0.
            ub[isist + nr_sist] = None

        # Geração Hidráulica
        col = 2 * nr_sist
        for isist in range(nr_sist):
            c[col] = 0.
            lb[col] = 0.
            ub[col] = None
            col += 1

        # Excesso de Energia
        col = 3 * nr_sist
        for isist in range(nr_sist):
            c[col] = const_ecx
            lb[col] = 0.
            ub[col] = None
            col += 1

        # Custo Imediato
        c[-2] = 1
        lb[-2] = 0.
        ub[-2] = None  # infinito float('inf')  #

        # Custo Futuro
        c[-1] = self.tx_desc_mensal
        lb[-1] = 0.
        ub[-1] = None  # infinito float('inf')  #

        return c, lb, ub

    def plot_convergencia(self, time: float):
        plt.figure()
        plt.plot(np.arange(1, self.iter + 1), self.ZINF, marker='o', label='ZINF')
        plt.errorbar(np.arange(1, self.iter + 1), self.ZSUP, 1.96 * np.array(self.sigma), marker='o', label='ZSUP')
        plt.xlabel('Iteração')
        plt.ylabel('Custo [R$]')
        plt.title(f'Tempo de processamento: {time} min')
        plt.legend()
        plt.show()


class pdde_SistMult(object):

    sist = None
    cf = None
    ZINF = list()
    ZSUP = list()
    iter = 0
    sigma = list()
    CI_serie = None

    def __init__(self, sistema, nr_months, aberturas, pesos_aber, afluencias, eolicas, pesos_eol, submarket_index,
                 nr_process: int):
        self.sist = sistema
        self.nr_months = nr_months
        self.aberturas = aberturas
        self.pesos_aber = pesos_aber
        self.afluencias = afluencias
        self.eolicas = eolicas
        self.pesos_eol = pesos_eol
        self.submarket_index = submarket_index
        self.nr_process = nr_process
        # tx_mensal = (1 + sistema.dger.TaxaDesconto) ** (1 / 12) - 1
        # self.tx_desc_mensal = 1 / (1 + tx_mensal)
        self.tx_desc_mensal = 1

    def forward_solve(self, iser, Aeq, Beq, A, B, A_GH, B_GH, c, lb, ub, EAIni, imes, ano, mes) -> tuple:

        nr_sist = len(self.submarket_index)
        nr_cen_eol = self.eolicas.shape[1]
        submarket_codes = [self.sist.submercado[i].Codigo for i in self.submarket_index]
        submercados = [x for x in self.sist.submercado if x.Codigo in submarket_codes]

        # Preenche Beq e B
        col_1 = nr_sist
        col_2 = 0
        for ieol in range(nr_cen_eol):
            for isist in range(nr_sist):
                submercado = submercados[isist]
                EA_vetor = np.array([EAIni[isist, iser, imes] ** 2, EAIni[isist, iser, imes], 1])
                if ieol == 0:
                    Beq[isist, 0] = EAIni[isist, iser, imes] + np.dot(submercado.ParamFC[:, mes], EA_vetor) * \
                                    submercado.FatorSeparacao * self.afluencias[isist, iser, imes] - np.dot(
                        submercado.ParamEVMin[:, mes], EA_vetor) - np.dot(submercado.ParamEVP[:, mes], EA_vetor) - submercado.EVM[
                                        ano, mes] - submercado.EDESVC[ano, mes]

                Beq[col_1, 0] = submercado.Mercado[ano, mes] - submercado.NaoSimuladas[ano, mes] - (1 - submercado.FatorSeparacao) * \
                                self.afluencias[isist, iser, imes] - np.dot(submercado.ParamEVMin[:, mes], EA_vetor) \
                                - self.eolicas[isist, ieol, imes] - submercado.EDESVF[ano, mes]  # - submercado.GTMIN[ano, mes]
                B_GH[col_2, 0] = np.dot(submercado.ParamGHMAX, EA_vetor) - (1 - submercado.FatorSeparacao) * \
                                 self.afluencias[isist, iser, imes] - np.dot(submercado.ParamEVMin[:, mes], EA_vetor)
                col_1 += 1
                col_2 += 1

            Beq[col_1, 0] = 0  # Nó Fictício
            col_1 += 1

        if A is not None and B is not None:
            A_full = np.concatenate((A_GH, A), axis=0)
            B_full = np.concatenate((B_GH, B), axis=0)
        else:
            A_full = A_GH
            B_full = B_GH

        # tempo = timer()
        [X, _, FVAL] = MosekProcess(Aeq, Beq, A_full, B_full, c, lb, ub).run()
        # print(f'Iteração {self.iter} ->  Tempo: {round(timer() - tempo, 3)} seg (FORWARD)')
        # # print('mes: ', imes, '    ser: ', iser,)

        # Atualiza armazenamento das hidrelétricas
        EAFin = np.array(X[:nr_sist])

        CustoImed = FVAL - const_vert * sum(X[nr_sist:2 * nr_sist]) - const_ecx * sum(
            X[(2 + nr_cen_eol) * nr_sist:(2 + 2 * nr_cen_eol) * nr_sist]) - self.tx_desc_mensal*X[-1]
        CustoFut = self.tx_desc_mensal*X[-1]

        return iser, EAFin, CustoImed, CustoFut

    def backward_solve(self, iser, Aeq, Beq, A, B, A_GH, B_GH, c, lb, ub, EAIni, imes, ano, mes) -> tuple:

        nr_aber = self.aberturas.shape[1]
        nr_sist = len(self.submarket_index)
        nr_cen_eol = self.eolicas.shape[1]
        submarket_codes = [self.sist.submercado[i].Codigo for i in self.submarket_index]
        submercados = [x for x in self.sist.submercado if x.Codigo in submarket_codes]

        LAMBDA = np.zeros((nr_aber, nr_sist))
        CUSTO = np.zeros((nr_aber, 1))
        for iaber in range(nr_aber):

            # Preenche Beq e B
            col_1 = nr_sist
            col_2 = 0
            for ieol in range(nr_cen_eol):

                for isist in range(nr_sist):
                    submercado = submercados[isist]
                    EA_vetor = np.array([EAIni[isist, iser, imes] ** 2, EAIni[isist, iser, imes], 1])
                    if ieol == 0:
                        Beq[isist, 0] = EAIni[isist, iser, imes] + np.dot(submercado.ParamFC[:, mes],
                                                                          EA_vetor) * \
                                        submercado.FatorSeparacao * self.aberturas[
                                            isist, iaber, imes] - np.dot(submercado.ParamEVMin[:, mes], EA_vetor) - \
                                        np.dot(submercado.ParamEVP[:, mes], EA_vetor) - submercado.EVM[
                                            ano, mes] - submercado.EDESVC[ano, mes]

                    Beq[col_1, 0] = submercado.Mercado[ano, mes] - submercado.NaoSimuladas[ano, mes] - \
                                    (1 - submercado.FatorSeparacao) * self.aberturas[
                                        isist, iaber, imes] - np.dot(submercado.ParamEVMin[:, mes], EA_vetor) - \
                                    self.eolicas[isist, ieol, imes] - submercado.EDESVF[ano, mes]  # - submercado.GTMIN[ano, mes]
                    B_GH[col_2, 0] = np.dot(submercado.ParamGHMAX, EA_vetor) - (
                            1 - submercado.FatorSeparacao) * \
                                     self.aberturas[isist, iaber, imes] - np.dot(submercado.ParamEVMin[:, mes],
                                                                                 EA_vetor)
                    col_1 += 1
                    col_2 += 1

                Beq[col_1, 0] = 0  # Nó Fictício
                col_1 += 1

            if A is not None and B is not None:
                A_full = np.concatenate((A_GH, A), axis=0)
                B_full = np.concatenate((B_GH, B), axis=0)
            else:
                A_full = A_GH
                B_full = B_GH

            # tempo = timer()
            [X, DUAL, FVAL] = MosekProcess(Aeq, Beq, A_full, B_full, c, lb, ub).run()
            # print(f'Iteração {self.iter} ->  Tempo: {round(timer() - tempo, 3)} seg (BACWARD)')
            # print('mes: ', imes, '    ser: ', iser, '     aber: ', iaber)

            CUSTO[iaber, 0] = FVAL - const_vert * sum(X[nr_sist:2 * nr_sist]) - const_ecx * sum(
                X[(2 + nr_cen_eol) * nr_sist:(2 + 2 * nr_cen_eol) * nr_sist])
            for isist in range(nr_sist):
                if DUAL[isist] > 0.:
                    LAMBDA[iaber, isist] = 0.
                else:
                    LAMBDA[iaber, isist] = DUAL[isist]

        lambda_mean = np.matmul(self.pesos_aber[imes, :], LAMBDA)
        custo_mean = np.matmul(self.pesos_aber[imes, :], CUSTO)

        return iser, lambda_mean, custo_mean

    def run_parallel(self):

        # Parâmetros iniciais
        nr_serfor = self.afluencias.shape[1]
        nr_aber = self.aberturas.shape[1]
        nr_cen_eol = self.eolicas.shape[1]
        submarket_codes = [self.sist.submercado[i].Codigo for i in self.submarket_index]
        submercados = [x for x in self.sist.submercado if x.Codigo in submarket_codes]
        nr_sist = len(submercados)
        termicas = [x for x in self.sist.conf_ut if x.Sist in submarket_codes]
        nr_ute = len(termicas)
        intercambios = [x for x in self.sist.intercambio if x.De in submarket_codes and x.Para in submarket_codes or
                        (x.De in submarket_codes and x.Para == 11) or (x.Para in submarket_codes and x.De == 11)]
        nr_intercambio = len(intercambios)
        mes_ini = self.sist.dger.MesInicioEstudo

        # Nó Fictício
        no_ficticio = 0
        if 11 in [x.De for x in intercambios]:
            no_ficticio = 1
            NF = np.zeros(nr_intercambio)
            for idx, interc in enumerate(intercambios):
                if interc.De == 11:
                    NF[idx] = -1
                elif interc.Para == 11:
                    NF[idx] = 1

        # Incializa matrizes da otimizacao
        Aeq = np.zeros(((1 + nr_cen_eol) * nr_sist + nr_cen_eol * no_ficticio,
                        (2 + 3 * nr_cen_eol) * nr_sist + nr_ute * nr_cen_eol + nr_intercambio * nr_cen_eol + 1))
        Beq = np.zeros(((1 + nr_cen_eol) * nr_sist + nr_cen_eol * no_ficticio, 1))
        A_GH = np.zeros((nr_cen_eol * nr_sist,
                         (2 + 3 * nr_cen_eol) * nr_sist + nr_ute * nr_cen_eol + nr_intercambio * nr_cen_eol + 1))
        B_GH = np.zeros((nr_cen_eol * nr_sist, 1))

        # Balanço Hídrico
        BH = np.zeros((nr_sist, (2 + 3 * nr_cen_eol) * nr_sist + nr_ute * nr_cen_eol + nr_intercambio * nr_cen_eol + 1))
        for isist in range(nr_sist):
            BH[isist, isist:2 * nr_sist:nr_sist] = 1  # contante de ea e evert
            BH[isist, isist + 2 * nr_sist:(2 + nr_cen_eol) * nr_sist:nr_sist] = self.pesos_eol  # contante de gh

        # Atendimento Demanda
        AD = np.zeros((nr_sist, 3 * nr_cen_eol * nr_sist + nr_ute + nr_intercambio))
        for isist in range(nr_sist):
            AD[isist, isist:nr_sist * (3 * nr_cen_eol):nr_sist * nr_cen_eol] = [1, -1, 1]  # constante de gh, exc e def
        for iusi in range(nr_ute):
            AD[submarket_codes.index(termicas[iusi].Sist), (3 * nr_cen_eol) * nr_sist + iusi] = 1  # constante de gt
        for idx, interc in enumerate(intercambios):
            if interc.Para == 11:
                AD[submarket_codes.index(interc.De), (
                            3 * nr_cen_eol) * nr_sist + nr_ute + idx] = -1  # intercambio saindo
            elif interc.De == 11:
                AD[submarket_codes.index(interc.Para), (
                            3 * nr_cen_eol) * nr_sist + nr_ute + idx] = 1  # intercambio chegando
            else:
                AD[submarket_codes.index(interc.De), (
                            3 * nr_cen_eol) * nr_sist + nr_ute + idx] = -1  # intercambio saindo
                AD[submarket_codes.index(interc.Para), (
                            3 * nr_cen_eol) * nr_sist + nr_ute + idx] = 1  # intercambio chegando

        # Insere BH, AD e NF na Aeq e Insere GH na matriz A
        Aeq[:nr_sist, :] = BH
        lin_1, lin_2 = nr_sist, 0
        col_1, col_2, col_3 = 2 * nr_sist, 2 * nr_sist + 3 * nr_cen_eol * nr_sist, 2 * nr_sist + 3 * nr_cen_eol * nr_sist + nr_cen_eol * nr_ute
        for ieol in range(nr_cen_eol):
            Aeq[lin_1:lin_1 + nr_sist, col_1:col_1 + 3 * nr_cen_eol * nr_sist] = AD[:, :3 * nr_cen_eol * nr_sist]  # Adiciona constantes de GH, EXC e DEF do AD
            Aeq[lin_1:lin_1 + nr_sist, col_2:col_2 + nr_ute] = AD[:, 3 * nr_cen_eol * nr_sist:3 * nr_cen_eol * nr_sist + nr_ute]  # Adiciona GT do AD
            Aeq[lin_1:lin_1 + nr_sist, col_3:col_3 + nr_intercambio] = AD[:, 3 * nr_cen_eol * nr_sist + nr_ute:]  # Adiciona INTERC do AD
            if no_ficticio:
                Aeq[lin_1 + nr_sist, col_3:col_3 + nr_intercambio] = NF  # Adiciona Nó Ficticio
            A_GH[lin_2:lin_2 + nr_sist, col_1:col_1 + 2 * nr_cen_eol * nr_sist] = AD[:, :2 * nr_cen_eol * nr_sist]  # Geração Hidráulica Máxima: GH - EXC
            # A_GH[lin_2:lin_2 + nr_sist, col_1:col_1 + nr_cen_eol * nr_sist] = AD[:, :nr_cen_eol * nr_sist]  # Geração Hidráulica Máxima: GH

            lin_1 += nr_sist + no_ficticio
            lin_2 += nr_sist
            col_1 += nr_sist
            col_2 += nr_ute
            col_3 += nr_intercambio

        # Inicializa os volumes das usinas
        EAIni = np.zeros((nr_sist, nr_serfor, self.nr_months + 1))
        for isist in range(nr_sist):
            EAIni[isist, :, 0] = submercados[isist].EAIni

        self.cf = list()
        for imes in range(self.nr_months):
            self.cf.append(fcf(imes))

        process = multiprocessing.Pool(processes=self.nr_process)

        # Processo iterativo
        self.iter = 1
        estab = 9e10
        self.ZINF = list()
        self.ZSUP = list()
        self.sigma = list()
        while True:

            t_pdde = timer()

            # FORWARD

            CustoImed = np.zeros((self.nr_months, nr_serfor))
            CustoFut = np.zeros((self.nr_months, nr_serfor))
            for imes in range(self.nr_months):

                # Definindo iano e imes
                ano = int((mes_ini + imes - 1) / 12)
                mes = (mes_ini + imes - 1) % 12

                # Determina a canalização das variáveis
                [c, lb, ub] = self.canalizacao(termicas, submercados, intercambios, nr_cen_eol, ano, mes)

                if imes < self.nr_months - 1:
                    [A, B] = self.cf[imes].calcAB(Aeq)
                else:
                    A = None
                    B = None

                # Multiprocessamento aplicado na resoluçao dos PLs em cada série de ENA
                t = timer()
                arguments_list = []
                for iser in range(nr_serfor):
                    arguments_list.append([iser, Aeq, Beq, A, B, A_GH, B_GH, c, lb, ub, EAIni, imes, ano, mes])
                # p = multiprocessing.Pool(processes=self.nr_process)
                result_list = process.starmap(self.forward_solve, arguments_list)
                # p.close()
                tempo = (timer() - t)
                print(
                    f'Iteração {self.iter}: Tempo={round(tempo, 4)}s={round(tempo / nr_serfor, 5)}s/processo   ->   mês: {imes}    (FORWARD)')

                for value in result_list:
                    EAIni[:, value[0], imes + 1] = value[1]
                    CustoImed[imes, value[0]] = value[2]
                    CustoFut[imes, value[0]] = value[3]

                # print('teste')

            CustoTot = CustoImed + CustoFut

            ZINF = np.mean(CustoTot[0, :])
            ZSUP = np.sum(np.mean(CustoImed, axis=1))
            self.ZINF.append(ZINF)
            self.ZSUP.append(ZSUP)

            self.CI_serie = np.sum(CustoImed, axis=0)

            sigma = (1 / nr_serfor) * np.sqrt(np.sum((self.CI_serie - np.mean(self.CI_serie)) ** 2))
            self.sigma.append(sigma)

            erro = self.ZSUP[self.iter - 1] - self.ZINF[self.iter - 1]
            tol_IC = 1.96 * self.sigma[self.iter-1]
            tol_perc = 0.005 * self.ZSUP[self.iter-1]

            print('Iteração:', self.iter,
                  ' - ZINF:', self.ZINF[self.iter-1],
                  ' - ZSUP:', self.ZSUP[self.iter-1],
                  ' - Erro:', erro,
                  ' - Tempo:', round(timer() - t_pdde, 2), 'seg')

            if self.iter >= 3:
                media = np.mean(self.ZINF[-3:])
                estab = np.abs(media - self.ZINF[-1])

            # Problem convergence verification
            if ((abs(erro) < tol_IC) and (abs(estab) < tol_perc)) or (abs(erro) < tol_perc):
                process.terminate()
                break

            # BACKWARD
            for imes in range(self.nr_months - 1, 0, -1):

                # Definindo iano e imes
                ano = int((mes_ini + imes - 1) / 12)
                mes = (mes_ini + imes - 1) % 12

                # Determina a canalização das variáveis
                [c, lb, ub] = self.canalizacao(termicas, submercados, intercambios, nr_cen_eol, ano, mes)

                if imes < self.nr_months - 1:
                    [A, B] = self.cf[imes].calcAB(Aeq)
                else:
                    A = None
                    B = None

                # Multiprocessamento aplicado na resoluçao dos PLs em cada série e abertura de ENA
                t = timer()
                arguments_list = []
                for iser in range(nr_serfor):
                    arguments_list.append([iser, Aeq, Beq, A, B, A_GH, B_GH, c, lb, ub, EAIni, imes, ano, mes])
                # p = multiprocessing.Pool(processes=self.nr_process)
                result_list = process.starmap(self.backward_solve, arguments_list)
                # p.terminate()
                tempo = (timer() - t)
                print(
                    f'Iteração {self.iter}: Tempo={round(tempo, 4)}s={round(tempo / (nr_serfor * nr_aber), 5)}s/processo   ->   mês: {imes}    (BACKWARD)')

                # Geração dos cortes da FCF
                for value in result_list:
                    ea_ini = EAIni[:nr_sist, value[0], imes]
                    lambda_mean = value[1]
                    custo_mean = value[2]

                    self.cf[imes - 1].geraFCF(ea_ini, custo_mean, lambda_mean)

            self.iter += 1

    def run(self):

        # Parâmetros iniciais
        nr_serfor = self.afluencias.shape[1]
        nr_aber = self.aberturas.shape[1]
        nr_cen_eol = self.eolicas.shape[1]
        submarket_codes = [self.sist.submercado[i].Codigo for i in self.submarket_index]
        submercados = [x for x in self.sist.submercado if x.Codigo in submarket_codes]
        nr_sist = len(submercados)
        termicas = [x for x in self.sist.conf_ut if x.Sist in submarket_codes]
        nr_ute = len(termicas)
        intercambios = [x for x in self.sist.intercambio if x.De in submarket_codes and x.Para in submarket_codes or
                        (x.De in submarket_codes and x.Para == 11) or (x.Para in submarket_codes and x.De == 11)]
        nr_intercambio = len(intercambios)
        mes_ini = self.sist.dger.MesInicioEstudo

        # Nó Fictício
        no_ficticio = 0
        if 11 in [x.De for x in intercambios]:
            no_ficticio = 1
            NF = np.zeros(nr_intercambio)
            for idx, interc in enumerate(intercambios):
                if interc.De == 11:
                    NF[idx] = -1
                elif interc.Para == 11:
                    NF[idx] = 1

        # Incializa matrizes da otimizacao
        Aeq = np.zeros(((1 + nr_cen_eol) * nr_sist + nr_cen_eol * no_ficticio, (2 + 3 * nr_cen_eol) * nr_sist + nr_ute * nr_cen_eol + nr_intercambio * nr_cen_eol + 1))
        Beq = np.zeros(((1 + nr_cen_eol) * nr_sist + nr_cen_eol * no_ficticio, 1))
        A_GH = np.zeros((nr_cen_eol * nr_sist, (2 + 3 * nr_cen_eol) * nr_sist + nr_ute * nr_cen_eol + nr_intercambio * nr_cen_eol + 1))
        B_GH = np.zeros((nr_cen_eol * nr_sist, 1))

        # Balanço Hídrico
        BH = np.zeros((nr_sist, (2 + 3 * nr_cen_eol) * nr_sist + nr_ute * nr_cen_eol + nr_intercambio * nr_cen_eol + 1))
        for isist in range(nr_sist):
            BH[isist, isist:2 * nr_sist:nr_sist] = 1  # contante de ea e evert
            BH[isist, isist + 2 * nr_sist:(2 + nr_cen_eol) * nr_sist:nr_sist] = self.pesos_eol  # contante de gh

        # Atendimento Demanda
        AD = np.zeros((nr_sist, 3 * nr_cen_eol * nr_sist + nr_ute + nr_intercambio))
        for isist in range(nr_sist):
            AD[isist, isist:nr_sist * (3 * nr_cen_eol):nr_sist * nr_cen_eol] = [1, -1, 1]  # constante de gh, exc e def
        for iusi in range(nr_ute):
            AD[submarket_codes.index(termicas[iusi].Sist), (3 * nr_cen_eol) * nr_sist + iusi] = 1  # constante de gt
        for idx, interc in enumerate(intercambios):
            if interc.Para == 11:
                AD[submarket_codes.index(interc.De), (3 * nr_cen_eol) * nr_sist + nr_ute + idx] = -1  # intercambio saindo
            elif interc.De == 11:
                AD[submarket_codes.index(interc.Para), (3 * nr_cen_eol) * nr_sist + nr_ute + idx] = 1  # intercambio chegando
            else:
                AD[submarket_codes.index(interc.De), (3 * nr_cen_eol) * nr_sist + nr_ute + idx] = -1  # intercambio saindo
                AD[submarket_codes.index(interc.Para), (3 * nr_cen_eol) * nr_sist + nr_ute + idx] = 1  # intercambio chegando

        # Insere BH, AD e NF na Aeq e Insere GH na matriz A
        Aeq[:nr_sist, :] = BH
        lin_1, lin_2 = nr_sist, 0
        col_1, col_2, col_3 = 2 * nr_sist, 2 * nr_sist + 3 * nr_cen_eol * nr_sist, 2 * nr_sist + 3 * nr_cen_eol * nr_sist + nr_cen_eol * nr_ute
        for ieol in range(nr_cen_eol):
            Aeq[lin_1:lin_1 + nr_sist, col_1:col_1 + 3 * nr_cen_eol * nr_sist] = AD[:, :3 * nr_cen_eol * nr_sist]  # Adiciona constantes de GH, EXC e DEF do AD
            Aeq[lin_1:lin_1 + nr_sist, col_2:col_2 + nr_ute] = AD[:, 3 * nr_cen_eol * nr_sist:3 * nr_cen_eol * nr_sist + nr_ute]  # Adiciona GT do AD
            Aeq[lin_1:lin_1 + nr_sist, col_3:col_3 + nr_intercambio] = AD[:, 3 * nr_cen_eol * nr_sist + nr_ute:]  # Adiciona INTERC do AD
            if no_ficticio:
                Aeq[lin_1 + nr_sist, col_3:col_3 + nr_intercambio] = NF  # Adiciona Nó Ficticio
            A_GH[lin_2:lin_2 + nr_sist, col_1:col_1 + 2*nr_cen_eol * nr_sist] = AD[:, :2*nr_cen_eol * nr_sist]  # Geração Hidráulica Máxima: GH - EXC
            # A_GH[lin_2:lin_2 + nr_sist, col_1:col_1 + nr_cen_eol * nr_sist] = AD[:, :nr_cen_eol * nr_sist]  # Geração Hidráulica Máxima: GH

            lin_1 += nr_sist + no_ficticio
            lin_2 += nr_sist
            col_1 += nr_sist
            col_2 += nr_ute
            col_3 += nr_intercambio

        # Inicializa os volumes das usinas
        EAIni = np.zeros((nr_sist, nr_serfor, self.nr_months + 1))
        for isist in range(nr_sist):
            EAIni[isist, :, 0] = submercados[isist].EAIni

        self.cf = list()
        for imes in range(self.nr_months):
            self.cf.append(fcf(imes))

        # Processo iterativo
        self.iter = 0
        erro = 9e10
        tol = 0.1
        tol_IC = 0.1
        tol_perc = 0.1
        estab = 9e10
        while ((abs(erro) > tol_IC) or (abs(estab) > tol_perc)) and (abs(erro) > tol_perc):  # and self.iter < 10:   # && (abs(estab) > tol) %&& (abs(erro) > tol_IC)
            t_pdde = timer()

            # FORWARD
            ZINF = 0.
            ZSUP = 0.

            CustoImed = np.zeros((self.nr_months, nr_serfor))
            CustoFut = np.zeros((self.nr_months, nr_serfor))
            CustoTot = np.zeros((self.nr_months, nr_serfor))
            for imes in range(self.nr_months):

                # Definindo iano e imes
                ano = int((mes_ini + imes - 1) / 12)
                mes = (mes_ini + imes - 1) % 12

                # Determina a canalização das variáveis
                [c, lb, ub] = self.canalizacao(termicas, submercados, intercambios, nr_cen_eol, ano, mes)

                if imes < self.nr_months - 1:
                    [A, B] = self.cf[imes].calcAB(Aeq)
                else:
                    A = None
                    B = None

                for iser in range(nr_serfor):

                    # Preenche Beq e B
                    col_1 = nr_sist
                    col_2 = 0
                    for ieol in range(nr_cen_eol):
                        for isist in range(nr_sist):
                            submercado = submercados[isist]
                            EA_vetor = np.array([EAIni[isist, iser, imes] ** 2, EAIni[isist, iser, imes], 1])
                            if ieol == 0:
                                Beq[isist, 0] = EAIni[isist, iser, imes] + np.dot(submercado.ParamFC[:, mes],
                                                                                  EA_vetor) * \
                                                submercado.FatorSeparacao * self.afluencias[isist, iser, imes] - np.dot(
                                    submercado.ParamEVMin[:, mes], EA_vetor) - \
                                                np.dot(submercado.ParamEVP[:, mes], EA_vetor) - submercado.EVM[
                                                    ano, mes] - submercado.EDESVC[ano, mes]

                            Beq[col_1, 0] = submercado.Mercado[ano, mes] - submercado.NaoSimuladas[ano, mes] - \
                                            submercado.GTMIN[ano, mes] - (1 - submercado.FatorSeparacao) * self.afluencias[
                                                isist, iser, imes] - np.dot(submercado.ParamEVMin[:, mes], EA_vetor) \
                                            - self.eolicas[isist, ieol, imes] - submercado.EDESVF[ano, mes]
                            B_GH[col_2, 0] = np.dot(submercado.ParamGHMAX, EA_vetor) - (1 - submercado.FatorSeparacao) * \
                                             self.afluencias[isist, iser, imes] - np.dot(submercado.ParamEVMin[:, mes],
                                                                                         EA_vetor)
                            col_1 += 1
                            col_2 += 1

                        Beq[col_1, 0] = 0  # Nó Fictício
                        col_1 += 1

                    if A is not None and B is not None:
                        A_full = np.concatenate((A_GH, A), axis=0)
                        B_full = np.concatenate((B_GH, B), axis=0)
                    else:
                        A_full = A_GH
                        B_full = B_GH

                    tempo = timer()
                    [X, _, FVAL] = MosekProcess(Aeq, Beq, A_full, B_full, c, lb, ub).run()
                    print(f'Iteração {self.iter} ->  Tempo: {round(timer() - tempo, 5)} seg (FORWARD)')
                    # print('mes: ', imes, '    ser: ', iser,)

                    # Atualiza armazenamento das hidrelétricas
                    for isist in range(nr_sist):
                        EAIni[isist, iser, imes + 1] = X[isist]

                    CustoImed[imes, iser] = FVAL - const_vert * sum(X[nr_sist:2 * nr_sist]) - const_ecx * sum(X[(2 + nr_cen_eol) * nr_sist:(2 + 2 * nr_cen_eol) * nr_sist]) - self.tx_desc_mensal*X[-1]
                    CustoFut[imes, iser] = self.tx_desc_mensal*X[-1]
                    CustoTot[imes, iser] = FVAL - const_vert * sum(X[nr_sist:2 * nr_sist]) - const_ecx * sum(X[(2 + nr_cen_eol) * nr_sist:(2 + 2 * nr_cen_eol) * nr_sist])

                if imes == 0:
                    custototal_mean = np.mean(CustoTot[imes, :])
                    ZINF += custototal_mean

                custoimediato_mean = np.mean(CustoImed[imes, :])
                ZSUP += custoimediato_mean

            self.ZINF.append(ZINF)
            self.ZSUP.append(ZSUP)

            self.CI_serie = np.sum(CustoImed, axis=0)

            sigma = (1 / nr_serfor) * np.sqrt(np.sum((self.CI_serie - np.mean(self.CI_serie)) ** 2))
            self.sigma.append(sigma)

            erro = self.ZSUP[self.iter] - self.ZINF[self.iter]
            tol_IC = 1.96 * self.sigma[self.iter]
            tol_perc = 0.005 * self.ZSUP[self.iter]

            if self.iter >= 2:
                media = np.mean(self.ZINF[-3:])
                estab = np.abs(media - self.ZINF[-1])

            # BACKWARD
            for imes in range(self.nr_months - 1, 0, -1):

                # Definindo iano e imes
                ano = int((mes_ini + imes - 1) / 12)
                mes = (mes_ini + imes - 1) % 12

                # Determina a canalização das variáveis
                [c, lb, ub] = self.canalizacao(termicas, submercados, intercambios, nr_cen_eol, ano, mes)

                if imes < self.nr_months - 1:
                    [A, B] = self.cf[imes].calcAB(Aeq)
                else:
                    A = None
                    B = None

                for iser in range(nr_serfor):

                    LAMBDA = np.zeros((nr_aber, nr_sist))
                    CUSTO = np.zeros((nr_aber, 1))
                    for iaber in range(nr_aber):

                        # Preenche Beq e B
                        col_1 = nr_sist
                        col_2 = 0
                        for ieol in range(nr_cen_eol):

                            for isist in range(nr_sist):
                                submercado = submercados[isist]
                                EA_vetor = np.array([EAIni[isist, iser, imes] ** 2, EAIni[isist, iser, imes], 1])
                                if ieol == 0:
                                    Beq[isist, 0] = EAIni[isist, iser, imes] + np.dot(submercado.ParamFC[:, mes],
                                                                                      EA_vetor) * \
                                                    submercado.FatorSeparacao * self.aberturas[
                                                        isist, iaber, imes] - np.dot(submercado.ParamEVMin[:, mes], EA_vetor) - \
                                                    np.dot(submercado.ParamEVP[:, mes], EA_vetor) - submercado.EVM[
                                                        ano, mes] - submercado.EDESVC[ano, mes]

                                Beq[col_1, 0] = submercado.Mercado[ano, mes] - submercado.NaoSimuladas[ano, mes] - \
                                                submercado.GTMIN[ano, mes] - \
                                                (1 - submercado.FatorSeparacao) * self.aberturas[
                                                    isist, iaber, imes] - np.dot(submercado.ParamEVMin[:, mes], EA_vetor) - \
                                                self.eolicas[isist, ieol, imes] - submercado.EDESVF[ano, mes]
                                B_GH[col_2, 0] = np.dot(submercado.ParamGHMAX, EA_vetor) - (
                                            1 - submercado.FatorSeparacao) * \
                                                 self.aberturas[isist, iaber, imes] - np.dot(submercado.ParamEVMin[:, mes],
                                                                                             EA_vetor)
                                col_1 += 1
                                col_2 += 1

                            Beq[col_1, 0] = 0  # Nó Fictício
                            col_1 += 1

                        if A is not None and B is not None:
                            A_full = np.concatenate((A_GH, A), axis=0)
                            B_full = np.concatenate((B_GH, B), axis=0)
                        else:
                            A_full = A_GH
                            B_full = B_GH

                        tempo = timer()
                        [X, DUAL, FVAL] = MosekProcess(Aeq, Beq, A_full, B_full, c, lb, ub).run()
                        print(f'Iteração {self.iter} ->  Tempo: {round(timer() - tempo, 5)} seg (BACWARD)')
                        # print('mes: ', imes, '    ser: ', iser, '     aber: ', iaber)

                        CUSTO[iaber, 0] = FVAL - const_vert * sum(X[nr_sist:2 * nr_sist]) - const_ecx * sum(
                            X[(2 + nr_cen_eol) * nr_sist:(2 + 2 * nr_cen_eol) * nr_sist])
                        for isist in range(nr_sist):
                            if DUAL[isist] > 0.:
                                LAMBDA[iaber, isist] = 0.
                            else:
                                LAMBDA[iaber, isist] = DUAL[isist]

                    lambda_mean = np.matmul(self.pesos_aber[imes, :], LAMBDA)
                    custo_mean = np.matmul(self.pesos_aber[imes, :], CUSTO)

                    ea_ini = np.zeros((nr_sist, 1))
                    for isist in range(nr_sist):
                        ea_ini[isist, 0] = EAIni[isist, iser, imes]

                    # Função que gera o corte da FCF
                    self.cf[imes - 1].geraFCF(ea_ini, custo_mean, lambda_mean)

            print('Iteração:', self.iter + 1,
                  ' - ZINF:', self.ZINF[self.iter],
                  ' - ZSUP:', self.ZSUP[self.iter],
                  ' - Erro:', erro,
                  ' - Tempo:', round(timer() - t_pdde, 2), 'seg')

            self.iter += 1

    def canalizacao(self, termicas, submercados, intercambios, nr_cen_eol, ano, mes):

        nr_sist = len(submercados)
        nr_ute = len(termicas)
        nr_interc = len(intercambios)

        c = np.zeros((2 + 3 * nr_cen_eol) * nr_sist + nr_ute * nr_cen_eol + nr_interc * nr_cen_eol + 1)
        lb = np.zeros((2 + 3 * nr_cen_eol) * nr_sist + nr_ute * nr_cen_eol + nr_interc * nr_cen_eol + 1)
        ub = np.zeros((2 + 3 * nr_cen_eol) * nr_sist + nr_ute * nr_cen_eol + nr_interc * nr_cen_eol + 1)

        # Energia Armazenada
        for isist in range(nr_sist):
            c[isist] = 0.
            lb[isist] = 0.  # submercados[isist].EAMIN[ano, mes]
            ub[isist] = submercados[isist].EAMAX[ano, mes]

        # Energia Vertida
        for isist in range(nr_sist):
            c[isist + nr_sist] = const_vert
            lb[isist + nr_sist] = 0.
            ub[isist + nr_sist] = None

        # Geração Hidráulica
        col = 2 * nr_sist
        for ieol in range(nr_cen_eol):
            for isist in range(nr_sist):
                c[col] = 0.
                lb[col] = 0.
                ub[col] = None
                col += 1

        # Excesso de Energia
        col = 2 * nr_sist + nr_sist * nr_cen_eol
        for ieol in range(nr_cen_eol):
            for isist in range(nr_sist):
                c[col] = const_ecx
                lb[col] = 0.
                ub[col] = 0.
                col += 1

        # Dados de Déficit
        col = 2 * nr_sist + 2 * (nr_sist * nr_cen_eol)
        for ieol in range(nr_cen_eol):
            for isist in range(nr_sist):
                c[col] = self.pesos_eol[0, ieol] * submercados[isist].CustoDeficit[0]
                lb[col] = 0.
                ub[col] = None  # infinito floa('inf')
                col += 1

        # Geração Térmica
        col = 2 * nr_sist + 3 * (nr_sist * nr_cen_eol)
        for ieol in range(nr_cen_eol):
            for iusi in termicas:
                c[col] = self.pesos_eol[0, ieol] * iusi.Custo[0]
                lb[col] = iusi.GTMin[ano, mes]  # 0
                ub[col] = iusi.GTMAX[ano, mes]  # - iusi.GTMin[ano, mes]
                col += 1

        # Intercâmbio
        col = 2 * nr_sist + 3 * (nr_sist * nr_cen_eol) + nr_cen_eol * nr_ute
        for ieol in range(nr_cen_eol):
            for interc in intercambios:
                c[col] = 0.
                lb[col] = interc.LimiteMinimo[ano, mes]
                ub[col] = interc.LimiteMaximo[ano, mes]
                col += 1

        # Custo Futuro
        c[-1] = self.tx_desc_mensal
        lb[-1] = 0.
        ub[-1] = None  # infinito float('inf')  #

        return c, lb, ub

    def plot_convergencia(self, time: float, processos: int = 1):
        plt.figure()
        plt.plot(np.arange(1, self.iter + 1), self.ZINF, marker='o', label='ZINF')
        plt.errorbar(np.arange(1, self.iter + 1), self.ZSUP, 1.96 * np.array(self.sigma), marker='o', label='ZSUP')
        plt.xlabel('Iteração')
        plt.ylabel('Custo [R$]')
        plt.title(f'Tempo de processamento: {time} min   Processos: {processos}')
        plt.legend()
        plt.show()