import numpy as np
from timeit import default_timer as timer
import multiprocessing

from Modules.EnergyPlanning.otimo import otimo
from Modules.EnergyPlanning.MosekOptimization import MosekProcess

const_vert = 0.00001
const_ecx = 0.00001


class Sim_Oper(object):

    sist = None
    resultados = None

    def __init__(self, sistema, nr_meses, afluencias, eolicas, fcf, submarket_index, nr_process: int):
        self.sist = sistema
        self.nr_meses = nr_meses
        self.afluencias = afluencias
        self.eolicas = eolicas
        self.submarket_index = submarket_index
        self.nr_process = nr_process
        self.fcf = fcf
        # tx_mensal = (1 + sistema.dger.TaxaDesconto) ** (1 / 12) - 1
        # self.tx_desc_mensal = 1 / (1 + tx_mensal)
        self.tx_desc_mensal = 1

    def forward_solve(self, iser, Aeq, Beq, A, B, A_GH, B_GH, c, lb, ub, EAIni, imes, ano, mes, ieol) -> tuple:

        nr_sist = len(self.submarket_index)
        submarket_codes = [self.sist.submercado[i].Codigo for i in self.submarket_index]
        submercados = [x for x in self.sist.submercado if x.Codigo in submarket_codes]

        dem_liq = np.zeros(len(submercados))
        outros_usos = np.zeros(len(submercados))

        # Preenche Beq e B
        col_1 = nr_sist
        col_2 = 0
        for isist in range(nr_sist):
            submercado = submercados[isist]
            EA_vetor = np.array([EAIni[isist, iser, imes] ** 2, EAIni[isist, iser, imes], 1])
            Beq[isist, 0] = EAIni[isist, iser, imes] + np.dot(submercado.ParamFC[:, mes],EA_vetor) * \
                            submercado.FatorSeparacao * self.afluencias[isist, iser, imes] - np.dot(
                            submercado.ParamEVMin[:, mes], EA_vetor) - \
                            np.dot(submercado.ParamEVP[:, mes], EA_vetor) - submercado.EVM[ano, mes] - submercado.EDESVC[ano, mes]
            outros_usos[isist] = np.dot(submercado.ParamEVMin[:, mes], EA_vetor) + submercado.EDESVF[ano, mes]
            dem_liq[isist] = submercado.Mercado[ano, mes] - submercado.NaoSimuladas[ano, mes] - outros_usos[isist]
            Beq[col_1, 0] = dem_liq[isist] - self.eolicas[isist, ieol, imes] - (1 - submercado.FatorSeparacao) *\
                             self.afluencias[isist, iser, imes]  #  - submercado.GTMIN[ano, mes]
            B_GH[col_2, 0] = np.dot(submercado.ParamGHMAX, EA_vetor) - (1 - submercado.FatorSeparacao) * \
                             self.afluencias[isist, iser, imes] - np.dot(submercado.ParamEVMin[:, mes],
                                                                         EA_vetor)
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
        # print(f'Iteração {self.iter} ->  Tempo: {round(timer() - tempo, 3)} seg (FORWARD)')
        # # print('mes: ', imes, '    ser: ', iser,)

        # Atualiza armazenamento das hidrelétricas
        EAFin = np.array(X[:nr_sist])

        CustoImed = FVAL - const_vert * sum(X[nr_sist:2 * nr_sist]) - const_ecx * sum(X[2 * nr_sist:4 * nr_sist]) - self.tx_desc_mensal*X[-1]
        CustoFut = self.tx_desc_mensal*X[-1]

        return iser, X, DUAL, EAFin, CustoImed, CustoFut, dem_liq

    def run(self):

        # Parâmetros iniciais
        nr_serfor = self.afluencias.shape[1]
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
        nr_months = self.afluencias.shape[2]

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

        self.resultados = list()
        for ieol in range(nr_cen_eol):
            self.resultados.append(otimo(nr_months, nr_serfor))

        # Incializa matrizes da otimizacao
        Aeq = np.zeros((2 * nr_sist + no_ficticio, 5 * nr_sist + nr_ute + nr_intercambio + 1))
        Beq = np.zeros((2 * nr_sist + no_ficticio, 1))
        A_GH = np.zeros((nr_sist, 5 * nr_sist + nr_ute + nr_intercambio + 1))
        B_GH = np.zeros((nr_sist, 1))

        # Balanço Hídrico
        BH = np.zeros((nr_sist, 5 * nr_sist + nr_ute + nr_intercambio + 1))
        for isist in range(nr_sist):
            BH[isist, isist:2 * nr_sist:nr_sist] = 1  # contante de ea e evert
            BH[isist, isist + 2 * nr_sist:3 * nr_sist:nr_sist] = 1  # contante de gh

        # Atendimento Demanda
        AD = np.zeros((nr_sist, 3 * nr_sist + nr_ute + nr_intercambio))
        for isist in range(nr_sist):
            AD[isist, isist:nr_sist * 3:nr_sist] = [1, -1, 1]  # constante de gh, exc e def
        for iusi in range(nr_ute):
            AD[submarket_codes.index(termicas[iusi].Sist), 3 * nr_sist + iusi] = 1  # constante de gt
        for idx, interc in enumerate(intercambios):
            if interc.Para == 11:
                AD[submarket_codes.index(interc.De), 3 * nr_sist + nr_ute + idx] = -1  # intercambio saindo
            elif interc.De == 11:
                AD[submarket_codes.index(interc.Para), 3 * nr_sist + nr_ute + idx] = 1  # intercambio chegando
            else:
                AD[submarket_codes.index(interc.De), 3 * nr_sist + nr_ute + idx] = -1  # intercambio saindo
                AD[submarket_codes.index(interc.Para), 3 * nr_sist + nr_ute + idx] = 1  # intercambio chegando

        # Insere BH, AD e NF na Aeq e Insere GH na matriz A
        Aeq[:nr_sist, :] = BH
        lin_1, lin_2 = nr_sist, 0
        col_1, col_2, col_3 = 2 * nr_sist, 2 * nr_sist + 3 * nr_sist, 2 * nr_sist + 3 * nr_sist + nr_ute
        Aeq[lin_1:lin_1 + nr_sist, col_1:col_1 + 3 * nr_sist] = AD[:, :3 * nr_sist]  # Adiciona constantes de GH, EXC e DEF do AD
        Aeq[lin_1:lin_1 + nr_sist, col_2:col_2 + nr_ute] = AD[:, 3 * nr_sist:3 * nr_sist + nr_ute]  # Adiciona GT do AD
        Aeq[lin_1:lin_1 + nr_sist, col_3:col_3 + nr_intercambio] = AD[:, 3 * nr_sist + nr_ute:]  # Adiciona INTERC do AD
        if no_ficticio:
            Aeq[lin_1 + nr_sist, col_3:col_3 + nr_intercambio] = NF  # Adiciona Nó Ficticio
        A_GH[lin_2:lin_2 + nr_sist, col_1:col_1 + 2 * nr_sist] = AD[:, :2 * nr_sist]  # Geração Hidráulica Máxima: GH - EXC
        # A_GH[lin_2:lin_2 + nr_sist, col_1:col_1 + nr_sist] = AD[:, :nr_sist]  # Geração Hidráulica Máxima: GH

        # Inicializa os volumes das usinas
        EAIni = np.zeros((nr_sist, nr_serfor, self.nr_meses + 1))
        for isist in range(nr_sist):
            EAIni[isist, :, 0] = submercados[isist].EAIni

        process = multiprocessing.Pool(processes=self.nr_process)

        # Despacho ótimo do sistema
        for ieol in range(nr_cen_eol):

            VAR = np.zeros((nr_serfor, 5 * nr_sist + nr_ute + nr_intercambio + 1, self.nr_meses))
            CUS = np.zeros((nr_serfor, self.nr_meses))
            LAM = np.zeros((nr_serfor, nr_sist, self.nr_meses))
            AL = np.zeros((nr_serfor, self.nr_meses))
            CM = np.zeros((nr_serfor, nr_sist, self.nr_meses))
            DEMLIQ = np.zeros((nr_serfor, nr_sist, self.nr_meses))
            for imes in range(self.nr_meses):

                # Definindo iano e imes
                ano = int((mes_ini + imes - 1) / 12)
                mes = (mes_ini + imes - 1) % 12

                # Determina a canalização das variáveis
                [c, lb, ub] = self.canalizacao(termicas, submercados, intercambios, ano, mes)

                if imes < self.nr_meses - 1:

                    FCF = self.fcf[imes]

                    ncol = Aeq.shape[1]
                    A = np.zeros((FCF.nr_cortes, ncol))
                    B = np.zeros((FCF.nr_cortes, 1))
                    for icor in range(FCF.nr_cortes):
                        for isist in range(nr_sist):
                            A[icor, isist] = FCF.coef_ea[icor][isist]
                            A[icor, -1] = -1
                        B[icor, 0] = -FCF.termo_i[icor]

                    AB = np.concatenate((A, B), axis=1)
                    AB = np.around(AB, decimals=2)
                    AB = np.unique(AB, axis=0)
                    remove = list()
                    for row in range(AB.shape[0]):
                        if len([x for x in AB[row, :] if x == 0.]) >= ncol:
                            remove.append(row)
                    if remove:
                        AB = np.delete(AB, remove, axis=0)

                    A = AB[:, :-1]
                    B_aux = AB[:, -1]

                    B = np.zeros((B_aux.size, 1))
                    B[:, 0] = B_aux

                else:

                    A = None
                    B = None

                # Multiprocessamento aplicado na resoluçao dos PLs em cada série de ENA
                t = timer()
                arguments_list = []
                for iser in range(nr_serfor):
                    arguments_list.append([iser, Aeq, Beq, A, B, A_GH, B_GH, c, lb, ub, EAIni, imes, ano, mes, ieol])
                # p = multiprocessing.Pool(processes=self.nr_process)
                result_list = process.starmap(self.forward_solve, arguments_list)
                # p.close()
                tempo = (timer() - t)
                print('Mês %i:  Tempo = %4.2f s = %4.2f s/processo' % (imes, round(tempo, 4), round(tempo / nr_serfor, 5)))

                for value in result_list:
                    EAIni[:, value[0], imes + 1] = value[3]

                    VAR[value[0], :, imes] = value[1]
                    CUS[value[0], imes] = value[4]
                    LAM[value[0], :, imes] = np.minimum(value[2][:nr_sist], 0)
                    AL[value[0], imes] = value[5]
                    CM[value[0], :, imes] = np.maximum(value[2][nr_sist:2*nr_sist], 0)
                    DEMLIQ[value[0], :, imes] = value[6]

            self.resultados[ieol].save(self.sist, VAR, CUS, LAM, CM, AL, DEMLIQ, self.submarket_index, self.afluencias)

        process.terminate()

    def canalizacao(self, termicas, submercados, intercambios, ano, mes):

        nr_sist = len(submercados)
        nr_ute = len(termicas)
        nr_interc = len(intercambios)

        c = np.zeros(5 * nr_sist + nr_ute + nr_interc + 1)
        lb = np.zeros(5 * nr_sist + nr_ute + nr_interc + 1)
        ub = np.zeros(5 * nr_sist + nr_ute + nr_interc + 1)

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
            ub[col] = 0.
            col += 1

        # Dados de Déficit
        col = 4 * nr_sist
        for isist in range(nr_sist):
            c[col] = submercados[isist].CustoDeficit[0]
            lb[col] = 0.
            ub[col] = None  # infinito floa('inf')
            col += 1

        # Geração Térmica
        col = 5 * nr_sist
        for iusi in termicas:
            c[col] = iusi.Custo[0]
            lb[col] = iusi.GTMin[ano, mes]
            ub[col] = iusi.GTMAX[ano, mes]  # - iusi.GTMin[ano, mes]
            col += 1

        # Intercâmbio
        col = 5 * nr_sist + nr_ute
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
