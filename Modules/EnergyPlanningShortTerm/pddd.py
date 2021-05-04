import numpy as np
from timeit import default_timer as timer
from matplotlib import pyplot as plt

from Modules.EnergyPlanningShortTerm.fcf import fcf
from Modules.EnergyPlanning.MosekOptimization import MosekProcess
from Modules.EnergyPlanningShortTerm.otimo import otimo

# Vetor de variáveis:
# X = [v_1 q_1 s_1 dv_1 qb_1 qe_1 gh_1 fqe_1 ... v_n q_n s_n dv_n qb_n qe_n gh_n fqe_n  # hidrelétricas
# gt_1... gt_m                                                               # termelétricas
# ci_1 ... ci_p ce_1 ... ce_q f12 f21 .... fij fji                           # contratos e intercambio
# def_1 ... def_p alfa]                                                      # deficit e custo futuro
# v: volume final / q: vazão turbinada / s: vazão vertida / dv: vazão desviada (retirada) da UHE /
# qb: vazão bombeada da UHE / qe: vazão evaporada / gh: geração hidráulica / fqe: variável de folga da vazão evaporada
# gt: geração termelétrica / u: variável decisão on/off (0 ou 1) /
# ya: variável auxiliar acionamento (0 ou 1) / ya: variável auxiliar desligamento (0 ou 1)
# ci: contrato importacao / ce: contrato exportacao / def: déficit  / alfa: custo futuro

fator = 2.63
nr_var_hidr = 8  # v, q, s, dv, qb, qe, gh, fgh
nr_var_term = 1  # gt
penal_vert = 0.0001
penal_folga_evap = 0.
penal_folga_gh = 0.


class pddd(object):

    dados = None
    cf = None
    ZINF = list()
    ZSUP = list()
    iter = 0
    sigma = list()
    CI_serie = None
    resultados = None

    def __init__(self, sistema, wind_energy, last_stage_fcf):
        self.dados = sistema
        self.wind_energy = wind_energy
        self.last_stage_fcf = last_stage_fcf

    def last_stage_fcf_management(self, ncol, nr_var_hidr):

        nr_hidr = self.last_stage_fcf.shape[1] - 1
        nr_cortes = self.last_stage_fcf.shape[0]
        A = np.zeros((nr_cortes, ncol))
        B = np.zeros((nr_cortes, 1))
        for icor in range(nr_cortes):
            for i in range(nr_hidr):
                A[icor, nr_var_hidr*i] = self.last_stage_fcf.loc[icor, i]
                A[icor, -1] = -1
            B[icor, 0] = -self.last_stage_fcf.loc[icor, 'indep']

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

        return A, B

    def fph_constraints_generate(self, nr_var_hidr: int, nr_var_all: int, stage: int) -> tuple:

        A_GH = np.empty((0, nr_var_all))
        B_GH = np.empty((0, 1))
        for i, uhe in enumerate(self.dados.hidr):

            if uhe.FunProdHidr and uhe.EngolMaxT is not None:  # Vol/Turb/Vert

                if uhe.EngolMaxT.loc[stage, 'ENGOL'] != 0:

                    for idx in uhe.FunProdHidr['FPH'].index:

                        new_array_A_GH = np.zeros((1, nr_var_all))

                        if 'Vol' in uhe.FunProdHidr['Tipo']:
                            new_array_A_GH[0, nr_var_hidr * i] = - uhe.FunProdHidr['FPH'].loc[idx, 'vol']  # v
                        if 'Turb' in uhe.FunProdHidr['Tipo']:
                            new_array_A_GH[0, nr_var_hidr * i + 1] = - uhe.FunProdHidr['FPH'].loc[idx, 'turb']  # q
                        if 'Vert' in uhe.FunProdHidr['Tipo']:
                            new_array_A_GH[0, nr_var_hidr * i + 2] = - uhe.FunProdHidr['FPH'].loc[idx, 'vert']  # s
                        new_array_A_GH[0, nr_var_hidr * i + 6] = 1  # gh
                        new_array_A_GH[0, nr_var_hidr * i + 7] = 1  # fgh

                        A_GH = np.append(A_GH, new_array_A_GH, axis=0)
                        indep = np.zeros((1, 1))
                        indep[0, 0] = max(0, uhe.FunProdHidr['FPH'].loc[idx, 'indep'])
                        B_GH = np.append(B_GH, indep, axis=0)

        # nr_cortes_fph = sum([len(x.FunProdHidr['FPH'].index) if x.FunProdHidr else 0 for x in self.dados.hidr])
        #
        # A_GH = np.zeros((nr_cortes_fph, nr_var_all))
        # B_GH = np.zeros((nr_cortes_fph, 1))
        # pos = 0
        # for i, uhe in enumerate(self.dados.hidr):
        #     if uhe.FunProdHidr:   # Vol/Turb/Vert
        #         for idx in uhe.FunProdHidr['FPH'].index:
        #             if 'Vol' in uhe.FunProdHidr['Tipo']:
        #                 A_GH[pos, nr_var_hidr * i] = - uhe.FunProdHidr['FPH'].loc[idx, 'vol']  # v
        #             if 'Turb' in uhe.FunProdHidr['Tipo']:
        #                 A_GH[pos, nr_var_hidr * i + 1] = - uhe.FunProdHidr['FPH'].loc[idx, 'turb']  # q
        #             if 'Vert' in uhe.FunProdHidr['Tipo']:
        #                 A_GH[pos, nr_var_hidr * i + 2] = - uhe.FunProdHidr['FPH'].loc[idx, 'vert']  # s
        #             A_GH[pos, nr_var_hidr * i + 6] = 1  # gh
        #             A_GH[pos, nr_var_hidr * i + 7] = 1  # fgh
        #             B_GH[pos, 0] = uhe.FunProdHidr['FPH'].loc[idx, 'indep']
        #
        #             pos += 1

        return A_GH, B_GH

    def run(self):

        # region Parâmetros iniciais
        nr_submercados = len([x for x in self.dados.sist if x.Sigla != 'FC'])
        nr_unid_term = sum([x.NumUnidades for x in self.dados.term])
        nr_hidr = len(self.dados.hidr)
        nr_interc = len(self.dados.interc)
        nr_cont_imp = sum([len(s.ContratoImportacaoT) if s.ContratoImportacaoT else 0 for s in self.dados.sist])
        nr_cont_exp = sum([len(s.ContratoExportacaoT) if s.ContratoExportacaoT else 0 for s in self.dados.sist])
        month = self.dados.conf.DataInicial.month
        delta_time = self.dados.conf.DiscTemporalT['Duracao'].values
        delta_time_acum = np.cumsum(delta_time)
        nr_stages = len(delta_time)
        nr_var_all = nr_var_hidr*nr_hidr + nr_var_term*nr_unid_term + nr_cont_imp + nr_cont_exp + nr_interc + nr_submercados + 1
        # endregion

        self.resultados = otimo()

        # region Balanço Hídrico
        Aeq_BH = np.zeros((nr_hidr, nr_var_hidr*nr_hidr+nr_var_term*nr_unid_term+nr_cont_imp+nr_cont_exp+nr_interc+nr_submercados+1))
        for i, uhe in enumerate(self.dados.hidr):
            Aeq_BH[i, nr_var_hidr*i:nr_var_hidr*i+6] = [1, fator, fator, fator, fator, fator]  # v, q, s, dv, qb, qe
            idx_mont = [k for k, x in enumerate(self.dados.hidr) if x.Codigo in uhe.Montantes]
            for j in idx_mont:
                Aeq_BH[i, nr_var_hidr*j+1:nr_var_hidr*j+5] = -fator  # q, s, dv, qb (montantes)
        # endregion

        # region Atendimento à Demanda
        Aeq_AD = np.zeros((nr_submercados, nr_var_hidr * nr_hidr + nr_var_term * nr_unid_term + nr_cont_imp + nr_cont_exp + nr_interc + nr_submercados + 1))
        lin = 0
        col_imp = nr_var_hidr * nr_hidr + nr_var_term * nr_unid_term
        col_exp = nr_var_hidr * nr_hidr + nr_var_term * nr_unid_term + nr_cont_imp
        for sub in self.dados.sist:
            if sub.Sigla != 'FC':
                # Geração Hidrelétrica e Bombeamento
                for i, uhe in enumerate(self.dados.hidr):
                    if uhe.Sist == sub.Codigo:
                        Aeq_AD[lin, nr_var_hidr * i + 6] = 1  # gh
                    if uhe.ParamUsinaElev:
                        Aeq_AD[lin, nr_var_hidr * i + 6] = -uhe.ParamUsinaElev['Taxa']  # qb
                # Geração Termelétrica
                col_term = 0
                for ute in self.dados.term:
                    for j, unid in enumerate(ute.UnidadeGeradoraT):
                        if ute.Sist == sub.Codigo:
                            Aeq_AD[lin, nr_var_hidr * nr_hidr + col_term] = 1  # gt
                        col_term += nr_var_term
                # Contratos Importação/Exportação
                n_imp = len(sub.ContratoImportacaoT)
                n_exp = len(sub.ContratoExportacaoT)
                Aeq_AD[lin, col_imp:col_imp + n_imp] = 1
                Aeq_AD[lin, col_exp:col_exp + n_exp] = -1
                col_imp += n_imp
                col_exp += n_exp
                # Intercâmbio
                for k, interc in enumerate(self.dados.interc):
                    if interc.Para == sub.Sigla:
                        Aeq_AD[lin, nr_var_hidr * nr_hidr + nr_var_term * nr_unid_term + nr_cont_imp + nr_cont_exp + k] = 1
                    elif interc.De == sub.Sigla:
                        Aeq_AD[lin, nr_var_hidr * nr_hidr + nr_var_term * nr_unid_term + nr_cont_imp + nr_cont_exp + k] = -1
                # Deficit
                Aeq_AD[lin, nr_var_hidr * nr_hidr + nr_var_term * nr_unid_term + nr_cont_imp + nr_cont_exp + nr_interc + lin] = 1

                lin += 1
        # endregion

        # region Fluxo Intercâmbio (Ivaiporã e Ficticio)
        Aeq_FCIV = np.zeros((2, nr_var_hidr * nr_hidr + nr_var_term * nr_unid_term + nr_cont_imp + nr_cont_exp + nr_interc + nr_submercados + 1))
        for i, interc in enumerate(self.dados.interc):
            if interc.Para == 'FC':
                Aeq_FCIV[0, nr_var_hidr * nr_hidr + nr_var_term * nr_unid_term + nr_cont_imp + nr_cont_exp + i] = 1
            elif interc.De == 'FC':
                Aeq_FCIV[0, nr_var_hidr * nr_hidr + nr_var_term * nr_unid_term + nr_cont_imp + nr_cont_exp + i] = -1
            elif interc.Para == 'IV':
                Aeq_FCIV[1, nr_var_hidr * nr_hidr + nr_var_term * nr_unid_term + nr_cont_imp + nr_cont_exp + i] = 1
            elif interc.De == 'IV':
                Aeq_FCIV[1, nr_var_hidr * nr_hidr + nr_var_term * nr_unid_term + nr_cont_imp + nr_cont_exp + i] = -1
        # endregion

        # region Define tempo (em horas) de permanência ligado/desligado antes do início do período de estudo
        # Variação Rampa Up e Down
        TOn = np.zeros((nr_unid_term, nr_stages+1))
        TOff = np.zeros((nr_unid_term, nr_stages+1))
        StatusTerm = np.empty((nr_unid_term, nr_stages), dtype=object)
        A_RT = np.empty((0, nr_var_all))
        GTermIni = np.zeros((nr_unid_term, nr_stages + 1))
        unid_term = 0
        for ute in self.dados.term:
            for j, unid in enumerate(ute.UnidadeGeradoraT):
                if ute.UnidadeGeradoraT[unid]['StatusIni'] == 'LIGADA':
                    TOn[unid_term, 0] = ute.UnidadeGeradoraT[unid]['TempoPermStatus']
                    TOff[unid_term, 0] = 0.
                elif ute.UnidadeGeradoraT[unid]['StatusIni'] == 'DESLIGADA':
                    TOff[unid_term, 0] = ute.UnidadeGeradoraT[unid]['TempoPermStatus']
                    TOn[unid_term, 0] = 0.

                if ute.UnidadeGeradoraT[unid]['RUp']:
                    new_array = np.zeros((1, nr_var_all))
                    new_array[0, nr_var_hidr*nr_hidr+nr_var_term*unid_term] = 1
                    A_RT = np.append(A_RT, new_array, axis=0)

                if ute.UnidadeGeradoraT[unid]['RDown']:
                    new_array = np.zeros((1, nr_var_all))
                    new_array[0, nr_var_hidr*nr_hidr+nr_var_term*unid_term] = -1
                    A_RT = np.append(A_RT, new_array, axis=0)

                GTermIni[unid_term, 0] = ute.UnidadeGeradoraT[unid]['GerIni']

                unid_term += 1
        # endregion

        # Inicializa os volumes das usinas
        VIni = np.zeros((nr_hidr, nr_stages + 1))
        Defl = np.zeros((nr_hidr, nr_stages))
        for i, uhe in enumerate(self.dados.hidr):
            VIni[i, 0] = uhe.VolMin + (1/100) * uhe.VolIni * uhe.VolUtil

        # Inicializa função de custo futuro
        self.cf = list()
        for i in range(nr_stages):
            self.cf.append(fcf(i))

        # Processo iterativo
        self.iter = 1
        estab = 9e10
        self.ZINF = list()
        self.ZSUP = list()

        VAR = np.zeros((nr_var_all, nr_stages))
        LAMBDA = np.zeros((nr_hidr, nr_stages))
        CMO = np.zeros((nr_submercados, nr_stages))

        while True:

            t_pddd = timer()
            CustoImed = np.zeros(nr_stages)
            CustoFut = np.zeros(nr_stages)

            # region FORWARD
            for stage in range(nr_stages):

                # Função de Custo Futuro (FCF)
                if stage < nr_stages - 1:
                    [A_FCF, B_FCF] = self.cf[stage].calcAB(nr_var_all, nr_var_hidr)
                elif stage == nr_stages - 1:
                    [A_FCF, B_FCF] = self.last_stage_fcf_management(nr_var_all, nr_var_hidr)
                else:
                    A_FCF = np.array([]).reshape(0, nr_var_all)
                    B_FCF = np.array([]).reshape(0, 1)

                # region Gera vetor termos independente igualdade (Beq: =) e desigualdade (B: <=)
                Aeq_EV = np.zeros((nr_hidr, nr_var_hidr * nr_hidr + nr_var_term * nr_unid_term + nr_cont_imp + nr_cont_exp + nr_interc + nr_submercados + 1))
                Beq_BH = np.zeros((nr_hidr, 1))
                Beq_EV = np.zeros((nr_hidr, 1))
                Beq_AD = np.zeros((nr_submercados, 1))
                Beq_FCIV = np.zeros((2, 1))
                for i, uhe in enumerate(self.dados.hidr):
                    # Balanço Hídrico
                    Beq_BH[i, 0] = VIni[i, stage] + fator * uhe.VazIncPrevistaT.loc[stage, 'VAZAO']
                    idx_mont = [(k, x.Codigo) for k, x in enumerate(self.dados.hidr) if x.Codigo in uhe.DefAntMontanteT.keys()]
                    for j in idx_mont:
                        if uhe.DefAntMontanteT[j[1]]['TempoViagem'] >= delta_time_acum[stage]:
                            hours_ant = int(uhe.DefAntMontanteT[j[1]]['TempoViagem'] - delta_time_acum[stage])
                            days_ant = int(hours_ant / 24)
                            if ~np.isnan(uhe.DefAntMontanteT[j[1]]['Defl'].loc[days_ant, 'DEFL']):
                                Beq_BH[i, 0] += fator * uhe.DefAntMontanteT[j[1]]['Defl'].loc[days_ant, 'DEFL']
                        else:
                            hours_ant = delta_time_acum[stage]-uhe.DefAntMontanteT[j[1]]['TempoViagem']
                            idx_stage = np.searchsorted(delta_time_acum, hours_ant)
                            Beq_BH[i, 0] += fator * Defl[j[0], idx_stage]
                    # Evaporação
                    # if uhe.Evaporacao['ValorInicial'][month-1] >= 0:
                    # Aeq_EV[i, nr_var_hidr * i] = - (1 / 2) * uhe.Evaporacao['Coeficiente'][month - 1]  # v
                    Aeq_EV[i, nr_var_hidr * i + 5] = 1  #  1 / (3.6e-3 * delta_time[stage])  # qe
                    # Aeq_EV[i, nr_var_hidr * i + 7] = 1  # fqe
                    Beq_EV[i, 0] = 0.
                    # Beq_EV[i, 0] = uhe.Evaporacao['ValorInicial'][month-1] + \
                    #                     (1/2) * uhe.Evaporacao['Coeficiente'][month-1] * VIni[i, stage] - \
                    #                     uhe.Evaporacao['Coeficiente'][month-1] * VIni[i, 0]
                    # else:
                    #     Aeq_EV[i, nr_var_hidr * i] = (1 / 2) * uhe.Evaporacao['Coeficiente'][month - 1]  # v
                    #     Aeq_EV[i, nr_var_hidr * i + 5] = - 1 / (0.0036 * delta_time[stage])  # qe
                    #     Aeq_EV[i, nr_var_hidr * i + 7] = -1  # fqe
                    #     Beq_EV[i, 0] = - uhe.Evaporacao['ValorInicial'][month - 1] - \
                    #                    (1 / 2) * uhe.Evaporacao['Coeficiente'][month - 1] * VIni[i, stage] - \
                    #                    uhe.Evaporacao['Coeficiente'][month - 1] * (uhe.VolMin - VIni[i, 0])

                # Atendimento à Demanda
                for i, sub in enumerate(self.dados.sist):
                    if sub.Sigla != 'FC':
                        Beq_AD[i, 0] = sub.CargaT.loc[stage, 'CARGA']
                        if sub.Sigla == 'NE':
                            Beq_AD[i, 0] -= self.wind_energy[stage]
                # endregion

                # region Definição do Status de Decisão das Termelétricas no Estágio Atual
                # ON: deve respeitar tempo de permanência ligado
                # OFF: = deve respeitar tempo de permanência desligado
                # FREE = unidade está livre para tomar decisão
                B_RT = np.zeros((A_RT.shape[0], 1))
                unid_term, CustoTerm, lin = 0, [], 0
                for i, ute in enumerate(self.dados.term):
                    for j, unid in enumerate(ute.UnidadeGeradoraT):
                        CustoTerm.append((ute.UnidadeGeradoraT[unid]['Operacao'].loc[stage, 'Custo'], i))
                        if TOn[unid_term, stage] >= ute.UnidadeGeradoraT[unid]['TOn'] and TOff[unid_term, stage] == 0.:
                            StatusTerm[unid_term, stage] = 'FREE'
                        elif TOff[unid_term, stage] >= ute.UnidadeGeradoraT[unid]['TOff'] and TOn[unid_term, stage] == 0.:
                            StatusTerm[unid_term, stage] = 'FREE'
                        elif TOn[unid_term, stage] < ute.UnidadeGeradoraT[unid]['TOn'] and TOff[unid_term, stage] == 0.:
                            StatusTerm[unid_term, stage] = 'ON'
                        elif TOff[unid_term, stage] < ute.UnidadeGeradoraT[unid]['TOff'] and TOn[unid_term, stage] == 0.:
                            StatusTerm[unid_term, stage] = 'OFF'
                        else:
                            print('Erro! Decisão termelétrica')
                            break

                        if ute.UnidadeGeradoraT[unid]['RUp']:
                            if ute.UnidadeGeradoraT[unid]['CapacidadeT'].loc[stage, 'CAPACIDADE'] != 0. and StatusTerm[unid_term, stage] != 'OFF':
                                B_RT[lin, 0] = ute.UnidadeGeradoraT[unid]['RUp'] * delta_time[stage] + GTermIni[unid_term, stage]
                            lin += 1
                        if ute.UnidadeGeradoraT[unid]['RDown']:
                            if ute.UnidadeGeradoraT[unid]['CapacidadeT'].loc[stage, 'CAPACIDADE'] != 0. and StatusTerm[unid_term, stage] != 'OFF':
                                B_RT[lin, 0] = ute.UnidadeGeradoraT[unid]['RDown'] * delta_time[stage] - GTermIni[unid_term, stage]
                            lin += 1

                        unid_term += 1
                # endregion

                # region Geração Hidráulica (cortes da FPH)
                A_GH, B_GH = self.fph_constraints_generate(nr_var_hidr=nr_var_hidr, nr_var_all=nr_var_all, stage=stage)
                # endregion

                # Composição das matrizes para otimização(igualdade)
                Aeq = np.concatenate((Aeq_BH, Aeq_AD, Aeq_EV, Aeq_FCIV), axis=0)
                Beq = np.concatenate([Beq_BH, Beq_AD, Beq_EV, Beq_FCIV], axis=0)
                # Aeq = np.concatenate((Aeq_BH, Aeq_EV, Aeq_FCIV), axis=0)
                # Beq = np.concatenate([Beq_BH, Beq_EV, Beq_FCIV], axis=0)

                # Concatena Rampa Térmica, Geracação Hidráulica e FCF
                A = np.concatenate((A_RT, A_GH, A_FCF), axis=0)
                B = np.concatenate((B_RT, B_GH, B_FCF), axis=0)
                # A = np.concatenate((A_RT, A_FCF), axis=0)
                # B = np.concatenate((B_RT, B_FCF), axis=0)

                # Determina a canalização das variáveis
                [c, lb, ub] = self.canalizacao(nr_hidr, nr_unid_term, nr_cont_imp, nr_cont_exp, nr_interc, nr_submercados, StatusTerm[:, stage], stage, month)

                t = timer()
                [X, DUAL, FVAL] = MosekProcess(Aeq, Beq, A, B, c, lb, ub).run()
                tempo = (timer() - t)
                print(f'Iteração {self.iter} : Tempo = {round(tempo, 4)} s   ->   Estágio: {stage}    (FORWARD)')

                VAR[:, stage] = X
                LAMBDA[:, stage] = DUAL[:nr_hidr]
                CMO[:, stage] = DUAL[nr_hidr:nr_hidr+nr_submercados]

                for i in range(nr_hidr):
                    VIni[i, stage + 1] = X[nr_var_hidr*i]
                    Defl[i, stage] = X[nr_var_hidr*i+1] + X[nr_var_hidr*i+2]
                for j in range(nr_unid_term):
                    gt = X[nr_var_hidr*nr_hidr+j]
                    GTermIni[j, stage + 1] = gt
                    if gt > 0.:  # Ligada
                        TOn[j, stage + 1] = TOn[j, stage] + delta_time[stage]
                        TOff[j, stage + 1] = 0
                    else:  # Desligada
                        TOff[j, stage + 1] = TOff[j, stage] + delta_time[stage]
                        TOn[j, stage + 1] = 0

                CustoImed[stage] = FVAL - X[-1] - penal_vert * sum(X[2:nr_var_hidr*nr_hidr:nr_var_hidr])  # - penal_folga_evap * sum(X[7:nr_var_hidr*nr_hidr:nr_var_hidr])
                CustoFut[stage] = X[-1]

            self.resultados.save(self.dados, VARIAVEIS=VAR, CUSTO=CustoImed, LAMBDA=LAMBDA, CMO=CMO, ALFA=CustoFut, nr_var_hidr=nr_var_hidr, nr_var_term=nr_var_term, wind_energy=self.wind_energy)

            # break
            # endregion

            CustoTot = CustoImed + CustoFut
            ZINF = CustoTot[0]
            ZSUP = np.sum(CustoImed)
            self.ZINF.append(ZINF)
            self.ZSUP.append(ZSUP)

            erro = self.ZSUP[self.iter - 1] - self.ZINF[self.iter - 1]
            tol_perc = 0.005 * self.ZSUP[self.iter-1]

            print('Iteração:', self.iter,
                  ' - ZINF:', self.ZINF[self.iter-1],
                  ' - ZSUP:', self.ZSUP[self.iter-1],
                  ' - Erro:', erro,
                  ' - Tempo:', round(timer() - t_pddd, 2), 'seg')

            if self.iter >= 3:
                media = np.mean(self.ZINF[-3:])
                estab = np.abs(media - self.ZINF[-1])

            # Problem convergence verification
            if (abs(estab) < tol_perc) or (abs(erro) < tol_perc):
                break

            # region BACKWARD
            for stage in range(nr_stages - 1, 0, -1):

                # Função de Custo Futuro (FCF)
                if stage < nr_stages - 1:
                    [A_FCF, B_FCF] = self.cf[stage].calcAB(nr_var_all, nr_var_hidr)
                elif stage == nr_stages - 1:
                    [A_FCF, B_FCF] = self.last_stage_fcf_management(nr_var_all, nr_var_hidr)
                else:
                    A_FCF = np.array([]).reshape(0, nr_var_all)
                    B_FCF = np.array([]).reshape(0, 1)

                # region Gera vetor termos independente igualdade (Beq: =) e desigualdade (B: <=)
                Aeq_EV = np.zeros((nr_hidr, nr_var_hidr * nr_hidr + nr_var_term * nr_unid_term + nr_cont_imp + nr_cont_exp + nr_interc + nr_submercados + 1))
                Beq_BH = np.zeros((nr_hidr, 1))
                Beq_EV = np.zeros((nr_hidr, 1))
                Beq_AD = np.zeros((nr_submercados, 1))
                Beq_FCIV = np.zeros((2, 1))
                for i, uhe in enumerate(self.dados.hidr):
                    # Balanço Hídrico
                    Beq_BH[i, 0] = VIni[i, stage] + fator * uhe.VazIncPrevistaT.loc[stage, 'VAZAO']
                    idx_mont = [(k, x.Codigo) for k, x in enumerate(self.dados.hidr) if
                                x.Codigo in uhe.DefAntMontanteT.keys()]
                    for j in idx_mont:
                        if uhe.DefAntMontanteT[j[1]]['TempoViagem'] >= delta_time_acum[stage]:
                            hours_ant = int(uhe.DefAntMontanteT[j[1]]['TempoViagem'] - delta_time_acum[stage])
                            days_ant = int(hours_ant / 24)
                            if ~np.isnan(uhe.DefAntMontanteT[j[1]]['Defl'].loc[days_ant, 'DEFL']):
                                Beq_BH[i, 0] += fator * uhe.DefAntMontanteT[j[1]]['Defl'].loc[days_ant, 'DEFL']
                        else:
                            hours_ant = delta_time_acum[stage] - uhe.DefAntMontanteT[j[1]]['TempoViagem']
                            idx_stage = np.searchsorted(delta_time_acum, hours_ant)
                            Beq_BH[i, 0] += fator * Defl[j[0], idx_stage]
                    # Evaporação
                    # if uhe.Evaporacao['ValorInicial'][month-1] >= 0:
                    # Aeq_EV[i, nr_var_hidr * i] = - (1 / 2) * uhe.Evaporacao['Coeficiente'][month - 1]  # v
                    Aeq_EV[i, nr_var_hidr * i + 5] = 1  # 1 / (3.6e-3 * delta_time[stage])  # qe
                    # Aeq_EV[i, nr_var_hidr * i + 7] = 1  # fqe
                    Beq_EV[i, 0] = 0.
                    # Beq_EV[i, 0] = uhe.Evaporacao['ValorInicial'][month-1] + \
                    #                     (1/2) * uhe.Evaporacao['Coeficiente'][month-1] * VIni[i, stage] - \
                    #                     uhe.Evaporacao['Coeficiente'][month-1] * VIni[i, 0]
                    # else:
                    #     Aeq_EV[i, nr_var_hidr * i] = (1 / 2) * uhe.Evaporacao['Coeficiente'][month - 1]  # v
                    #     Aeq_EV[i, nr_var_hidr * i + 5] = - 1 / (0.0036 * delta_time[stage])  # qe
                    #     Aeq_EV[i, nr_var_hidr * i + 7] = -1  # fqe
                    #     Beq_EV[i, 0] = - uhe.Evaporacao['ValorInicial'][month - 1] - \
                    #                    (1 / 2) * uhe.Evaporacao['Coeficiente'][month - 1] * VIni[i, stage] - \
                    #                    uhe.Evaporacao['Coeficiente'][month - 1] * (uhe.VolMin - VIni[i, 0])

                # Atendimento à Demanda
                for i, sub in enumerate(self.dados.sist):
                    if sub.Sigla != 'FC':
                        Beq_AD[i, 0] = sub.CargaT.loc[stage, 'CARGA']
                        if sub.Sigla == 'NE':
                            Beq_AD[i, 0] -= self.wind_energy[stage]
                # endregion

                # region Definição do Status de Decisão das Termelétricas no Estágio Atual
                # ON: deve respeitar tempo de permanência ligado
                # OFF: = deve respeitar tempo de permanência desligado
                # FREE = unidade está livre para tomar decisão
                B_RT = np.zeros((A_RT.shape[0], 1))
                unid_term, CustoTerm, lin = 0, [], 0
                for i, ute in enumerate(self.dados.term):
                    for j, unid in enumerate(ute.UnidadeGeradoraT):
                        CustoTerm.append((ute.UnidadeGeradoraT[unid]['Operacao'].loc[stage, 'Custo'], i))
                        if TOn[unid_term, stage] >= ute.UnidadeGeradoraT[unid]['TOn'] and TOff[unid_term, stage] == 0.:
                            StatusTerm[unid_term, stage] = 'FREE'
                        elif TOff[unid_term, stage] >= ute.UnidadeGeradoraT[unid]['TOff'] and TOn[
                            unid_term, stage] == 0.:
                            StatusTerm[unid_term, stage] = 'FREE'
                        elif TOn[unid_term, stage] < ute.UnidadeGeradoraT[unid]['TOn'] and TOff[unid_term, stage] == 0.:
                            StatusTerm[unid_term, stage] = 'ON'
                        elif TOff[unid_term, stage] < ute.UnidadeGeradoraT[unid]['TOff'] and TOn[
                            unid_term, stage] == 0.:
                            StatusTerm[unid_term, stage] = 'OFF'
                        else:
                            print('Erro! Decisão termelétrica')
                            break

                        if ute.UnidadeGeradoraT[unid]['RUp']:
                            if ute.UnidadeGeradoraT[unid]['CapacidadeT'].loc[stage, 'CAPACIDADE'] != 0. and StatusTerm[unid_term, stage] != 'OFF':
                                B_RT[lin, 0] = ute.UnidadeGeradoraT[unid]['RUp'] * delta_time[stage] + GTermIni[
                                    unid_term, stage]
                            lin += 1
                        if ute.UnidadeGeradoraT[unid]['RDown']:
                            if ute.UnidadeGeradoraT[unid]['CapacidadeT'].loc[stage, 'CAPACIDADE'] != 0. and StatusTerm[unid_term, stage] != 'OFF':
                                B_RT[lin, 0] = ute.UnidadeGeradoraT[unid]['RDown'] * delta_time[stage] - GTermIni[
                                    unid_term, stage]
                            lin += 1

                        unid_term += 1
                # endregion

                # region Geração Hidráulica (cortes da FPH)
                A_GH, B_GH = self.fph_constraints_generate(nr_var_hidr=nr_var_hidr, nr_var_all=nr_var_all, stage=stage)
                # endregion

                # Composição das matrizes para otimização(igualdade)
                Aeq = np.concatenate((Aeq_BH, Aeq_AD, Aeq_EV, Aeq_FCIV), axis=0)
                Beq = np.concatenate([Beq_BH, Beq_AD, Beq_EV, Beq_FCIV], axis=0)
                # Aeq = np.concatenate((Aeq_BH, Aeq_EV, Aeq_FCIV), axis=0)
                # Beq = np.concatenate([Beq_BH, Beq_EV, Beq_FCIV], axis=0)

                # Concatena Rampa Térmica, Geracação Hidráulica e FCF
                A = np.concatenate((A_RT, A_GH, A_FCF), axis=0)
                B = np.concatenate((B_RT, B_GH, B_FCF), axis=0)
                # A = np.concatenate((A_RT, A_FCF), axis=0)
                # B = np.concatenate((B_RT, B_FCF), axis=0)

                # Determina a canalização das variáveis
                [c, lb, ub] = self.canalizacao(nr_hidr, nr_unid_term, nr_cont_imp, nr_cont_exp, nr_interc, nr_submercados, StatusTerm[:, stage], stage, month)

                t = timer()
                [X, DUAL, FVAL] = MosekProcess(Aeq, Beq, A, B, c, lb, ub).run()
                tempo = (timer() - t)
                print(f'Iteração {self.iter} : Tempo = {round(tempo, 4)} s   ->   Estágio: {stage}    (BACKWARD)')

                # Geração dos cortes da FCF
                vol_ini = VIni[:, stage]
                custo = FVAL - penal_vert * sum(X[2:nr_var_hidr*nr_hidr:nr_var_hidr]) # - penal_folga_evap * sum(X[7:nr_var_hidr*nr_hidr:nr_var_hidr])
                lagrange = [min(x, 0.) for x in DUAL[:nr_hidr]]

                self.cf[stage - 1].geraFCF(vol_ini, custo, lagrange)

                # endregion

            self.iter += 1

    def canalizacao(self, nr_hidr, nr_unid_term, nr_cont_imp, nr_cont_exp, nr_interc, nr_submercado, StatusTerm, stage, month):

        c = np.zeros(nr_hidr*nr_var_hidr+nr_unid_term*nr_var_term+nr_cont_imp+nr_cont_exp+nr_interc+nr_submercado+1)
        lb = np.zeros(nr_hidr*nr_var_hidr+nr_unid_term*nr_var_term+nr_cont_imp+nr_cont_exp+nr_interc+nr_submercado+1)
        ub = np.zeros(nr_hidr*nr_var_hidr+nr_unid_term*nr_var_term+nr_cont_imp+nr_cont_exp+nr_interc+nr_submercado+1)

        # Hidrelétrica
        for i, uhe in enumerate(self.dados.hidr):
            # Volume Armazenado
            c[nr_var_hidr*i] = 0.
            lb[nr_var_hidr*i] = uhe.VolMin
            ub[nr_var_hidr*i] = uhe.VolMax
            # Vazão Turbinada
            c[nr_var_hidr * i + 1] = 0.
            lb[nr_var_hidr * i + 1] = 0.
            if uhe.EngolMaxT is not None:
                ub[nr_var_hidr * i + 1] = uhe.EngolMaxT.loc[stage, 'ENGOL']
            # Vazão Vertida
            c[nr_var_hidr * i + 2] = penal_vert
            lb[nr_var_hidr * i + 2] = 0.
            ub[nr_var_hidr * i + 2] = None
            # Vazão Desviada
            c[nr_var_hidr * i + 3] = 0.
            if np.isnan(uhe.DesvioAguaT.loc[stage, 'TX']):
                lb[nr_var_hidr * i + 3] = 0.
                ub[nr_var_hidr * i + 3] = 0.
            else:
                if uhe.DesvioAguaT.loc[stage, 'TX'] >= 0.:
                    lb[nr_var_hidr * i + 3] = 0.
                    ub[nr_var_hidr * i + 3] = uhe.DesvioAguaT.loc[stage, 'TX']
                else:
                    lb[nr_var_hidr * i + 3] = uhe.DesvioAguaT.loc[stage, 'TX']
                    ub[nr_var_hidr * i + 3] = 0.
            # Vazão Bombeada
            c[nr_var_hidr * i + 4] = 0.
            if uhe.ParamUsinaElev:
                lb[nr_var_hidr * i + 4] = uhe.ParamUsinaElev['VazMin']
                ub[nr_var_hidr * i + 4] = uhe.ParamUsinaElev['VazMax']
            # Vazão Evaporada
            c[nr_var_hidr * i + 5] = 0.
            lb[nr_var_hidr * i + 5] = None
            ub[nr_var_hidr * i + 5] = None

            # vol_ref = uhe.VolMin + (1/100) * uhe.VolIni * uhe.VolUtil
            # if uhe.Evaporacao['ValorInicial'][month - 1] >= 0:
            #     lb[nr_var_hidr * i + 5] = None
            #     ub[nr_var_hidr * i + 5] = None # abs(uhe.Evaporacao['ValorInicial'][month - 1] + \
            #                            # (1/2) * uhe.Evaporacao['Coeficiente'][month - 1] * uhe.VolMax + \
            #                            #  (1 / 2) * uhe.Evaporacao['Coeficiente'][month - 1] * vol_ref +
            #                            # uhe.Evaporacao['Coeficiente'][month - 1] * (uhe.VolMin - vol_ref))
            # else:
            #     lb[nr_var_hidr * i + 5] = None  #  - abs(uhe.Evaporacao['ValorInicial'][month - 1] + \
            #                            # (1/2) * uhe.Evaporacao['Coeficiente'][month - 1] * uhe.VolMax + \
            #                            #  (1 / 2) * uhe.Evaporacao['Coeficiente'][month - 1] * vol_ref +
            #                            # uhe.Evaporacao['Coeficiente'][month - 1] * (uhe.VolMin - vol_ref))
            #     ub[nr_var_hidr * i + 5] = None

            # lb[nr_var_hidr * i + 5] = None  # -abs(uhe.Evaporacao['ValorInicial'][month - 1] + \
            #                           # (1 / 2) * uhe.Evaporacao['Coeficiente'][month - 1] * uhe.VolMax + \
            #                           # uhe.Evaporacao['Coeficiente'][month - 1] * (uhe.VolMin - uhe.VolMax))
            # ub[nr_var_hidr * i + 5] = None  # abs(uhe.Evaporacao['ValorInicial'][month - 1] + \
                                      # (1 / 2) * uhe.Evaporacao['Coeficiente'][month - 1] * uhe.VolMax + \
                                      # uhe.Evaporacao['Coeficiente'][month - 1] * (uhe.VolMin - uhe.VolMax))
            # print(f'lb: {lb[nr_var_hidr * i + 5]}     ub: {ub[nr_var_hidr * i + 5]}    coeficiente: {uhe.Evaporacao["Coeficiente"][month - 1]}')

            # Geração Hidrelétrica
            c[nr_var_hidr * i + 6] = 0.
            lb[nr_var_hidr * i + 6] = 0.
            # ub[nr_var_hidr * i + 6] = None
            if uhe.PotEfetT is not None:
                ub[nr_var_hidr * i + 6] = uhe.PotEfetT.loc[stage, 'POT']
            # # Variável Folga Geração Hidrelétrica
            c[nr_var_hidr * i + 7] = 0.
            lb[nr_var_hidr * i + 7] = None
            ub[nr_var_hidr * i + 7] = None  # abs(uhe.Evaporacao['ValorInicial'][month - 1])

        # Termelétrica
        pos = 0
        for ute in self.dados.term:
            for j, unid in enumerate(ute.UnidadeGeradoraT):
                # Geração Termelétrica
                c[nr_var_hidr * nr_hidr + pos] = ute.UnidadeGeradoraT[unid]['Operacao'].loc[stage, 'Custo']
                if StatusTerm[pos] == 'ON':
                    lb[nr_var_hidr * nr_hidr + pos] = 0.  # ute.UnidadeGeradoraT[unid]['GerMinAcion']
                    ub[nr_var_hidr * nr_hidr + pos] = ute.UnidadeGeradoraT[unid]['CapacidadeT'].loc[stage, 'CAPACIDADE']
                elif StatusTerm[pos] == 'OFF':
                    lb[nr_var_hidr * nr_hidr + pos] = 0.
                    ub[nr_var_hidr * nr_hidr + pos] = 0.
                elif StatusTerm[pos] == 'FREE':
                    lb[nr_var_hidr * nr_hidr + pos] = 0.
                    ub[nr_var_hidr * nr_hidr + pos] = ute.UnidadeGeradoraT[unid]['CapacidadeT'].loc[stage, 'CAPACIDADE']
                else:
                    print('Decisão Termelétrica não definida!')
                    break

                pos += nr_var_term

        # Contratos de Importação/Exportação
        pos_imp, pos_exp, pos_interc = 0, 0, 0
        for sub in self.dados.sist:
            if sub.Sigla != 'FC':
                # Contratos Importação:
                for i, cont in enumerate(sub.ContratoImportacaoT):
                    c[nr_var_hidr * nr_hidr + nr_var_term * nr_unid_term + pos_imp + i] = sub.ContratoImportacaoT[cont].loc[stage, 'Custo']
                    lb[nr_var_hidr * nr_hidr + nr_var_term * nr_unid_term + pos_imp + i] = sub.ContratoImportacaoT[cont].loc[stage, 'LimInf']
                    ub[nr_var_hidr * nr_hidr + nr_var_term * nr_unid_term + pos_imp + i] = sub.ContratoImportacaoT[cont].loc[stage, 'LimSup']
                # Contratos Exportação:
                for i, cont in enumerate(sub.ContratoExportacaoT):
                    c[nr_var_hidr * nr_hidr + nr_var_term * nr_unid_term + pos_exp + i] = -sub.ContratoExportacaoT[cont].loc[stage, 'Custo']
                    lb[nr_var_hidr * nr_hidr + nr_var_term * nr_unid_term + pos_exp + i] = sub.ContratoExportacaoT[cont].loc[stage, 'LimInf']
                    ub[nr_var_hidr * nr_hidr + nr_var_term * nr_unid_term + pos_exp + i] = sub.ContratoExportacaoT[cont].loc[stage, 'LimSup']

                pos_imp += len(sub.ContratoImportacaoT)
                pos_exp += len(sub.ContratoExportacaoT)

        # Intercâmbio
        for idx, interc in enumerate(self.dados.interc):
            c[nr_var_hidr * nr_hidr + nr_var_term * nr_unid_term + nr_cont_imp + nr_cont_exp + idx] = 0.
            lb[nr_var_hidr * nr_hidr + nr_var_term * nr_unid_term + nr_cont_imp + nr_cont_exp + idx] = 0.
            ub[nr_var_hidr * nr_hidr + nr_var_term * nr_unid_term + nr_cont_imp + nr_cont_exp + idx] = interc.LimiteT.loc[stage, 'Limite']

        # Déficit
        pos = 0
        for sub in self.dados.sist:
            if sub.Sigla != 'FC':
                c[nr_var_hidr * nr_hidr + nr_var_term * nr_unid_term + nr_cont_imp + nr_cont_exp + nr_interc + pos] = sub.CustoDeficit
                lb[nr_var_hidr * nr_hidr + nr_var_term * nr_unid_term + nr_cont_imp + nr_cont_exp + nr_interc + pos] = 0.
                ub[nr_var_hidr * nr_hidr + nr_var_term * nr_unid_term + nr_cont_imp + nr_cont_exp + nr_interc + pos] = None

                pos += 1

        # Custo Futuro
        c[-1] = 1.
        lb[-1] = 0.
        ub[-1] = None  # infinito float('inf')  #

        return c, lb, ub

    def plot_convergencia(self, time: float, processos: int = 1):
        plt.figure()
        plt.plot(np.arange(1, self.iter + 1), self.ZINF, marker='o', label='ZINF')
        plt.errorbar(np.arange(1, self.iter + 1), self.ZSUP, 1.96 * np.array(self.sigma), marker='o', label='ZSUP')
        plt.xlabel('Iteração')
        plt.ylabel('Custo [R$]')
        plt.title('Tempo de processamento: %4.2f min   Processos: %i' % (time, processos))
        plt.legend()
        plt.show()
