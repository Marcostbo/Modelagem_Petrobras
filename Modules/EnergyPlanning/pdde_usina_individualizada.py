# coding=utf-8

from fcf import fcf
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer
from cvxopt.modeling import op, variable, matrix, dot
import cvxopt.solvers as solvers
import time

class pdde_SistIsol(object):

    sist = None
    cf = None
    ZINF = list()
    ZSUP = list()
    iter = 0
    sigma = list()
    CI_serie = None

    def __init__(self, sistema):
        self.sist = sistema

    def run(self, nr_meses, aberturas, pesos_aber, afluencias, demandas, pesos_dem):

        solvers.options['show_progress'] = False
        # solvers.options['glpk'] = dict(msg_lev='GLP_MSG_OFF')
        type = 'dense'
        solver = 'default'

        # Parâmetros iniciais
        nr_serfor = afluencias.shape[1]
        nr_aber = aberturas.shape[1]
        nr_dem = demandas.shape[0]
        nr_sist = self.sist.conf_uh[0].Sist
        nr_uhe = len(self.sist.conf_uh)
        nr_ute = len(self.sist.conf_ut)

        # Incializa matrizes da otimizacao
        Aeq = np.zeros((nr_uhe + nr_dem + 1, 3 * nr_uhe + nr_dem * (2 + nr_ute) + 1))
        Beq = np.zeros(nr_uhe + nr_dem + 1)

        BH = np.zeros((nr_uhe, 3 * nr_uhe))
        jusantes = [i.Jusante for i in self.sist.conf_uh]
        for iusi in range(nr_uhe):
            BH[iusi][iusi * 3] = 1
            BH[iusi][iusi * 3 + 1] = 2.592
            BH[iusi][iusi * 3 + 2] = 2.592
            codigo = self.sist.conf_uh[iusi].Codigo
            try:
                lista = [i for i, x in enumerate(jusantes) if x == codigo]
                for jusi in lista:
                    BH[iusi][jusi * 3 + 1] = -2.592
                    BH[iusi][jusi * 3 + 2] = -2.592
            except ValueError:
                pass

        AD = np.ones((1, 2 + nr_ute))
        ET = np.zeros((1, 3 * nr_uhe))
        for iusi in range(nr_uhe):
            ET[0][iusi * 3 + 1] = self.sist.conf_uh[iusi].Ro65[0][0]

        # Insere BH e AD na Aeq
        Aeq[0:nr_uhe, 0:3 * nr_uhe] = BH
        lin = nr_uhe
        col = 3 * nr_uhe
        for idem in range(nr_dem):
            Aeq[lin, col:col + nr_ute + 2] = AD
            Aeq[nr_uhe + nr_dem, col + nr_ute] = -pesos_dem[0][idem]
            lin += 1
            col += AD.shape[1]
        Aeq[nr_uhe + nr_dem, 0:3 * nr_uhe] = ET

        # Determina a canalização das variáveis
        [c, lb, ub] = self.canalizacao(nr_uhe, nr_ute, nr_sist, nr_dem, pesos_dem)

        # Inicializa os volumes das usinas
        VIni = np.zeros((nr_uhe, nr_serfor, nr_meses + 1))
        for iusi in range(nr_uhe):
            VIni[iusi, :, 0] = self.sist.conf_uh[iusi].VolMin + (1 / 100) * self.sist.conf_uh[iusi].VolIni * self.sist.conf_uh[iusi].VolUtil

        self.cf = list()
        for imes in range(nr_meses):
            self.cf.append(fcf(imes))

        # Processo iterativo
        iter_LP = 1
        self.iter = 0
        erro = 9e10
        tol = 0.1
        tol_IC = 0.1
        tol_perc = 0.1
        estab = 9e10
        while ((abs(erro) > tol_IC) or (abs(estab) > tol_perc)) and (abs(erro) > tol_perc):   # && (abs(estab) > tol) %&& (abs(erro) > tol_IC)

            t = timer()

            # FORWARD
            ZINF = 0.
            ZSUP = 0.

            CustoImed = np.zeros((nr_meses, nr_serfor))
            CustoFut = np.zeros((nr_meses, nr_serfor))
            CustoTot = np.zeros((nr_meses, nr_serfor))
            for imes in range(nr_meses):

                if imes < nr_meses-1:
                    [A, B, nr_cortes] = self.cf[imes].calcAB(Aeq)
                else:
                    A = np.array([]).reshape(0, 3 * nr_uhe + nr_dem * (2 + nr_ute) + 1)
                    B = np.array([]).reshape(0, 1)
                    nr_cortes = 0

                for iser in range(nr_serfor):

                    # Preenche Beq
                    for iusi in range(nr_uhe):
                        Beq[iusi] = VIni[iusi, iser, imes] + 2.592 * afluencias[iusi, iser, imes]
                    Beq[nr_uhe:-1] = [demandas[i, imes] for i in range(nr_dem)]
                    Beq[-1] = 0.

                    x = variable(3 * nr_uhe + nr_dem * (2 + nr_ute) + 1)  # inicializa variaveis
                    [fob, restricoes, Aeq_matrix, Beq_matrix, A_matrix, B_matrix, c_matrix] = self.reuni(x, Aeq, Beq, A, B, c, lb, ub, nr_uhe, nr_ute, nr_dem, nr_cortes)

                    problema = op(fob, restricoes)
                    problema.solve(type, solver)
                    sol = solvers.lp(c_matrix, A_matrix, B_matrix, Aeq_matrix, Beq_matrix)

                    if problema.status != 'optimal' and sol['status'] != 'optimal':
                        print('CONVERGÊNCIA NÃO ALCANÇADA: ', problema.status, '-> FORWARD')
                    print('mês: ', imes, '   série: ', iser)

                    # Atualiza armazenamento das hidrelétricas
                    for iusi in range(nr_uhe):
                        VIni[iusi, iser, imes+1] = x.value[3*iusi]

                    CustoImed[imes, iser] = fob.value()[0] - 0.001 * sum(x.value[2:3*nr_uhe:3]) - x.value[-1]
                    CustoFut[imes, iser] = x.value[-1]
                    CustoTot[imes, iser] = fob.value()[0] - 0.001 * sum(x.value[2:3*nr_uhe:3])

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
            for imes in range(nr_meses-1, 0, -1):

                if imes < nr_meses-1:
                    [A, B, nr_cortes] = self.cf[imes].calcAB(Aeq)
                else:
                    A = np.array([]).reshape(0, 3 * nr_uhe + nr_dem * (2 + nr_ute) + 1)
                    B = np.array([]).reshape(0, 1)
                    nr_cortes = 0

                for iser in range(nr_serfor):

                    LAMBDA = np.zeros((nr_aber, nr_uhe))
                    CUSTO = np.zeros(nr_aber)
                    for iaber in range(nr_aber):

                        # Preenche Beq
                        for iusi in range(nr_uhe):
                            Beq[iusi] = VIni[iusi, iser, imes] + 2.592 * aberturas[iusi, iaber, imes]
                        Beq[nr_uhe:-1] = [demandas[i, imes] for i in range(nr_dem)]
                        Beq[-1] = 0.

                        x = variable(3 * nr_uhe + nr_dem * (2 + nr_ute) + 1)  # inicializa variaveis
                        [fob, restricoes, Aeq_matrix, Beq_matrix, A_matrix, B_matrix, c_matrix] = self.reuni(x, Aeq, Beq, A, B, c, lb, ub, nr_uhe, nr_ute, nr_dem, nr_cortes)

                        problema = op(fob, restricoes)
                        problema.solve(type, solver)
                        sol = solvers.lp(c_matrix, A_matrix, B_matrix, Aeq_matrix, Beq_matrix)

                        if problema.status != 'optimal' and sol['status'] != 'optimal':
                            print('CONVERGÊNCIA NÃO ALCANÇADA: ', problema.status, '-> BACKWARD')
                        print('mês: ', imes, '   série: ', iser, '   abertura: ', iaber)

                        CUSTO[iaber] = fob.value()[0] - 0.001 * sum(x.value[2:3 * nr_uhe:3])
                        for iusi in range(nr_uhe):
                            if restricoes[0].multiplier.value[iusi] > 0.:
                                LAMBDA[iaber, iusi] = -restricoes[0].multiplier.value[iusi] # 0.
                            else:
                                LAMBDA[iaber, iusi] = 0.

                    lambda_mean = np.matmul(pesos_aber[imes, :], LAMBDA)
                    custo_mean = np.matmul(pesos_aber[imes, :], CUSTO)

                    vol_ini = np.zeros((nr_uhe, 1))
                    for iusi in range(nr_uhe):
                        vol_ini[iusi, 0] = VIni[iusi, iser, imes]

                    # Função que gera o corte da FCF
                    self.cf[imes-1].geraFCF(vol_ini, custo_mean, lambda_mean)


            print('Iteração:', self.iter + 1,
                  ' - ZINF:', self.ZINF[self.iter],
                  ' - ZSUP:', self.ZSUP[self.iter],
                  ' - Erro:', erro,
                  ' - Tempo:', round(timer() - t, 2), 'seg')

            self.iter += 1

    def canalizacao(self, nr_uhe, nr_ute, nr_sist, nr_dem, pesos_dem):
        c = np.zeros(3 * nr_uhe + nr_dem * (2 + nr_ute) + 1)
        lb = np.zeros(3 * nr_uhe + nr_dem * (2 + nr_ute) + 1)
        ub = np.zeros(3 * nr_uhe + nr_dem * (2 + nr_ute) + 1)

        c_energ = np.zeros(2 + nr_ute)
        lb_energ = np.zeros(2 + nr_ute)
        ub_energ = np.zeros(2 + nr_ute)

        # Dados das Hidrelétricas
        col = 0
        for iusi in range(nr_uhe):
            c[col] = 0.
            c[col + 1] = 0.
            c[col + 2] = 0.001
            lb[col] = self.sist.conf_uh[iusi].VolMin
            lb[col + 1] = 0.
            lb[col + 2] = 0.
            ub[col] = self.sist.conf_uh[iusi].VolMax
            ub[col + 1] = self.sist.conf_uh[iusi].Engolimento
            ub[col + 2] = 9e15  #infinito
            col += 3

        # Dados das Termelétricas
        col = 0
        for iute in range(nr_ute):
            c_energ[col] = self.sist.conf_ut[iute].Custo[0]
            lb_energ[col] = 0.
            ub_energ[col] = self.sist.conf_ut[iute].Potencia
            col += 1

        # Dados de Energia de Todas Hidrelétricas
        col = nr_ute
        energ = 0
        for iusi in range(nr_uhe):
            energ += self.sist.conf_uh[iusi].Engolimento * self.sist.conf_uh[iusi].Ro65[0][0]
        c_energ[col] = 0.
        lb_energ[col] = 0.
        ub_energ[col] = energ

        # Dados de Déficit
        col = nr_ute + 1
        c_energ[col] = self.sist.submercado[nr_sist].CustoDeficit[0]
        lb_energ[col] = 0.
        ub_energ[col] = 9e15  #infinito

        # Dados completos
        col = 3 * nr_uhe
        for idem in range(nr_dem):
            c[col:col + c_energ.size] = pesos_dem[0][idem] * c_energ
            lb[col:col + c_energ.size] = lb_energ
            ub[col:col + c_energ.size] = ub_energ
            col += c_energ.size
        c[-1] = 1.
        lb[-1] = 0.
        ub[-1] = 9e15  #infinito

        lb = lb.transpose()
        ub = ub.transpose()

        return c, lb, ub

    def reuni(self, x, Aeq, Beq, A, B, c, lb, ub, nr_uhe, nr_ute, nr_dem, nr_cortes):

        # Eliminar restrições de limites superiores
        aux = list()    # linhas eliminadas
        aux_2 = list()  # linhas usadas
        col = 0
        for iuhe in range(nr_uhe):
            aux.append(col+2)
            aux_2.append(col)
            aux_2.append(col+1)
            col += 3
        col = 3*nr_uhe
        for idem in range(nr_dem):
            for iute in range(nr_ute):
                aux_2.append(col+iute)
            aux_2.append(col+nr_ute)
            aux.append(col+nr_ute+1)
            col += nr_ute+2
        aux.append(len(x)-1)
        ub = np.delete(ub, aux)
        ID_ub = np.eye(len(x))
        ID_ub = np.delete(ID_ub, aux, 0)
        ID = np.eye(len(x))

        if nr_cortes != 0:
            AB = np.concatenate((A, B), axis=1)
            AB = np.around(AB, decimals=2)
            AB = np.unique(AB, axis=0)
            A = AB[:, :-1]
            B = AB[:, -1]

            A = np.concatenate((A, -ID), axis=0)
            A = np.concatenate((A, ID_ub), axis=0)
            B = np.concatenate((B, -lb), axis=0)
            B = np.concatenate((B, ub), axis=0)

        else:
            A = np.concatenate((-ID, ID_ub), axis=0)
            B = np.concatenate((-lb, ub), axis=0)

        # Modelagem do problema
        Aeq = matrix(Aeq)
        Beq = matrix(Beq)
        A = matrix(A)
        B = matrix(B)
        lb = matrix(lb)
        ub = matrix(ub)
        c = matrix(c)
        ID = matrix(ID)
        ID_ub = matrix(ID_ub)

        restricoes = list()
        fob = dot(c, x)
        restricoes.append(Aeq * x == Beq)
        # if nr_cortes != 0:
        restricoes.append(A * x <= B)
        # restricoes.append(ID * x >= lb)
        # for ivar in range(len(aux_2)):
        #     restricoes.append(x[aux_2[ivar]] <= ub[aux_2[ivar]])
        # restricoes.append(ID_ub * x <= ub)

        return fob, restricoes, Aeq, Beq, A, B, c

    def plot_convergencia(self):
        plt.figure()
        plt.plot(np.arange(1, self.iter+1), self.ZINF, marker='o', label='ZINF')
        plt.errorbar(np.arange(1, self.iter+1), self.ZSUP, 1.96*np.array(self.sigma), marker='o', label='ZSUP')
        plt.xlabel('Iteração')
        plt.ylabel('Custo [R$]')
        plt.legend()
        plt.show()

class pdde_SistIsolFCI(object):

    sist = None
    cf = None
    ZINF = list()
    ZSUP = list()
    iter = 0
    sigma = list()
    CI_serie = None

    def __init__(self, sistema):
        self.sist = sistema

    def run(self, nr_meses, aberturas, pesos_aber, afluencias, FCI):

        solvers.options['show_progress'] = False
        # solvers.options['glpk'] = dict(maxiters=500)
        solvers.options['glpk'] = dict(msg_lev='GLP_MSG_OFF')
        solver = 'default'

        # Parâmetros iniciais
        nr_serfor = afluencias.shape[1]
        nr_aber = aberturas.shape[1]
        nr_sist = self.sist.conf_uh[0].Sist
        nr_uhe = len(self.sist.conf_uh)
        nr_ute = len(self.sist.conf_ut)

        # Incializa matrizes da otimizacao
        Aeq = np.zeros((nr_uhe + 1, 3 * nr_uhe + 3))
        Beq = np.zeros((nr_uhe + 1, 1))

        BH = np.zeros((nr_uhe, 3 * nr_uhe + 3))
        jusantes = [i.Jusante for i in self.sist.conf_uh]
        for iusi in range(nr_uhe):
            BH[iusi][iusi * 3] = 1
            BH[iusi][iusi * 3 + 1] = 2.592
            BH[iusi][iusi * 3 + 2] = 2.592
            codigo = self.sist.conf_uh[iusi].Codigo
            try:
                lista = [i for i, x in enumerate(jusantes) if x == codigo]
                for jusi in lista:
                    BH[iusi][jusi * 3 + 1] = -2.592
                    BH[iusi][jusi * 3 + 2] = -2.592
            except ValueError:
                pass

        EH = np.zeros((1, 3 * nr_uhe + 3))
        for iusi in range(nr_uhe):
            EH[0][iusi * 3 + 1] = self.sist.conf_uh[iusi].Ro65[0][0]
        EH[0, 3 * nr_uhe] = -1

        # Insere BH e EH na Aeq
        Aeq[0:nr_uhe, :] = BH
        Aeq[-1, :] = EH

        # Inicializa os volumes das usinas
        VIni = np.zeros((nr_uhe, nr_serfor, nr_meses + 1))
        for iusi in range(nr_uhe):
            VIni[iusi, :, 0] = self.sist.conf_uh[iusi].VolMin + (1 / 100) * self.sist.conf_uh[iusi].VolIni * self.sist.conf_uh[iusi].VolUtil

        self.cf = list()
        for imes in range(nr_meses):
            self.cf.append(fcf(imes))

        # Processo iterativo
        iter_LP = 1
        self.iter = 0
        erro = 9e15
        tol = 0.1
        tol_IC = 0.1
        tol_perc = 0.1
        estab = 9e15
        while ((abs(erro) > tol_IC) or (abs(estab) > tol_perc)) and (abs(erro) > tol_perc):   # && (abs(estab) > tol) %&& (abs(erro) > tol_IC)

            t = timer()

            # FORWARD
            ZINF = 0.
            ZSUP = 0.

            CustoImed = np.zeros((nr_meses, nr_serfor))
            CustoFut = np.zeros((nr_meses, nr_serfor))
            CustoTot = np.zeros((nr_meses, nr_serfor))
            for imes in range(nr_meses):

                # Cria as matrizes A e B contendo os cortes da FCI
                A_fci = np.zeros((FCI.Cortes_FCI[imes].inc.size, 3 * nr_uhe + 3))
                B_fci = np.zeros((FCI.Cortes_FCI[imes].inc.size, 1))
                A_fci[:, 3 * nr_uhe] = FCI.Cortes_FCI[imes].inc
                A_fci[:, 3 * nr_uhe + 1] = -1
                B_fci[:, 0] = -FCI.Cortes_FCI[imes].indepe

                if imes < nr_meses-1:
                    [A, B, nr_cortes] = self.cf[imes].calcAB(Aeq)
                else:
                    A = np.array([]).reshape(0, 3 * nr_uhe + 3)
                    B = np.array([]).reshape(0, 1)
                    nr_cortes = 0

                # Acrescente os cortes da FCI
                A = np.concatenate((A, A_fci), axis=0)
                B = np.concatenate((B, B_fci), axis=0)

                # Determina a canalização das variáveis
                [c, lb, ub] = self.canalizacao(nr_uhe, FCI, imes)

                for iser in range(nr_serfor):

                    # Preenche Beq
                    for iusi in range(nr_uhe):
                        Beq[iusi, 0] = VIni[iusi, iser, imes] + 2.592 * afluencias[iusi, iser, imes]
                    Beq[-1, 0] = 0.

                    x = variable(3 * nr_uhe + 3)  # inicializa variaveis
                    [fob, restricoes, Aeq_matrix, Beq_matrix, A_matrix, B_matrix, c_matrix] = self.reuni(x, Aeq, Beq, A, B, c, lb, ub)

                    problema = op(fob, restricoes)
                    problema.solve('dense', solver)

                    if problema.status != 'optimal':
                        print('CONVERGÊNCIA NÃO ALCANÇADA: ', problema.status, '-> FORWARD')

                    # Atualiza armazenamento das hidrelétricas
                    for iusi in range(nr_uhe):
                        VIni[iusi, iser, imes+1] = problema.variables()[0].value[3*iusi]

                    CustoImed[imes, iser] = x.value[3*nr_uhe+1]
                    CustoFut[imes, iser] = x.value[-1]
                    CustoTot[imes, iser] = fob.value()[0] - 0.001 * sum(x.value[2:3*nr_uhe:3])

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
            for imes in range(nr_meses-1, 0, -1):

                # Cria as matrizes A e B contendo os cortes da FCI
                A_fci = np.zeros((FCI.Cortes_FCI[imes].inc.size, 3 * nr_uhe + 3))
                B_fci = np.zeros((FCI.Cortes_FCI[imes].inc.size, 1))
                A_fci[:, 3 * nr_uhe] = FCI.Cortes_FCI[imes].inc
                A_fci[:, 3 * nr_uhe + 1] = -1
                B_fci[:, 0] = -FCI.Cortes_FCI[imes].indepe

                if imes < nr_meses - 1:
                    [A, B, nr_cortes] = self.cf[imes].calcAB(Aeq)
                else:
                    A = np.array([]).reshape(0, 3 * nr_uhe + 3)
                    B = np.array([]).reshape(0, 1)
                    nr_cortes = 0

                # Acrescente os cortes da FCI
                A = np.concatenate((A, A_fci), axis=0)
                B = np.concatenate((B, B_fci), axis=0)

                # Determina a canalização das variáveis
                [c, lb, ub] = self.canalizacao(nr_uhe, FCI, imes)

                for iser in range(nr_serfor):

                    LAMBDA = np.zeros((nr_aber, nr_uhe))
                    CUSTO = np.zeros((nr_aber, 1))
                    for iaber in range(nr_aber):

                        # Preenche Beq
                        for iusi in range(nr_uhe):
                            Beq[iusi, 0] = VIni[iusi, iser, imes] + 2.592 * aberturas[iusi, iaber, imes]
                        Beq[-1, 0] = 0.

                        x = variable(3 * nr_uhe + 3)  # inicializa variaveis
                        [fob, restricoes, Aeq_matrix, Beq_matrix, A_matrix, B_matrix, c_matrix] = self.reuni(x, Aeq, Beq, A, B, c, lb, ub)

                        problema = op(fob, restricoes)
                        problema.solve('dense', solver)

                        if problema.status != 'optimal':
                            print('CONVERGÊNCIA NÃO ALCANÇADA: ', problema.status, '-> BACKWARD')

                        CUSTO[iaber, 0] = fob.value()[0] - 0.001 * sum(x.value[2:3 * nr_uhe:3])
                        for iusi in range(nr_uhe):
                            if restricoes[0].multiplier.value[iusi] > 0.:
                                LAMBDA[iaber, iusi] = -restricoes[0].multiplier.value[iusi]
                            else:
                                LAMBDA[iaber, iusi] = 0.

                    lambda_mean = np.matmul(pesos_aber[imes, :], LAMBDA)
                    custo_mean = np.matmul(pesos_aber[imes, :], CUSTO)

                    vol_ini = np.zeros((nr_uhe, 1))
                    for iusi in range(nr_uhe):
                        vol_ini[iusi, 0] = VIni[iusi, iser, imes]

                    # Função que gera o corte da FCF
                    self.cf[imes-1].geraFCF(vol_ini, custo_mean, lambda_mean)

            print('Iteração:', self.iter + 1,
                  ' - ZINF:', self.ZINF[self.iter],
                  ' - ZSUP:', self.ZSUP[self.iter],
                  ' - Erro:', erro,
                  ' - Tempo:', round(timer() - t, 2), 'seg')

            self.iter += 1

    def canalizacao(self, nr_uhe, FCI, imes):
        c = np.zeros((1, 3 * nr_uhe + 3))
        lb = np.zeros((1, 3 * nr_uhe + 3))
        ub = np.zeros((1, 3 * nr_uhe + 3))

        # Dados das Hidrelétricas
        col = 0
        for iusi in range(nr_uhe):
            c[0, col] = 0.
            c[0, col + 1] = 0.
            c[0, col + 2] = 0.001
            lb[0, col] = self.sist.conf_uh[iusi].VolMin
            lb[0, col + 1] = 0.
            lb[0, col + 2] = 0.
            ub[0, col] = self.sist.conf_uh[iusi].VolMax
            ub[0, col + 1] = self.sist.conf_uh[iusi].Engolimento
            ub[0, col + 2] = 9e15  #infinito
            col += 3

        # Energia Hidreletrica Total
        col = 3 * nr_uhe
        c[0, col] = 0.
        lb[0, col] = np.min(FCI.Pontos_FCI[imes].ghidr)
        ub[0, col] = np.max(FCI.Pontos_FCI[imes].ghidr)

        # Beta e Alfa
        c[0, 3 * nr_uhe + 1:] = 1.
        lb[0, 3 * nr_uhe + 1:] = [np.min(FCI.Pontos_FCI[imes].beta), 0]
        ub[0, 3 * nr_uhe + 1:] = [np.max(FCI.Pontos_FCI[imes].beta), 9e15]

        lb = lb.transpose()
        ub = ub.transpose()

        return c, lb, ub

    def reuni(self, x, Aeq, Beq, A, B, c, lb, ub):

        Aeq = matrix(Aeq)
        Beq = matrix(Beq)
        A = matrix(A)
        B = matrix(B)
        lb = matrix(lb)
        ub = matrix(ub)
        c = matrix(c)

        restricoes = list()
        ID = matrix(np.eye(len(x)))
        fob = c * x
        restricoes.append(Aeq * x == Beq)
        restricoes.append(A * x <= B)
        restricoes.append(ID * x >= lb)
        restricoes.append(ID * x <= ub)

        return fob, restricoes, Aeq, Beq, A, B, c

    def plot_convergencia(self):
        plt.figure()
        plt.plot(np.arange(1, self.iter + 1), self.ZINF, marker='o', label='ZINF')
        plt.errorbar(np.arange(1, self.iter + 1), self.ZSUP, 1.96*np.array(self.sigma), marker='o', label='ZSUP')
        plt.xlabel('Iteração')
        plt.ylabel('Custo [R$]')
        plt.legend()
        plt.show()