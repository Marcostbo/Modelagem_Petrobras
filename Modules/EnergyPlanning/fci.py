# coding utf-8

import numpy as np
import itertools
from mosek import *
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class pontos_struct(object):

    beta = None
    ghidr = None

    def __init__(self, estagio):
        self.beta = []
        self.ghidr = []
        self.estagio = estagio


class cortes_struct(object):

    inc = None
    indepe = None

    def __init__(self, estagio):
        self.inc = []
        self.indepe = []
        self.estagio = estagio


class CustoImediato_Isol(object):

    sist = None
    Cortes_FCI = None
    Pontos_FCI = None

    def __init__(self, sistema, nr_months, eolicas, pesos_eol, submarket_index, nr_process: int):
        self.sist = sistema
        self.nr_months = nr_months
        self.eolicas = eolicas
        self.pesos_eol = pesos_eol
        self.submarket_index = submarket_index
        self.nr_process = nr_process

    def run(self):

        # Parâmetros iniciais
        nr_cen_eol = self.eolicas.shape[1]
        submarket_codes = [self.sist.submercado[i].Codigo for i in self.submarket_index]
        nr_ute = len([x for x in self.sist.conf_ut if x.Sist in submarket_codes])
        termicas = [x for x in self.sist.conf_ut if x.Sist in submarket_codes]
        submercado = [x for x in self.sist.submercado if x.Codigo in submarket_codes][0]
        mes_ini = self.sist.dger.MesInicioEstudo

        self.Cortes_FCI = list()
        self.Pontos_FCI = list()
        for imes in range(self.nr_months):
            self.Cortes_FCI.append(cortes_struct(imes))
            self.Pontos_FCI.append(pontos_struct(imes))

        # Algoritmo da FCI para cada mês
        valor1 = np.zeros((1, nr_cen_eol))
        for imes in range(self.nr_months):

            # Definindo iano e imes
            ano = int((mes_ini + imes) / 12)
            mes = (mes_ini + imes - 1) % 12

            # Dados de termeletricas e hidreletricas
            custo_term = np.zeros((1, nr_ute))
            gterm_max = np.zeros((1, nr_ute))
            for iute, termica in enumerate(termicas):
                custo_term[0, iute] = termica.Custo[0]
                gterm_max[0, iute] = termica.GTMAX[ano, mes] - termica.GTMin[ano, mes]
            EA_vetor = np.array([submercado.EAMAX[ano, mes] ** 2, submercado.EAMAX[ano, mes], 1])
            ghidr_max = np.dot(submercado.ParamGHMAX, EA_vetor)
            custo_term_ord = np.sort(custo_term)
            ind = np.argsort(custo_term)
            gterm_max_ord = np.zeros((1, nr_ute))
            for iute in range(nr_ute):
                gterm_max_ord[0, iute] = gterm_max[0, ind[0, iute]]

            # Dados deficit
            custo_def = submercado.CustoDeficit[0]

            # Energia hidrelétrica máxima
            valor1[0, :] = np.minimum(submercado.Mercado[ano, mes] - submercado.NaoSimuladas[ano, mes] - self.eolicas[0, :, imes], ghidr_max)
            energ_max = np.matmul(valor1, np.transpose(self.pesos_eol))

            # Energia hidrelétrica mínima e intermediárias
            lista = np.zeros((1, nr_ute))
            VALOR = np.zeros((nr_ute, nr_cen_eol))
            for iute in range(nr_ute):
                valor2 = np.maximum(submercado.Mercado[ano, mes] - submercado.NaoSimuladas[ano, mes] - self.eolicas[0, :, imes] - np.sum(gterm_max_ord[0, :iute + 1]), 0)
                valor3 = np.minimum(valor2, ghidr_max)
                lista[0, iute] = np.matmul(valor3, np.transpose(self.pesos_eol))
                VALOR[iute, :] = valor3

            # Energia total e por demanda
            energ1 = np.concatenate((energ_max, lista, np.zeros((1, 1))), axis=1)
            energ2 = np.concatenate((valor1, VALOR, np.zeros((1, nr_cen_eol))), axis=0)  # + np.zeros((1, nr_dem))

            # Aplicacao de Baleriaux em cada ponto
            BETA = np.zeros((nr_ute+2, 1))
            for k in range(nr_ute+2):

                # Metodo de Baleriaux
                energy_not_supply = np.zeros((1, nr_ute+1))
                beta_dem = np.zeros((nr_cen_eol, 1))
                gterm_dem = np.zeros((nr_cen_eol, nr_ute))
                load_energy = submercado.Mercado[ano, mes] - submercado.NaoSimuladas[ano, mes] - self.eolicas[0, :, imes] - energ2[k, :]
                deficit = np.zeros((1, nr_cen_eol))
                for idem in range(nr_cen_eol):

                    # step 2: Calculo dos suprimentos de energia
                    energy_not_supply[0, 0] = load_energy[idem]
                    for ii in range(nr_ute):
                        soma_gmax = np.sum(gterm_max_ord[0, :ii + 1])
                        energy_not_supply[0, ii + 1] = np.maximum(load_energy[idem] - soma_gmax, 0)

                    # step 3: Geracao termeletrica otima
                    for ii in range(nr_ute):
                        gterm_dem[idem, ii] = energy_not_supply[0, ii] - energy_not_supply[0, ii + 1]

                    # step 4: Deficit
                    deficit[0, idem] = energy_not_supply[0, -1]

                    # step 5: Custo por demanda
                    beta_dem[idem, 0] = np.matmul(custo_term_ord, gterm_dem[idem, :]) + custo_def * deficit[0, idem]

                BETA[k, 0] = np.matmul(self.pesos_eol, beta_dem)

            # Alocar pontos por estagio
            AUX1 = np.concatenate((np.round(BETA, 4), np.round(np.transpose(energ1), 4)), axis=1)
            AUX2 = np.unique(AUX1, axis=0)

            self.Pontos_FCI[imes].beta = np.transpose(AUX2[:, 0])
            self.Pontos_FCI[imes].ghidr = np.transpose(AUX2[:, 1])

            # Determinacao das inclinacoes e termos independentes dos cortes gerados para a FCI
            inc_fci = np.zeros((1, AUX2.shape[0] - 1))
            indepe_fci = np.zeros((1, AUX2.shape[0] - 1))
            for k in range(AUX2.shape[0] - 1):
                inc_fci[0, k] = (self.Pontos_FCI[imes].beta[k+1] - self.Pontos_FCI[imes].beta[k]) / (self.Pontos_FCI[imes].ghidr[k+1] - self.Pontos_FCI[imes].ghidr[k])
                indepe_fci[0, k] = self.Pontos_FCI[imes].beta[k] - inc_fci[0, k] * self.Pontos_FCI[imes].ghidr[k]

            AUX1 = np.concatenate((np.transpose(inc_fci), np.transpose(indepe_fci)), axis=1)
            AUX2 = np.unique(AUX1, axis=0)

            self.Cortes_FCI[imes].inc = np.transpose(AUX2[:, 0])
            self.Cortes_FCI[imes].indepe = np.transpose(AUX2[:, 1])

            print('teste')


class CustoImediato_Multi(object):

    sist = None
    Cortes_FCI = None
    Pontos_FCI = None
    areas = None
    ghidrs_max = None
    GHIDR = None
    BETA = None

    def __init__(self, sistema):
        self.sist = sistema

    def run(self, nr_meses, demandas, pesos_dem, nr_disc):

        # Parâmetros iniciais
        nr_dem = demandas.shape[1]
        self.GHIDR = list()
        self.BETA = list()

        self.Cortes_FCI = list()
        self.Pontos_FCI = list()
        for imes in range(nr_meses):
            self.Cortes_FCI.append(cortes_struct(imes))
            self.Pontos_FCI.append(pontos_struct(imes))

        # Inicialização dos dados por área
        hidr_sist = [isist.Sist for i, isist in enumerate(self.sist.conf_uh)]
        term_sist = [isist.Sist for i, isist in enumerate(self.sist.conf_ut)]
        custo_term_full = list()
        gterm_max_full = list()
        ghidr_max_full = list()
        ind_term_full = list()
        ind_hidr_full = list()
        for iar in range(4):
            ind1 = [index for index, value in enumerate(term_sist) if value == iar+1]
            custo_term_full.append([self.sist.conf_ut[ind1[x]].Custo[0] for x in range(len(ind1))])
            gterm_max_full.append([self.sist.conf_ut[ind1[x]].Potencia for x in range(len(ind1))])
            ind_term_full.append(ind1)

            ind2 = [index for index, value in enumerate(hidr_sist) if value == iar+1]
            energ = list()
            for i in range(len(ind2)):
                energ.append(self.sist.conf_uh[ind2[i]].Engolimento * self.sist.conf_uh[ind2[i]].Ro65[0][0])
            ghidr_max_full.append(energ)
            ind_hidr_full.append(ind2)

        # Elimina áreas que não possuem hdrelétricas/termelétricas
        gterm_max = list()
        ghidr_max = list()
        custo_term = list()
        self.areas = list()
        deficit = list()
        for iar in range(4):
            if custo_term_full[iar] != [] and ghidr_max_full[iar] != []:
                gterm_max.append(gterm_max_full[iar])
                ghidr_max.append(ghidr_max_full[iar])
                custo_term.append(custo_term_full[iar])
                deficit.append(self.sist.submercado[iar].CustoDeficit[0])
                self.areas.append(iar+1)
        nr_areas = len(self.areas)

        #TODO: Algoritmo
        ano = 0
        mes = self.sist.mes_ini - 1
        self.ghidrs_max = list()
        Beta = np.zeros((np.product(nr_disc[:nr_areas]), nr_meses))
        for imes in range(nr_meses):

            # Intercâmbio
            interc = np.zeros((4, 4))
            comb_areas = np.array(list(itertools.combinations(self.areas, 2)))
            DE = [iinterc.De for i, iinterc in enumerate(self.sist.intercambio)]
            PARA = [iinterc.Para for i, iinterc in enumerate(self.sist.intercambio)]
            for k in range(comb_areas.shape[0]):
                ind1 = [index for index in range(len(DE)) if DE[index] == (comb_areas[k, 0]) and PARA[index] == (comb_areas[k, 1])]
                ind2 = [index for index in range(len(DE)) if DE[index] == (comb_areas[k, 1]) and PARA[index] == (comb_areas[k, 0])]
                if ind1 != []:
                    interc[comb_areas[k, 0]-1, comb_areas[k, 1]-1] = self.sist.intercambio[ind1[0]].LimiteMaximo[ano, mes]
                    interc[comb_areas[k, 1]-1, comb_areas[k, 0]-1] = self.sist.intercambio[ind2[0]].LimiteMaximo[ano, mes]

            for k in range(3, -1, -1):
                if (k+1) not in self.areas:
                    interc = np.delete(interc, k, axis=0)
                    interc = np.delete(interc, k, axis=1)

            mes += 1
            if mes == 12:
                mes = 0
                ano += 1

            ghidrs_max = np.zeros(len(ghidr_max))
            for k in range(len(ghidr_max)):
                ghidrs_max[k] = np.sum(ghidr_max[k])
            ghidrs_max = [ghidrs_max[k] for k in range(len(ghidrs_max)) if ghidrs_max[k] != 0]
            nr_hidrs = len(ghidrs_max)

            custos_term = [custo for area in custo_term for custo in area]
            gterms_max = [gmax for area in gterm_max for gmax in area]
            nr_terms = len(custos_term)
            nr_ar_terms = len(custo_term)
            custos = list()
            for i in range(nr_terms):
                custos.append(custos_term[i])
            for j in range(len(deficit)):
                custos.append(deficit[j])

            # Determina matrizes para o solver de otimização
            lb_pad = np.zeros(nr_hidrs+nr_terms+nr_areas+nr_areas*(nr_areas-1))
            ub_pad = np.zeros(nr_hidrs + nr_terms + nr_areas + nr_areas * (nr_areas - 1))
            ub_pad[0:len(ghidrs_max)] = ghidrs_max
            ub_pad[len(ghidrs_max):len(ghidrs_max)+nr_terms] = gterms_max
            ub_pad[len(ghidrs_max)+nr_terms:len(ghidrs_max) + nr_terms+nr_areas] = None

            Aeq_pad = np.zeros((nr_areas, nr_hidrs+nr_terms+nr_areas+nr_areas*(nr_areas-1)))
            col = 0
            # Hidrelétricas
            for iar in range(nr_hidrs):
                Aeq_pad[iar, col] = 1
                col += 1
            # Termelétricas
            for iar in range(nr_ar_terms):
                for icol in range(nr_terms):
                    if icol in ind_term_full[iar]:
                        Aeq_pad[iar, col] = 1
                        col += 1
            # Déficit
            for iar in range(nr_areas):
                Aeq_pad[iar, col] = 1
                col += 1
            # Intercâmbio
            comb = np.array(list(itertools.combinations(np.arange(0, nr_areas, 1), 2)))
            for k in range(comb.shape[0]):
                Aeq_pad[comb[k, :], col:col+2] = [[-1, 1], [1, -1]]
                ub_pad[col:col+2] = [interc[comb[k, 0], comb[k, 1]], interc[comb[k, 1], comb[k, 0]]]
                col += 2

            # Determina as matrizes para o solver
            Aeq = np.zeros((nr_dem*nr_areas, nr_dem*(nr_hidrs+nr_terms+nr_areas+nr_areas*(nr_areas-1))))
            c = np.zeros(nr_dem*(nr_hidrs+nr_terms+nr_areas+nr_areas*(nr_areas-1)))
            lb = np.zeros(nr_dem * (nr_hidrs + nr_terms + nr_areas + nr_areas * (nr_areas - 1)))
            ub = np.zeros(nr_dem * (nr_hidrs + nr_terms + nr_areas + nr_areas * (nr_areas - 1)))
            Beq = np.zeros(nr_dem*nr_areas)
            lin = 0
            col = 0
            for idem in range(nr_dem):
                carga = np.zeros(nr_areas)
                for iar in range(nr_areas):
                    carga[iar] = demandas[iar, idem, imes]
                Aeq[lin:lin+nr_areas, col:col+nr_hidrs+nr_terms+nr_areas+nr_areas*(nr_areas-1)] = Aeq_pad
                fob_pad = list()
                for ihidr in range(nr_hidrs):
                    fob_pad.append(0)
                for iterm in range(len(custos_term)):
                    fob_pad.append(pesos_dem[0, idem]*custos_term[iterm])
                for idef in range(nr_areas):
                    fob_pad.append(pesos_dem[0, idem]*deficit[idef])
                for iinterc in range(nr_areas*(nr_areas-1)):
                    fob_pad.append(0)
                c[col:col+nr_hidrs+nr_terms+nr_areas+nr_areas*(nr_areas-1)] = fob_pad
                lb[col:col + nr_hidrs + nr_terms + nr_areas + nr_areas * (nr_areas - 1)] = lb_pad
                ub[col:col + nr_hidrs + nr_terms + nr_areas + nr_areas * (nr_areas - 1)] = ub_pad
                Beq[lin:lin+nr_areas] = carga

                lin += nr_areas
                col += nr_hidrs+nr_terms+nr_areas+nr_areas*(nr_areas-1)

            # tempo = timer()
            [X, DUAL, FVAL] = self.runMosek(Aeq, Beq, c, lb, ub)
            # print('Tempo-LINPROG:', round(timer() - tempo, 3), 'seg')

            # Discretização dos valores de geração hidrelétrica a serem utilizadas no algoritmo
            N = nr_disc
            Aeq_new = np.zeros((nr_dem * nr_areas + nr_areas, nr_dem * (nr_hidrs + nr_terms + nr_areas + nr_areas * (nr_areas - 1))))
            energ_max = list()
            energ_disc = list()
            for k in range(nr_areas):
                Aeq_new[:nr_dem*nr_areas, :] = Aeq
                Aeq_new[nr_dem*nr_areas+k, k:-1:nr_hidrs+nr_terms+nr_areas+nr_areas*(nr_areas-1)] = pesos_dem
                energ_max.append(np.matmul(pesos_dem, X[k:-1:nr_hidrs+nr_terms+nr_areas+nr_areas*(nr_areas-1)])[0])
                energ_disc.append(np.linspace(0, energ_max[k], N[k]))
            self.ghidrs_max.append(energ_max)

            Energ = list()
            if nr_areas == 2:
                for k in range(N[0]):
                    for m in range(N[1]):
                        Energ.append([energ_disc[0][k], energ_disc[1][m]])
            elif nr_areas == 3:
                for k in range(N[0]):
                    for m in range(N[1]):
                        for n in range(N[2]):
                            Energ.append([energ_disc[0][k], energ_disc[1][m], energ_disc[2][n]])
            elif nr_areas == 4:
                for k in range(N[0]):
                    for m in range(N[1]):
                        for n in range(N[2]):
                            for r in range(N[3]):
                                Energ.append([energ_disc[0][k], energ_disc[1][m], energ_disc[2][n], energ_disc[3][r]])

            # Algoritmo
            Beq_new = np.zeros(nr_dem * nr_areas + nr_areas)
            Beq_new[:nr_dem * nr_areas] = Beq
            for k in range(len(Energ)):

                for r in range(nr_areas):
                    Beq_new[nr_dem * nr_areas + r] = Energ[k][r]

                # tempo = timer()
                [X, DUAL, FVAL] = self.runMosek(Aeq_new, Beq_new, c, lb, ub)
                # print('Tempo-LINPROG:', round(timer() - tempo, 3), 'seg')

                Beta[k][imes] = FVAL

            self.GHIDR.append(Energ)
            self.BETA.append([Beta[x][imes] for x in range(Beta.shape[0])])

         # TODO: Convex-Hull
        for imes in range(nr_meses):

            P = np.concatenate((np.array(self.GHIDR[imes]), np.array([[Beta[x][imes]] for x in range(Beta.shape[0])])), axis=1)
            P = np.unique(P, axis=0)
            hull = ConvexHull(P)

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(P[:, 0], P[:, 1], P[:, 2])
            # ax.scatter(P[hull.vertices, 0], P[hull.vertices, 1], P[hull.vertices, 2], 'ro')
            # ax.set_xlabel('gh1')
            # ax.set_ylabel('gh2')
            # ax.set_zlabel('Beta')
            #
            # plt.show()

            coeficientes = np.zeros((hull.equations.shape[0], nr_areas+1))
            for iplan in range(coeficientes.shape[0]):
                for iar in range(nr_areas):
                    coeficientes[iplan, iar] = -hull.equations[iplan, iar] / hull.equations[iplan, nr_areas]
                coeficientes[iplan, -1] = -hull.equations[iplan, -1] / hull.equations[iplan, nr_areas]

            coeficientes = coeficientes[~np.isnan(coeficientes).any(axis=1)]  # elimina linhas com Nan
            coeficientes = coeficientes[~np.isinf(coeficientes).any(axis=1)]  # ekimina linhas com Inf

            fig = plt.figure()
            ax = fig.gca(projection='3d')

            # Make data.
            X = np.arange(-5, 5, 0.25)
            Y = np.arange(-5, 5, 0.25)
            X, Y = np.meshgrid(X, Y)
            R = np.sqrt(X ** 2 + Y ** 2)
            Z = np.sin(R)

            # Plot the surface.
            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)

           # Alocar pontos por estagio
            AUX1 = np.concatenate((np.round(BETA, 4), np.round(np.transpose(energ1), 4)), axis=1)
            AUX2 = np.unique(AUX1, axis=0)

            self.Pontos_FCI[imes].beta = np.transpose(AUX2[:, 0])
            self.Pontos_FCI[imes].ghidr = np.transpose(AUX2[:, 1])

            # Determinacao das inclinacoes e termos independentes dos cortes gerados para a FCI
            inc_fci = np.zeros((1, AUX2.shape[0] - 1))
            indepe_fci = np.zeros((1, AUX2.shape[0] - 1))
            for k in range(AUX2.shape[0] - 1):
                inc_fci[0, k] = (self.Pontos_FCI[imes].beta[k + 1] - self.Pontos_FCI[imes].beta[k]) / (
                        self.Pontos_FCI[imes].ghidr[k + 1] - self.Pontos_FCI[imes].ghidr[k])
                indepe_fci[0, k] = self.Pontos_FCI[imes].beta[k] - inc_fci[0, k] * \
                                   self.Pontos_FCI[imes].ghidr[k]

            AUX1 = np.concatenate((np.transpose(inc_fci), np.transpose(indepe_fci)), axis=1)
            AUX2 = np.unique(AUX1, axis=0)

            self.Cortes_FCI[imes].inc = np.transpose(AUX2[:, 0])
            self.Cortes_FCI[imes].indepe = np.transpose(AUX2[:, 1])


    def runMosek(self, Aeq, Beq, c, lb, ub):

        inf = 0.0

        bkx = [0.] * len(c)
        blx = [0.] * len(c)
        bux = [0.] * len(c)
        for ivar in range(len(c)):
            if lb[ivar] < ub[ivar]:
                bkx[ivar] = boundkey.ra
                blx[ivar] = lb[ivar]
                bux[ivar] = ub[ivar]
            elif lb[ivar] == ub[ivar]:
                bkx[ivar] = boundkey.fx
                blx[ivar] = lb[ivar]
                bux[ivar] = ub[ivar]
            else:
                bkx[ivar] = boundkey.lo
                blx[ivar] = lb[ivar]
                bux[ivar] = +inf
        asub = list()
        aval = list()
        for icol in range(len(c)):
            aux1 = list()
            aux2 = list()
            for ilin in range(Aeq.shape[0]):
                if Aeq[ilin, icol] != 0.:
                    aux1.append(ilin)
                    aux2.append(Aeq[ilin, icol])
            asub.append(aux1)
            aval.append(aux2)

        asub = [x for x in asub if x != []]
        aval = [x for x in aval if x != []]

        bkc = [0.] * (Aeq.shape[0])
        blc = [0.] * (Aeq.shape[0])
        buc = [0.] * (Aeq.shape[0])
        for ilin in range(Aeq.shape[0]):
            bkc[ilin] = boundkey.fx
            blc[ilin] = Beq[ilin]
            buc[ilin] = Beq[ilin]

        numvar = len(bkx)
        numcon = len(bkc)

        task = Env().Task()
        task.appendvars(numvar)
        task.appendcons(numcon)

        for i in range(numvar):
            task.putcj(i, c[i])
            task.putvarbound(i, bkx[i], blx[i], bux[i])
        for i in range(len(asub)):
            task.putacol(i, asub[i], aval[i])
        for j in range(numcon):
            task.putconbound(j, bkc[j], blc[j], buc[j])

        task.putobjsense(objsense.minimize)

        task.optimize()

        EXITFLAG = task.getsolsta(soltype.bas)

        if EXITFLAG == solsta.optimal:

            X = [0.] * numvar
            DUAL = [0.] * numcon
            task.getxx(soltype.bas, X)
            task.gety(soltype.bas, DUAL)
            FVAL = task.getprimalobj(soltype.bas)

        else:
            print('CONVERGÊNCIA NÃO ALCANÇADA: ')

        return X, DUAL, FVAL