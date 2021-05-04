from Modules.EnergyPlanning.otimo_propriedades import *
import numpy as np


class otimo(object):

    Sist = None
    GTerm = None
    Interc = None
    custototal = None
    custoimediato = None
    custofuturo = None
    nr_estagios = None
    nr_cenarios_ena = None

    def __init__(self, nr_estagios, nr_cenarios_ena):
        self.nr_estagios = nr_estagios
        self.nr_cenarios_ena = nr_cenarios_ena

    def save(self, sistema, VARIAVEIS, CUSTO, LAMBDA, CMO, ALFA, DEMLIQ, submarket_index, afluencias):

        submercados_name = {1: 'Sudeste', 2: 'Sul', 3: 'Nordeste', 4: 'Norte', 11: 'Fictício'}
        submarket_codes = [sistema.submercado[i].Codigo for i in submarket_index]
        submercados = [x for x in sistema.submercado if x.Codigo in submarket_codes]
        nr_sist = len(submercados)
        termicas = [x for x in sistema.conf_ut if x.Sist in submarket_codes]
        nr_ute = len(termicas)
        intercambios = [x for x in sistema.intercambio if x.De in submarket_codes and x.Para in submarket_codes or
                        (x.De in submarket_codes and x.Para == 11) or (x.Para in submarket_codes and x.De == 11)]
        mes_ini = sistema.dger.MesInicioEstudo

        # Salva dados de todas as usinas termelétricas
        self.GTerm = list()
        cont = 0
        for iute, termica in enumerate(termicas):
            self.GTerm.append(otimoterm())
            self.GTerm[cont].Nome = termica.Nome
            self.GTerm[cont].GT = VARIAVEIS[:, 5 * nr_sist + iute, :]
            self.GTerm[cont].GT_MAX = termica.GTMAX
            self.GTerm[cont].Sistema = submercados_name[termica.Sist]

            cont += 1

        # Salva dados do sistema: GHidr, GTerm, Deficit, ....
        self.Sist = list()
        for isist in range(nr_sist):
            self.Sist.append(otimosist(submercados[isist].Nome))
            self.Sist[isist].EArm = VARIAVEIS[:, isist, :]
            self.Sist[isist].EVert = VARIAVEIS[:, nr_sist+isist, :]
            self.Sist[isist].GHidr = VARIAVEIS[:, 2*nr_sist+isist, :] + (1 - submercados[isist].FatorSeparacao)*afluencias[isist, :, :]
            self.Sist[isist].Exc = VARIAVEIS[:, 3*nr_sist+isist, :]
            self.Sist[isist].Deficit = VARIAVEIS[:, 4*nr_sist+isist, :]
            self.Sist[isist].ValorAgua = LAMBDA[:, isist, :]
            self.Sist[isist].CMO = CMO[:, isist, :]
            self.Sist[isist].DemLiq = DEMLIQ[:, isist, :]

            # GTMin = np.zeros((self.nr_cenarios_ena, self.nr_estagios))
            # for imes in range(self.nr_estagios):
            #     ano = int((mes_ini + imes) / 12)
            #     mes = (mes_ini + imes - 1) % 12
            #     GTMin[:, imes] = submercados[isist].GTMIN[ano, mes]

            GTerm = np.zeros((self.nr_cenarios_ena, self.nr_estagios))
            for term in self.GTerm:
                if term.Sistema == submercados_name[submarket_codes[isist]]:
                    GTerm += term.GT

            self.Sist[isist].GTerm = GTerm  # + GTMin

        # Salva dados de Intercâmbio
        self.Interc = list()
        for interc, intercambio in enumerate(intercambios):
            self.Interc.append(otimointerc())
            self.Interc[interc].De = intercambio.De
            self.Interc[interc].Para = intercambio.Para
            self.Interc[interc].INT = VARIAVEIS[:, 5*nr_sist+nr_ute+interc, :]
            self.Interc[interc].INT_MAX = intercambio.LimiteMaximo[:, :].reshape((1, intercambio.LimiteMaximo.size))[0, mes_ini-1:mes_ini-1+self.nr_estagios]
            self.Interc[interc].Nome = '%s->%s' % (submercados_name[intercambio.De], submercados_name[intercambio.Para])

        self.custototal = CUSTO + ALFA
        self.custofuturo = ALFA
        self.custoimediato = CUSTO

        return