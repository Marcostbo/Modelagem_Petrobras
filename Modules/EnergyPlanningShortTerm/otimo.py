from Modules.EnergyPlanningShortTerm.otimo_propriedades import *
import numpy as np


class otimo(object):

    Hidreletrica = None
    Termeletrica = None
    Submercado = None
    Intercambio = None
    CustoTotal = None
    CustoImediato = None
    CustoFuturo = None

    def save(self, sistema, VARIAVEIS, CUSTO, LAMBDA, CMO, ALFA, nr_var_hidr, nr_var_term, wind_energy):

        nr_hidr = len(sistema.hidr)
        nr_unid_term = sum([x.NumUnidades for x in sistema.term])
        nr_interc = len(sistema.interc)
        nr_cont_imp = sum([len(s.ContratoImportacaoT) if s.ContratoImportacaoT else 0 for s in sistema.sist])
        nr_cont_exp = sum([len(s.ContratoExportacaoT) if s.ContratoExportacaoT else 0 for s in sistema.sist])

        GerHidrSist = np.zeros((4, len(ALFA)))
        GerTermSist = np.zeros((4, len(ALFA)))

        self.Hidreletrica = list()
        cont = 0
        # Salva dados das hidrelétricas
        for i, uhe in enumerate(sistema.hidr):
            self.Hidreletrica.append(otimohidr())
            self.Hidreletrica[cont].Nome = uhe.Nome
            self.Hidreletrica[cont].Submercado = uhe.Sist
            self.Hidreletrica[cont].VolArm = VARIAVEIS[nr_var_hidr*i, :]
            self.Hidreletrica[cont].Turb = VARIAVEIS[nr_var_hidr * i + 1, :]
            self.Hidreletrica[cont].Vert = VARIAVEIS[nr_var_hidr * i + 2, :]
            self.Hidreletrica[cont].Desv = VARIAVEIS[nr_var_hidr * i + 3, :]
            self.Hidreletrica[cont].Bomb = VARIAVEIS[nr_var_hidr * i + 4, :]
            self.Hidreletrica[cont].Evap = VARIAVEIS[nr_var_hidr * i + 5, :]
            self.Hidreletrica[cont].GHidr = VARIAVEIS[nr_var_hidr * i + 6, :]
            GerHidrSist[uhe.Sist - 1, :] += VARIAVEIS[nr_var_hidr * i + 6, :]
            self.Hidreletrica[cont].ValorAgua = LAMBDA[i, :]

            cont += 1

        # Salva dados de todas as usinas termelétricas
        self.Termeletrica = list()
        cont = 0
        for i, ute in enumerate(sistema.term):
            for j, unid in enumerate(ute.UnidadeGeradoraT):
                self.Termeletrica.append(otimoterm())
                self.Termeletrica[cont].Nome = ute.Nome
                self.Termeletrica[cont].Unidade = unid
                self.Termeletrica[cont].Submercado = ute.Sist
                self.Termeletrica[cont].GTerm = VARIAVEIS[nr_var_hidr*nr_hidr+cont, :]
                GerTermSist[ute.Sist - 1, :] += VARIAVEIS[nr_var_hidr*nr_hidr+cont, :]

                cont += 1

        # Salva dados de todos os submercados
        self.Submercado = list()
        cont = 0
        nr_imp_aux, nr_exp_aux = 0, 0
        for i, sub in enumerate(sistema.sist):
            if sub.Sigla != 'FC':
                n_imp = len(sub.ContratoImportacaoT)
                n_exp = len(sub.ContratoExportacaoT)
                self.Submercado.append(otimosist())
                self.Submercado[cont].Nome = sub.Nome
                self.Submercado[cont].ContImp = np.sum(VARIAVEIS[nr_var_hidr*nr_hidr+nr_var_term*nr_unid_term+nr_imp_aux:nr_var_hidr*nr_hidr+nr_var_term*nr_unid_term+nr_imp_aux+n_imp, :], axis=0)
                self.Submercado[cont].ContExp = np.sum(VARIAVEIS[nr_var_hidr*nr_hidr+nr_var_term*nr_unid_term+nr_imp_aux+n_imp:nr_var_hidr*nr_hidr+nr_var_term*nr_unid_term+nr_imp_aux+n_imp+n_exp, :], axis=0)
                self.Submercado[cont].Deficit = VARIAVEIS[nr_var_hidr*nr_hidr+nr_var_term*nr_unid_term+nr_cont_imp+nr_cont_exp+nr_interc+cont, :]
                self.Submercado[cont].Cmo = CMO[cont, :]
                self.Submercado[cont].Carga = sub.CargaT.loc[:, 'CARGA']
                if sub.Sigla == 'NE':
                    self.Submercado[cont].GerEolica = wind_energy
                self.Submercado[cont].GerHidr = GerHidrSist[cont, :]
                self.Submercado[cont].GerTerm = GerTermSist[cont, :]

                nr_imp_aux += n_imp
                nr_exp_aux += n_exp

                cont += 1

        # Salva dados de Intercâmbio
        self.Intercambio = list()
        cont = 0
        for interc, intercambio in enumerate(sistema.interc):
            self.Intercambio.append(otimointerc())
            self.Intercambio[interc].De = intercambio.De
            self.Intercambio[interc].Para = intercambio.Para
            self.Intercambio[interc].Interc = VARIAVEIS[nr_var_hidr*nr_hidr+nr_var_term*nr_unid_term+nr_cont_imp+nr_cont_exp+interc, :]
            self.Intercambio[interc].IntercMax = intercambio.LimiteT.loc[:, 'Limite']
            self.Intercambio[interc].Nome = intercambio.Sigla

            cont += 1

        self.CustoTotal = CUSTO + ALFA
        self.CustoFuturo = ALFA
        self.CustoImediato = CUSTO

        return