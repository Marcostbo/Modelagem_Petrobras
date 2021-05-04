import pandas as pd
import numpy as np
from numpy import array
# from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from timeit import default_timer as timer
from calendar import monthrange
from datetime import date, datetime, timedelta as delta
from Modules.ConvexHull.chull import ConvexHull


class hidr(object):

    # Dados de cadastro das usinas hidreletricas (presentes no HIDR.DAT)
    Nome = None             # Nome da UHE
    Posto = None            # Numero do Posto
    Codigo = None           # Codigo da UHE
    Bdh = None              # Desvio - Nao sei qual e esta informacao ??????
    Sist = None             # Submercado
    Empr = None             # Codigo da empresa
    Jusante = None          # Codigo de Jusante
    Desvio = None           # Desvio
    VolMin = None           # Volume Minimo
    VolMax = None           # Volume Maximo
    VolMinVert = None       # Volume Minimo para Vertimento
    VolMinDesv = None       # Volume Minimo para Desvio
    CotaMin = None          # Cota Minima
    CotaMax = None          # Cota Maxima
    CotaMed = None          # Cota Média
    PolCotaVol = None       # Polinomio Cota-Volume
    PolCotaArea = None      # Polinomio Cota-Area
    CoefEvapMensal = None         # Coeficientes de Evaporacao
    NumConjMaq = None       # Numero de Conjuntos de Maquinas
    MaqporConj = None       # Numero de Maquinas por Conjunto
    PEfporConj = None       # POtencia Efetiva por Maquina do Conjunto

    CF_HBQT = None          # Nao sei qual e esta informacao ??????
    CF_HBQG = None          # Nao sei qual e esta informacao ??????
    CF_HBPT = None          # Nao sei qual e esta informacao ??????

    AltEfetConj = None      # Altura de Queda Efetiva do Conjunto
    VazEfetConj = None      # Vazao Efetiva do Conjunto
    ProdEsp = None          # Produtibilidade Especifica
    PerdaHid = None         # Perda Hidraulica
    NumPolVNJ = None        # Numero de Polinomios Vazao Nivel Jusante

    PolVazNivJus = None     # Polinomios Vazao Nivel Jusante

    CotaRefNivelJus = None  # Cota Referencia Nivel de Jusante
    CFMed = None            # Cota Media do Canal de Fuga
    InfVertCanalFuga = None     # Informacao Canal de Fuga - Nao sei qual e esta informacao ??????
    FatorCargaMax = None    # Fator de Caga Maximo - Nao sei qual e esta informacao ?????????
    FatorCargaMin = None    # Fator de Caga Minimo - Nao sei qual e esta informacao ?????????
    VazMin = None           # Vazao Minima Obrigatoria
    UnidBase = None         # Numero de Unidades de Base
    TipoTurb = None         # Tipo de Turbina Hidraulica
    RepresConj = None      # Representacao Conjunto de Maquina - Nao sei qual e esta informacao ?????
    TEIF = None
    TEIFH = None            # Taxa Equivalente de Indisponibilidade Forcada Hidraulica
    IP = None               # Indisponibilidade Programada
    TipoPerda = None        # Tipo Perda Hidraulica
    Data = None             # Nao sei qual e esta informacao ??????
    Observ = None           # Observacao
    VolRef = None           # Volume de Referencia
    TipoReg = None          # Tipo de Regulacao

    # Dados adicionado após leiura do hidr.dat
    Status = 'NE'
    ProdEqv = None
    VolUtil = None
    VazEfet = None
    AltEfet = None
    PotEfet = None
    EngolMax = None
    EngolMaxT = None  # adicionado
    PotEfetT = None   # adicionado
    Montantes = None
    Evaporacao = None
    FunProdHidr = None
    VazIncPrevistaT = None
    VolumeEsperaT = None
    DesvioAguaT = None
    ManutencaoT = None
    VolIni = None
    Ree = None
    DefAntMontanteT = None
    ParamUsinaElev = None
    RestricaoOperacaoT = None

    # # Dados Adicionais Especificados no arquivo de configuracao hidraulica (CONFHD)
    # Ree = None
    # Status = None
    # VolIni = None
    # Modif = None
    # AnoI = None
    # AnoF = None

    # Dados Adicinais Calculados para as Usinas pertecentes a configuracao hidraulica (CONFHD)
    # VolUtil = None
    # VazEfet = None
    # PotEfet = None
    # PotNom = None
    # Ro65 = None             # PDTMED (NEWAVE) - PROD. ASSOCIADA A ALTURA CORRESPONDENTE A 65% DO V.U.
    # Ro50 = None
    # RoMax = None            # PDTMAX (NEWAVE) - PROD. ASSOCIADA A ALTURA MAXIMA
    # RoMin = None            # PDTMIN (NEWAVE) - PROD. ASSOCIADA A ALTURA MINIMA
    # RoEquiv = None          # PRODT (NEWAVE) - PROD. EQUIVALENTE ( DO VOL. MINIMO AO VOL. MAXIMO )
    # RoEquiv65 = None        # PRODTM (NEWAVE) - PROD. EQUIVALENTE ( DO VOL. MINIMO A 65% DO V.U. )
    # Engolimento = None
    # RoAcum = None           # PDTARM (NEWAVE) - PROD. ACUM. PARA CALCULO DA ENERGIA ARMAZENADA
    # RoAcum65 = None         # PDAMED (NEWAVE) - PROD. ACUM. PARA CALCULO DA ENERGIA ARMAZENADA CORRESPONDENTE A 65% DO V.U.
    # RoAcumMax = None        # PDCMAX e PDVMAX (NEWAVE) - PROD. ACUM.
    # RoAcumMed = None        # PDTCON, PDCMED e PDVMED (NEWAVE) - PROD. ACUM.
    # RoAcumMin = None        # PDCMIN e PDVMIN (NEWAVE) - PROD. ACUM.
    #
    # RoAcum_A_Ree = None
    # RoAcum_B_Ree = None
    # RoAcum_C_Ree = None
    # RoAcum_A_Sist = None
    # RoAcum_B_Sist = None
    # RoAcum_C_Sist = None
    #
    # RoAcumEntreResRee = None
    # RoAcumEntreResSist = None
    #
    # RoAcumDesvAguaEquiv = None

    # Vazoes Naturais, Incrementais e Par(p)
    # Vazoes = None       # Historico de Vazoes naturais (imes, ilag)
    # FAC = None          # Funcao de Autocorrelacao (imes, ilag)
    # FACP = None         # Funcao de Autocorrelacao Parcial (imes, ilag)
    # CoefParp = None     # Coeficientes do Modelo par(p) (imes,ilag)
    # CoefIndParp  = None     # Coeficientes independentes do Modelo par(p) (imes) - Aditivo = 0 - Multiplicativo > 0
    # Ordem = None        # Ordem do modelo par(p) para todos os meses (mes)
    #
    # # Parametros da usina Dependentes do Tempo - Especificados (MODIF.DAT)
    # VolMinT = None     # Volume Mínimo Operativo (pode variar mes a mes)
    # VolMaxT = None     # Volume Maximo Operativo (pode variar mes a mes)
    # VolMinP = None     # Volume Mínimo com adocao de penalidade (pode variar mes a mes)
    # VazMinT = None     # Vazao Minima pode variar mes a mes
    # CFugaT  = None     # Cota do Canal de Fuga (pode varia mes a mes)
    #
    # # Parametros relativos a expansao hidrica que variam no tempo para usinas 'EE' e 'NE' (EXPH)
    # StatusVolMorto = None       # Status do Volume Morto - 0: Nao Comecou Encher - 1: Enchendo - 2: Cheio
    # VolMortoTempo = None        # Evolucao do Volume Minimo da Usina
    # StatusMotoriz  = None       # Status da Motorizacao  - 0: Nao Comecou Motorizar - 1: Motorizando - 3: Motorizada
    # UnidadesTempo = None        # Numero de Unidades em cada mes
    # EngolTempo = None           # Evolucao do Engolimento Maximo da Usina
    # PotenciaTempo = None        # Evolucao da Potencia Instalada da Usina

    # VazDesv = None

    # Hydro plants parameters calculus
    def parameters_calculus_run(self, data, init_date: datetime) -> list:

        for idx, uhe in enumerate(data):

            data[idx].Montantes = self.get_upstream_list(uhe=uhe, data=data)
            data[idx].VolUtil = self.useful_volume_calculus(data=data[idx])
            data[idx].Evaporacao = self.evaporation_calculus(data=data[idx])
            if data[idx].TipoTurb != 0:
                data[idx].VazEfet = self.effective_inflow_calculus(data=data[idx])
                data[idx].AltEfet = self.effective_height_calculus(data=data[idx])
                data[idx].PotEfet = self.effective_power_calculus(data=data[idx])
                data[idx].EngolMax = self.maximum_turbination_calculus(data=data[idx])
                data[idx].EngolMaxT, data[idx].PotEfetT = self.maintenance_parameters_calculus(data=data[idx], ref_date=init_date)
                data[idx].ProdEqv = self.produtibility_calculus(data=data[idx])

        for idx, uhe in enumerate(data):
            data[idx].EarmMax = self.energy_storage_calculus(data=data, pos=idx)

        return data

    def produtibility_calculus(self, data) -> float:

        # Cálculo da Produtibilidade
        # Cota máxima
        if data.VolMin != data.VolMax:
            cota = 0.
            for i in range(5):
                cota += (float(data.PolCotaVol[i]) / (i + 1)) * ((data.VolMax ** (i + 1)) - (data.VolMin ** (i + 1)))
            cota /= data.VolUtil
        else:
            cota = 0
            for i in range(5):
                cota += float(data.PolCotaVol[i]) * (data.VolRef ** i)

        # Altura líquida e produtibilidade equivalente
        if data.TipoPerda == 1:
            h_eq = (cota - data.CFMed) * (1 - (data.PerdaHid / 100))
        elif data.TipoPerda == 2:
            h_eq = cota - data.CFMed - data.PerdaHid
        else:
            print('Tipo de perdas não identificado!')

        prod = data.ProdEsp * h_eq

        return prod

    def energy_storage_calculus(self,  data, pos) -> float:

        # Cálculo da Energia Armazenada
        if data[pos].TipoTurb == 0:

            earm_max = 0.

            return earm_max

        else:

            produtib = data[pos].ProdEqv
            pos_uhe = pos
            while True:
                jusante = data[pos_uhe].Jusante
                if jusante != 0:
                    try:
                        idx_jusante = [i for i, x in enumerate(data) if x.Codigo == jusante][0]
                    except:
                        if jusante == 58:
                            jusante = 61
                        elif jusante == 128:
                            jusante = 130
                        elif jusante == 149:
                            jusante = 154
                        elif jusante == 268:
                            jusante = 275
                        elif jusante == 282:
                            jusante = 34
                        elif jusante == 313:
                            jusante = 315
                        elif jusante == 186:
                            break
                        idx_jusante = [i for i, x in enumerate(data) if x.Codigo == jusante][0]

                    if data[idx_jusante].TipoTurb != 0:
                        produtib += data[idx_jusante].ProdEqv
                    pos_uhe = idx_jusante
                else:
                    break

            earm_max = (1 / 2.6298) * data[pos].VolUtil * produtib

            return earm_max

    def evaporation_calculus(self, data) -> dict:

        if data.Status != 'EX':
            return dict()

        value = dict()
        value['ValorInicial'] = list()
        value['Coeficiente'] = list()
        vol = data.VolMin + (1/100) * data.VolIni * data.VolUtil

        for i in range(12):

            coef_mensal = data.CoefEvapMensal[i]

            # Evaporação Inicial
            h_mon, area = 0, 0
            for j in range(5):
                h_mon += data.PolCotaVol[j] * (vol ** j)
            for j in range(5):
                area += data.PolCotaArea[j] * (h_mon ** j)
            evapo = (1/(3.6 * 24 * monthrange(date.today().year, (i+1))[1])) * coef_mensal * area
            value['ValorInicial'].append(evapo)

            # Coeficiente
            aux1, aux2 = 0, 0
            for j in range(1, 5):
                aux1 += j * data.PolCotaArea[j] * (h_mon ** (j-1))
                aux2 += j * data.PolCotaVol[j] * (vol ** (j-1))
            coeficiente = (1/(3.6*24*monthrange(date.today().year, (i+1))[1])) * coef_mensal * aux1 * aux2
            value['Coeficiente'].append(coeficiente)

        return value

    def get_upstream_list(self, uhe, data) -> list:

        jusantes = [i.Jusante for i in data]
        lista = [data[i].Codigo for i, x in enumerate(jusantes) if x == uhe.Codigo]

        return lista

    def useful_volume_calculus(self, data):

        value = data.VolMax - data.VolMin
        return value

    def effective_inflow_calculus(self, data):

        value = float(np.vdot(data.MaqporConj, data.VazEfetConj))
        return value

    def effective_height_calculus(self, data):

        value = float(np.vdot(data.MaqporConj, data.AltEfetConj) / sum(data.MaqporConj))
        return value

    def effective_power_calculus(self, data):

        value = float(np.vdot(data.MaqporConj, data.PEfporConj))

        return value

    def maximum_turbination_calculus(self, data) -> float:

        hmon = 0
        for i in range(5):
            vol = data.VolMin + (1/100) * data.VolIni * data.VolUtil
            hmon += data.PolCotaVol[i] * (vol**i)

        if data.TipoTurb == 1 or self.TipoTurb == 3:
            alpha = 0.5
        else:
            alpha = 0.2
        if data.TipoPerda == 1:
            hliq = (hmon - data.CFMed) * (1 - (data.PerdaHid / 100))
        else:
            hliq = hmon - data.CFMed - data.PerdaHid

        engol = data.VazEfet * (hliq/data.AltEfet) ** alpha

        return engol

    def maintenance_parameters_calculus(self, data, ref_date: datetime) -> tuple:

        engol_max = data.VazIncPrevistaT[['DI', 'HI', 'MI']].copy()
        engol_max['ENGOL'] = np.nan

        pot = data.VazIncPrevistaT[['DI', 'HI', 'MI']].copy()
        pot['POT'] = np.nan

        for idx in engol_max.index:

            MaqPorConj = data.MaqporConj.copy()
            unid_maintenance = []

            for i in data.ManutencaoT.index:

                # Interval Maintenance
                if data.ManutencaoT.loc[i, 'DI'] < engol_max.loc[0, 'DI']:
                    new_month = ref_date.month + 1
                    new_year = ref_date.year
                    if new_month == 13:
                        new_month = 1
                        new_year = new_year + 1
                    init_date = datetime(new_year, new_month, int(data.ManutencaoT.loc[i, 'DI']),
                                         int(data.ManutencaoT.loc[i, 'HI']), int(data.ManutencaoT.loc[i, 'MI']*30))
                else:
                    init_date = datetime(ref_date.year, ref_date.month, int(data.ManutencaoT.loc[i, 'DI']),
                                         int(data.ManutencaoT.loc[i, 'HI']), int(data.ManutencaoT.loc[i, 'MI']*30))
                if data.ManutencaoT.loc[i, 'DF'] < engol_max.loc[0, 'DI']:
                    new_month = ref_date.month + 1
                    new_year = ref_date.year
                    if new_month == 13:
                        new_month = 1
                        new_year = new_year + 1
                    end_date = datetime(new_year, new_month, int(data.ManutencaoT.loc[i, 'DF']),
                                        int(data.ManutencaoT.loc[i, 'HF']), int(data.ManutencaoT.loc[i, 'MF']*30))
                else:
                    end_date = datetime(ref_date.year, ref_date.month, int(data.ManutencaoT.loc[i, 'DF']),
                                        int(data.ManutencaoT.loc[i, 'HF']), int(data.ManutencaoT.loc[i, 'MF']*30))
                length = int((end_date - init_date).total_seconds() / 60 / 30)
                interval_maintenance = [init_date + delta(minutes=30*i) for i in range(length)]

                # Interval study stage
                if engol_max.loc[idx, 'DI'] < engol_max.loc[0, 'DI']:
                    new_month = ref_date.month + 1
                    new_year = ref_date.year
                    if new_month == 13:
                        new_month = 1
                        new_year = new_year + 1
                    init_date = datetime(new_year, new_month, int(engol_max.loc[idx, 'DI']),
                                         int(engol_max.loc[idx, 'HI']), int(engol_max.loc[idx, 'MI']*30))
                else:
                    init_date = datetime(ref_date.year, ref_date.month, int(engol_max.loc[idx, 'DI']),
                                         int(engol_max.loc[idx, 'HI']), int(engol_max.loc[idx, 'MI']*30))

                if idx < engol_max.index[-1]:
                    if engol_max.loc[idx+1, 'DI'] < engol_max.loc[0, 'DI']:
                        new_month = ref_date.month + 1
                        new_year = ref_date.year
                        if new_month == 13:
                            new_month = 1
                            new_year = new_year + 1
                        end_date = datetime(new_year, new_month, int(engol_max.loc[idx+1, 'DI']),
                                            int(engol_max.loc[idx+1, 'HI']), int(engol_max.loc[idx+1, 'MI']*30))
                    else:
                        end_date = datetime(ref_date.year, ref_date.month, int(engol_max.loc[idx+1, 'DI']),
                                            int(engol_max.loc[idx+1, 'HI']), int(engol_max.loc[idx+1, 'MI']*30))
                else:
                    if engol_max.loc[idx, 'DI'] < engol_max.loc[0, 'DI']:
                        new_month = ref_date.month + 1
                        new_year = ref_date.year
                        if new_month == 13:
                            new_month = 1
                            new_year = new_year + 1
                        end_date = datetime(new_year, new_month, int(engol_max.loc[idx, 'DI']),
                                            23, 30)
                    else:
                        end_date = datetime(ref_date.year, ref_date.month, int(engol_max.loc[idx, 'DI']),
                                            23, 30)

                length = int((end_date - init_date).total_seconds() / 60 / 30)
                interval_stage = [init_date + delta(minutes=30*i) for i in range(length)]

                # Test
                condition = [True if x in interval_stage else False for x in interval_maintenance]

                true_count = sum(condition)

                if (true_count >= int(len(interval_stage) / 2)) and int(data.ManutencaoT.loc[i, 'INDUNI']) not in unid_maintenance:
                    pos = int(data.ManutencaoT.loc[i, 'INDGRUPO'] - 1)
                    MaqPorConj[pos] = MaqPorConj[pos] - 1
                    unid_maintenance.append(data.ManutencaoT.loc[i, 'INDUNI'])

            MaqPorConj_maintenance = [max(x, 0) for x in MaqPorConj]

            # Engolimento Máximo
            hmon = 0
            for i in range(5):
                vol = data.VolMin + (1 / 100) * data.VolIni * data.VolUtil
                hmon += data.PolCotaVol[i] * (vol ** i)

            if data.TipoTurb == 1 or self.TipoTurb == 3:
                alpha = 0.5
            else:
                alpha = 0.2
            if data.TipoPerda == 1:
                hliq = (hmon - data.CFMed) * (1 - (data.PerdaHid / 100))
            else:
                hliq = hmon - data.CFMed - data.PerdaHid

            vaz_efet = float(np.vdot(MaqPorConj_maintenance, data.VazEfetConj))
            if sum(MaqPorConj_maintenance) > 0:
                alt_efet = float(np.vdot(MaqPorConj_maintenance, data.AltEfetConj) / sum(MaqPorConj_maintenance))
                value_engol_max = vaz_efet * (hliq / alt_efet) ** alpha
            else:
                value_engol_max = 0.
            value_pot = float(np.vdot(MaqPorConj_maintenance, data.PEfporConj))
            engol_max.loc[idx, 'ENGOL'] = value_engol_max
            pot.loc[idx, 'POT'] = value_pot

        # engol_max = engol_max.ffill(axis=0)
        # pot = pot.ffill(axis=0)

        return engol_max, pot

    def fph_param_calculus(self, data, disc_turb: int, disc_vol: int = None, disc_vert: int = None):

        turb_disc = np.linspace(start=0., stop=data.EngolMax, num=disc_turb)

        if disc_vol:

            # vol_ini = data.VolMin + (1/100)*data.VolIni*data.VolUtil
            # variation = 0.2
            # vol_disc = np.linspace(start=(1-variation)*vol_ini, stop=(1+variation)*vol_ini, num=disc_vol)
            vol_disc = np.linspace(start=data.VolMin, stop=data.VolMax, num=disc_vol)

            if disc_vert:

                vert_disc = np.linspace(start=0., stop=5*data.EngolMax, num=disc_vert)

                vol, turb, vert = np.meshgrid(vol_disc, turb_disc, vert_disc)
                fph = self.fph_function(data, vol, turb, vert)
                vol_1d = np.reshape(vol, vol.size)
                turb_1d = np.reshape(turb, turb.size)
                vert_1d = np.reshape(vert, vert.size)
                fph_1d = np.reshape(fph, fph.size)
                points = np.array([[vol_1d[i], turb_1d[i], vert_1d[i], fph_1d[i]] for i in range(fph_1d.size)])
                output = ConvexHull().run(points=points, variables=['vol', 'turb', 'vert'])
                return output

            else:

                vol, turb = np.meshgrid(vol_disc, turb_disc)
                vert = np.zeros(turb.shape)
                fph = self.fph_function(data, vol, turb, vert)
                vol_1d = np.reshape(vol, vol.size)
                turb_1d = np.reshape(turb, turb.size)
                fph_1d = np.reshape(fph, fph.size)
                points = np.array([[vol_1d[i], turb_1d[i], fph_1d[i]] for i in range(fph_1d.size)])
                output = ConvexHull().run(points=points, variables=['vol', 'turb'])
                return output

        else:

            if disc_vert:

                vert_disc = np.linspace(start=0., stop=5 * data.EngolMax, num=disc_vert)

                turb, vert = np.meshgrid(turb_disc, vert_disc)
                vol = data.VolMax * np.ones(turb.shape)
                fph = self.fph_function(data, vol, turb, vert)
                turb_1d = np.reshape(turb, turb.size)
                vert_1d = np.reshape(vert, vert.size)
                fph_1d = np.reshape(fph, fph.size)
                points = np.array([[turb_1d[i], vert_1d[i], fph_1d[i]] for i in range(fph_1d.size)])
                output = ConvexHull().run(points=points, variables=['turb', 'vert'])
                return output

            else:

                turb = turb_disc
                vol = data.VolMax * np.ones(turb.shape)
                vert = np.zeros(turb.shape)
                fph = self.fph_function(data, vol, turb, vert)
                points = np.array([[turb[i], fph[i]] for i in range(fph.size)])
                output = ConvexHull().run(points=points, variables=['turb'])
                return output

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # surf = ax.plot_surface(vol, turb, fph, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        # ax.set_xlabel('Volume [$hm^{3}$]')
        # ax.set_ylabel('Vazão Turbinada [$m^{3}/s$]')
        # ax.set_zlabel('Potência [$MW$]')
        # ax.set_zlim(0.95*np.min(fph), 1.05*np.max(fph))
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        # # fig.colorbar(surf, shrink=0.5, aspect=5)
        # plt.show()

        # chull = ConvexHull(values)

    def fph_function(self, data, vol, turb, vert) -> array:

        hmon = 0
        hjus = 0
        for i in range(5):
            hmon += data.PolCotaVol[i] * (vol ** i)
            hjus += data.PolVazNivJus[0][i] * ((turb+vert) ** i)
        if data.TipoPerda == 1:
            fph = data.ProdEsp * turb * (hmon - hjus) * (1 - (data.PerdaHid / 100))
            return fph
        elif data.TipoPerda == 2:
            fph = data.ProdEsp * turb * (hmon - hjus - data.PerdaHid)
            return fph
        else:
            print('Tipo de perdas não tratado no problema!')
            quit()

    def all_hydroplants_fph_calculus(self, data: list) -> list:

        for idx, uhe in enumerate(data):

            if data[idx].TipoTurb != 0:

                if data[idx].Status == 'EX':

                    data[idx].FunProdHidr = dict()
                    if data[idx].VolMin != data[idx].VolMax:
                        if data[idx].InfVertCanalFuga == 1 and any([True if x != 0. else False for x in data[idx].PolVazNivJus[0][1:]]):
                            data[idx].FunProdHidr['Tipo'] = 'Vol/Turb/Vert'
                            data[idx].FunProdHidr['FPH'] = self.fph_param_calculus(data=data[idx], disc_vol=5, disc_turb=5, disc_vert=5)
                        else:
                            data[idx].FunProdHidr['Tipo'] = 'Vol/Turb'
                            data[idx].FunProdHidr['FPH'] = self.fph_param_calculus(data=data[idx], disc_vol=5, disc_turb=5)

                    else:
                        if data[idx].InfVertCanalFuga == 1 and any([True if x != 0. else False for x in data[idx].PolVazNivJus[0][1:]]):
                            data[idx].FunProdHidr['Tipo'] = 'Turb/Vert'
                            data[idx].FunProdHidr['FPH'] = self.fph_param_calculus(data=data[idx], disc_turb=5, disc_vert=5)
                        else:
                            data[idx].FunProdHidr['Tipo'] = 'Turb'
                            data[idx].FunProdHidr['FPH'] = self.fph_param_calculus(data=data[idx], disc_turb=10)

        return data