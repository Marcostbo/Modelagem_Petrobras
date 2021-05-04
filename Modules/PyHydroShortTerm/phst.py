import os
from typing import List
from datetime import datetime, timedelta as delta
import pandas as pd
from tempfile import NamedTemporaryFile
import struct
import numpy as np
import logging

from .Config import *
from .term import term
from .hidr import hidr
from .conf import conf
from .sist import sist
from .interc import interc

from Utils.Logger import get_logger

logger = get_logger(module_name=__name__)


class phst(object):

    hidr = list()
    term = list()
    sist = list()
    interc = list()
    conf = conf

    arquivos: list = []
    titulo: str = ''
    diretorio: str = ''

    def __init__(self, diretorio: str):

        self.diretorio = diretorio
        self.arquivos = self.read_dessem_arq_file(name_file='dessem.arq')

        logger.info(msg='Iniciando processo de leitura do deck de entrada!')

        # Managemet Read Files
        hidr_data, vazoes, init_date = self.management_read_files()

        logger.info(msg='Leitura do deck de entrada realizada com sucesso!')

        logger.info(msg='Iniciando processo de criação dos objetos da classe!')

        # Generate objects
        self.conf_create(init_date=init_date)
        self.variable_create(param=hidr_data, vazoes=vazoes)

        logger.info(msg='Objetos criados com sucesso!')

        # Hydro plants variable calculus
        self.hidr = hidr().parameters_calculus_run(data=self.hidr, init_date=self.conf.DataInicial)

        # Thermal plants variable calculus
        self.term = term().parameters_calculus_run(data=self.term, init_date=self.conf.DataInicial)

    # region Fill Objects (hidr, sist, term, ...)
    def conf_create(self, init_date: datetime):

        self.conf.DataInicial = init_date
        for i, file in enumerate(list_files_data):
            for key, value in file.items():
                if value['tipo'] == 'CONFIG':
                    if key == 'TM':
                        self.conf.DiscTemporalT = value['dados'][['DI', 'HI', 'MI', 'Duracao', 'Patamar']]
                    elif key == 'TX':
                        self.conf.TxJurosAnual = float(value['dados']['TX'])

    def variable_create(self, param: list, vazoes: pd.DataFrame) -> None:

        # Initialization all temporal series variable
        # Select only hydro plants with data parameters
        all_hidr = []
        for uhe in param:
            if uhe.Nome:
                all_hidr.append(uhe)

        # HIDR
        for idx_hidr, uhe in enumerate(all_hidr):
            all_hidr[idx_hidr].VazIncPrevistaT = self.conf.DiscTemporalT[['DI', 'HI', 'MI']].copy()
            all_hidr[idx_hidr].VazIncPrevistaT['VAZAO'] = np.nan
            all_hidr[idx_hidr].VolumeEsperaT = self.conf.DiscTemporalT[['DI', 'HI', 'MI']].copy()
            all_hidr[idx_hidr].VolumeEsperaT['VOL'] = np.nan
            all_hidr[idx_hidr].DesvioAguaT = self.conf.DiscTemporalT[['DI', 'HI', 'MI']].copy()
            all_hidr[idx_hidr].DesvioAguaT['TX'] = np.nan
            # all_hidr[idx_hidr].Manutencao = self.conf.DiscTemporal[['DI', 'HI', 'MI']].copy()
            # all_hidr[idx_hidr].Manutencao['MANUT'] = np.nan
            all_hidr[idx_hidr].ManutencaoT = pd.DataFrame(data=[])
            all_hidr[idx_hidr].DefAntMontanteT = dict()
            all_hidr[idx_hidr].ParamUsinaElev = dict()
            all_hidr[idx_hidr].RestricaoOperacaoT = dict()
        for idx_vazoes in vazoes.index:
            dd = vazoes.loc[idx_vazoes, 'DI']
            hr = vazoes.loc[idx_vazoes, 'HI']
            mh = vazoes.loc[idx_vazoes, 'MI']
            vazao = float(vazoes.loc[idx_vazoes, 'VAZAO'])
            idx_hidr = [all_hidr[x].Codigo for x in range(len(all_hidr))].index(vazoes.loc[idx_vazoes, 'NUM'])
            all_hidr[idx_hidr].VazIncPrevistaT.loc[(all_hidr[idx_hidr].VazIncPrevistaT['DI'] == dd) &
                                                   (all_hidr[idx_hidr].VazIncPrevistaT['HI'] == hr) &
                                                   (all_hidr[idx_hidr].VazIncPrevistaT['MI'] == mh), 'VAZAO'] = vazao

        # TERM
        condition = True
        for i, file in enumerate(list_files_data):
            for key, value in file.items():
                if key == 'CADUSIT' and condition:
                    value_sorted = value['dados'].sort_values(by=['NUM'], ascending=True)
                    value_sorted.index = [i for i in range(value_sorted.shape[0])]
                    for idx in value_sorted.index:
                        self.term.append(term())
                        self.term[idx].Codigo = int(value_sorted.loc[idx, 'NUM'])
                        self.term[idx].Nome = value_sorted.loc[idx, 'NOME'].strip()
                        self.term[idx].Sist = int(value_sorted.loc[idx, 'NSUB'])
                        self.term[idx].DataInicioOper = datetime(int(value_sorted.loc[idx, 'ANO']), int(value_sorted.loc[idx, 'MES']),
                                                                 int(value_sorted.loc[idx, 'DIA']), int(value_sorted.loc[idx, 'HORA']),
                                                                 30*int(value_sorted.loc[idx, 'MI']))
                        self.term[idx].NumUnidades = int(value_sorted.loc[idx, 'NUNI'])
                        self.term[idx].UnidadeGeradoraT = dict()
                        self.term[idx].ManutencaoT = pd.DataFrame(data=[])
                    condition = False
                    break
            if not condition:
                break

        # Initialization all temporal series variable
        # TERM
        for idx in value_sorted.index:
            self.term[idx].RestricaoGeracaoT = self.conf.DiscTemporalT[['DI', 'HI', 'MI']].copy()
            self.term[idx].RestricaoGeracaoT['GMIN/DECR'] = np.nan
            self.term[idx].RestricaoGeracaoT['GMAX/CRES'] = np.nan
            self.term[idx].RestricaoGeracaoT['TIPO'] = np.nan
            self.term[idx].RestricaoGeracaoT['GINI'] = np.nan
            self.term[idx].GeracaoFixaT = self.conf.DiscTemporalT[['DI', 'HI', 'MI']].copy()
            self.term[idx].GeracaoFixaT['GERACAO'] = np.nan

        # SIST
        for i, file in enumerate(list_files_data):
            for key, value in file.items():
                if key == 'SIST' and value['tipo'] == 'SIST':
                    value_sorted = value['dados'].sort_values(by=['NUM'], ascending=True)
                    value_sorted.index = [i for i in range(value_sorted.shape[0])]
                    for idx in value_sorted.index:
                        self.sist.append(sist())
                        self.sist[idx].Codigo = int(value_sorted.loc[idx, 'NUM'])
                        self.sist[idx].Nome = value_sorted.loc[idx, 'NOME'].strip()
                        self.sist[idx].Sigla = value_sorted.loc[idx, 'SIG'].strip()
                        if value_sorted.loc[idx, 'SIG'].strip() != 'FC':
                            self.sist[idx].ContratoImportacaoT = dict()
                            self.sist[idx].ContratoExportacaoT = dict()
                            # self.sist[idx].CustoDeficit = self.conf.DiscTemporal[['DI', 'HI', 'MI']].copy()

        # INTERC
        for i, file in enumerate(list_files_data):
            for key, value in file.items():
                if key == 'IA' and value['tipo'] == 'INTERC':
                    aux = [(value['dados'].loc[i, 'SS1'].strip(), value['dados'].loc[i, 'SS2'].strip()) for i in value['dados'].index]
                    de_para = set(aux)
                    cont = 0
                    for x in de_para:
                        self.interc.append(interc())
                        self.interc[cont].Codigo = cont + 1
                        self.interc[cont].Sigla = f'{x[0]}->{x[1]}'
                        self.interc[cont].De = x[0]
                        self.interc[cont].Para = x[1]
                        self.interc[cont].LimiteT = self.conf.DiscTemporalT[['DI', 'HI', 'MI']].copy()
                        self.interc[cont].LimiteT['Limite'] = np.nan

                        self.interc.append(interc())
                        self.interc[cont+1].Codigo = cont + 2
                        self.interc[cont+1].Sigla = f'{x[1]}->{x[0]}'
                        self.interc[cont+1].De = x[1]
                        self.interc[cont+1].Para = x[0]
                        self.interc[cont+1].LimiteT = self.conf.DiscTemporalT[['DI', 'HI', 'MI']].copy()
                        self.interc[cont+1].LimiteT['Limite'] = np.nan

                        cont += 2

                    for idx in value['dados'].index:
                        di = int(value['dados'].loc[idx, 'DI'])
                        hi = int(value['dados'].loc[idx, 'HI'])
                        mi = int(value['dados'].loc[idx, 'MI'])
                        de = value['dados'].loc[idx, 'SS1'].strip()
                        para = value['dados'].loc[idx, 'SS2'].strip()
                        valor1 = float(value['dados'].loc[idx, 'SS1->SS2'])
                        valor2 = float(value['dados'].loc[idx, 'SS2->SS1'])
                        idx_interc1 = [self.interc[i].Sigla for i in range(len(self.interc))].index(f"{de}->{para}")
                        idx_interc2 = [self.interc[i].Sigla for i in range(len(self.interc))].index(f"{para}->{de}")

                        self.interc[idx_interc1].LimiteT.loc[(self.interc[idx_interc1].LimiteT['DI'] == di) &
                                                            (self.interc[idx_interc1].LimiteT['HI'] == hi) &
                                                            (self.interc[idx_interc1].LimiteT['MI'] == mi), 'Limite'] = valor1
                        self.interc[idx_interc2].LimiteT.loc[(self.interc[idx_interc2].LimiteT['DI'] == di) &
                                                            (self.interc[idx_interc2].LimiteT['HI'] == hi) &
                                                            (self.interc[idx_interc2].LimiteT['MI'] == mi), 'Limite'] = valor2

        # HIDR (restrição operativa)
        rest = config_operuh['OPERUH REST']['dados']
        elem = config_operuh['OPERUH ELEM']['dados']
        lim = config_operuh['OPERUH LIM']['dados']
        var = config_operuh['OPERUH VAR']['dados']
        for idx in rest.index:
            id_rest = int(rest.loc[idx, 'NUM'])
            tipo = rest.loc[idx, 'TIP'].strip()
            considera = bool(int(rest.loc[idx, 'FLAG']))
            vini = None if not isinstance(rest.loc[idx, 'VALORINI'], str) or rest.loc[idx, 'VALORINI'] == '.' else float(rest.loc[idx, 'VALORINI'])
            code_uhe = int(elem[elem['NUM'] == id_rest]['NUSI'].values[0])
            idx_hidr = [all_hidr[i].Codigo for i in range(len(all_hidr))].index(code_uhe)
            var_codes_list = elem[elem['NUM'] == id_rest]['COD'].values
            for var_code in var_codes_list:
                variable = constraints_variable_codes[var_code]
                all_hidr[idx_hidr].RestricaoOperacaoT[variable] = dict()
                all_hidr[idx_hidr].RestricaoOperacaoT[variable]['Considera'] = considera
                all_hidr[idx_hidr].RestricaoOperacaoT[variable]['ValorInicial'] = vini
                all_hidr[idx_hidr].RestricaoOperacaoT[variable]['TipoRestricao'] = tipo
                if tipo == 'L':
                    all_hidr[idx_hidr].RestricaoOperacaoT[variable]['Restricao'] = self.conf.DiscTemporalT[['DI', 'HI', 'MI']].copy()
                    linf = None if np.isnan(lim[lim['NUM'] == id_rest]['LINF'].values[0]) else float(lim[lim['NUM'] == id_rest]['LINF'].values[0])
                    lsup = None if np.isnan(lim[lim['NUM'] == id_rest]['LSUP'].values[0]) else float(lim[lim['NUM'] == id_rest]['LSUP'].values[0])
                    all_hidr[idx_hidr].RestricaoOperacaoT[variable]['Restricao'].loc[0, 'LInf'] = linf
                    all_hidr[idx_hidr].RestricaoOperacaoT[variable]['Restricao'].loc[0, 'LSup'] = lsup
                elif tipo == 'V':
                    all_hidr[idx_hidr].RestricaoOperacaoT[variable]['Restricao'] = self.conf.DiscTemporalT[['DI', 'HI', 'MI']].copy()
                    ramp_dec_perc = None if np.isnan(var[var['NUM'] == id_rest]['RampDecrPerc'].values[0]) else float(var[var['NUM'] == id_rest]['RampDecrPerc'].values[0])
                    ramp_acr_perc = None if np.isnan(var[var['NUM'] == id_rest]['RampAcrescPerc'].values[0]) else float(var[var['NUM'] == id_rest]['RampAcrescPerc'].values[0])
                    ramp_dec_abs = None if np.isnan(var[var['NUM'] == id_rest]['RampDecrAbs'].values[0]) else float(var[var['NUM'] == id_rest]['RampDecrAbs'].values[0])
                    ramp_acr_abs = None if np.isnan(var[var['NUM'] == id_rest]['RampAcrescAbs'].values[0]) else float(var[var['NUM'] == id_rest]['RampAcrescAbs'].values[0])
                    all_hidr[idx_hidr].RestricaoOperacaoT[variable]['Restricao'].loc[0, 'RampDecrescPerc'] = ramp_dec_perc
                    all_hidr[idx_hidr].RestricaoOperacaoT[variable]['Restricao'].loc[0, 'RampAcrescPerc'] = ramp_acr_perc
                    all_hidr[idx_hidr].RestricaoOperacaoT[variable]['Restricao'].loc[0, 'RampDecrescAbs'] = ramp_dec_abs
                    all_hidr[idx_hidr].RestricaoOperacaoT[variable]['Restricao'].loc[0, 'RampAcrescAbs'] = ramp_acr_abs

        # Fill parameters
        for i, file in enumerate(list_files_data):
            for key, value in file.items():

                # HIDR
                if value['tipo'] == 'HIDR':

                    if key == 'UH':
                        for idx_uh in value['dados'].index:
                            idx_hidr = [all_hidr[x].Codigo for x in range(len(all_hidr))].index(value['dados'].loc[idx_uh, 'NUM'])
                            all_hidr[idx_hidr].VolIni = float(value['dados'].loc[idx_uh, 'VINI'])
                            all_hidr[idx_hidr].Ree = int(value['dados'].loc[idx_uh, 'NUMREE'])
                            all_hidr[idx_hidr].Status = 'EX'

                    elif key == 'VE':
                        for idx_uh in value['dados'].index:
                            idx_hidr = [all_hidr[x].Codigo for x in range(len(all_hidr))].index(value['dados'].loc[idx_uh, 'NUM'])
                            di = value['dados'].loc[idx_uh, 'DI']
                            hi = value['dados'].loc[idx_uh, 'HI']
                            mi = value['dados'].loc[idx_uh, 'MI']
                            vol = value['dados'].loc[idx_uh, 'VOL']
                            all_hidr[idx_hidr].VolumeEsperaT.loc[(all_hidr[idx_hidr].VolumeEsperaT['DI'] == di) &
                                                                 (all_hidr[idx_hidr].VolumeEsperaT['HI'] == hi) &
                                                                 (all_hidr[idx_hidr].VolumeEsperaT['MI'] == mi), 'VOL'] = vol

                    elif key == 'TVIAG':
                        for idx_uh in value['dados'].index:
                            idx_hidr = [all_hidr[x].Codigo for x in range(len(all_hidr))].index(value['dados'].loc[idx_uh, 'USIJUS'])
                            mont = int(value['dados'].loc[idx_uh, 'USIMON'])
                            all_hidr[idx_hidr].DefAntMontanteT[mont] = {'TempoViagem': value['dados'].loc[idx_uh, 'TEMPO'],
                                                                        'Defl': pd.DataFrame(data=[], columns=['DI', 'DEFL'])}
                            all_hidr[idx_hidr].DefAntMontanteT[mont]['Defl']['DI'] = [(self.conf.DataInicial - delta(days=15 - i)).day for i in range(15)]

                    elif key == 'DA':
                        for idx_uh in value['dados'].index:
                            idx_hidr = [all_hidr[x].Codigo for x in range(len(all_hidr))].index(value['dados'].loc[idx_uh, 'NUM'])
                            di = value['dados'].loc[idx_uh, 'DI']
                            tx = value['dados'].loc[idx_uh, 'TX']
                            all_hidr[idx_hidr].DesvioAguaT.loc[all_hidr[idx_hidr].DesvioAguaT['DI'] == di, 'TX'] = tx

                    elif key == 'MH':
                        hydro_list = sorted(set(list(value['dados']['NUM'])))
                        for hydro in hydro_list:
                            idx_hidr = [all_hidr[x].Codigo for x in range(len(all_hidr))].index(hydro)
                            all_hidr[idx_hidr].ManutencaoT = all_hidr[idx_hidr].ManutencaoT.append(value['dados'][value['dados']['NUM'] == hydro][['INDGRUPO', 'INDUNI', 'DI', 'HI', 'MI', 'DF', 'HF', 'MF']], ignore_index=True)

                    elif key == 'USIE':
                        for idx_uh in value['dados'].index:
                            idx_hidr = [all_hidr[x].Nome.upper() for x in range(len(all_hidr))].index(value['dados'].loc[idx_uh, 'NOME'].upper())
                            all_hidr[idx_hidr].ParamUsinaElev = {'Montante': int(value['dados'].loc[idx_uh, 'NUMMONT']), 'Jusante': int(value['dados'].loc[idx_uh, 'NUMJUS']),
                                                                  'VazMin': float(value['dados'].loc[idx_uh, 'QMIN']), 'VazMax': float(value['dados'].loc[idx_uh, 'QMAX']),
                                                                  'Taxa': float(value['dados'].loc[idx_uh, 'TX'])}

                    elif key == 'AC':
                        for idx_uh in value['dados'].index:
                            idx_hidr = [all_hidr[x].Codigo for x in range(len(all_hidr))].index(value['dados'].loc[idx_uh, 'NUM'])
                            param = value['dados'].loc[idx_uh, 'PARAM'].strip()
                            valor = value['dados'].loc[idx_uh, 'VALOR']
                            if param == 'COFEVA':
                                all_hidr[idx_hidr].CoefEvapMensal[int(valor.split()[0])-1] = float(valor.split()[1])
                            elif param == 'COTVAZ':
                                all_hidr[idx_hidr].PolVazNivJus[int(valor.split()[0])] = float(valor.split()[1])
                            elif param == 'COTTAR':
                                all_hidr[idx_hidr].PolCotaArea[int(valor.split()[0])] = float(valor.split()[1])
                            elif param == 'COTVOL':
                                all_hidr[idx_hidr].PolCotaVol[int(valor.split()[0])] = float(valor.split()[1])
                            elif param == 'DESVIO':  # verificar porque tem dois valores em DESVIO
                                all_hidr[idx_hidr].Desvio = int(valor.split()[0])
                            elif param == 'JUSMED':
                                all_hidr[idx_hidr].CFMed = float(valor.strip())
                            elif param == 'NUMCON':
                                all_hidr[idx_hidr].NumConjMaq = int(valor.strip())
                            elif param == 'NUMJUS':
                                all_hidr[idx_hidr].Jusante = int(valor.strip())
                            elif param == 'NUMMAQ':
                                all_hidr[idx_hidr].MaqporConj[int(valor.split()[0])-1] = int(valor.split()[1])
                            elif param == 'NUMPOS':
                                all_hidr[idx_hidr].Posto = int(valor.strip())
                            elif param == 'PERHID':
                                all_hidr[idx_hidr].PerdaHid = float(valor.strip())
                            elif param == 'POTEFE':
                                all_hidr[idx_hidr].PEfporConj[int(valor.split()[0])-1] = float(valor.split()[1])
                            elif param == 'PRODESP':
                                all_hidr[idx_hidr].ProdEsp = float(valor.strip())
                            elif param == 'TAXFOR':
                                all_hidr[idx_hidr].TEIF = float(valor.strip())
                            elif param == 'TAXMAN':
                                all_hidr[idx_hidr].IP = float(valor.strip())
                            elif param == 'VOLMAX':
                                all_hidr[idx_hidr].VolMax = float(valor.strip())
                            elif param == 'VOLMIN':
                                all_hidr[idx_hidr].VolMin = float(valor.strip())
                            elif param == 'VSVERT':
                                all_hidr[idx_hidr].VolMinVert = float(valor.strip())
                            elif param == 'VMDESV':
                                all_hidr[idx_hidr].VolMinDesv = float(valor.strip())
                            elif param == 'JUSENA':
                                all_hidr[idx_hidr].JusEna = int(valor.strip())
                            elif param == 'VAZEFE':
                                all_hidr[idx_hidr].VazEfetConj[int(valor.split()[0])-1] = float(valor.split()[1])
                            elif param == 'ALTEFE':
                                all_hidr[idx_hidr].AltEfetConj[int(valor.split()[0])-1] = float(valor.split()[1])
                            else:
                                logger.info(msg=f'Parâmetro {param} não tratado na alteração de cadstro (AC)!')

                    elif key == 'DEFANT':

                        for idx_uh in value['dados'].index:
                            idx_hidr = [all_hidr[x].Codigo for x in range(len(all_hidr))].index(value['dados'].loc[idx_uh, 'NUMJUS'])
                            mont = value['dados'].loc[idx_uh, 'NUMMON']
                            di = value['dados'].loc[idx_uh, 'DI']
                            all_hidr[idx_hidr].DefAntMontanteT[mont]['Defl'].loc[all_hidr[idx_hidr].DefAntMontanteT[mont]['Defl']['DI'] == di, 'DEFL'] = value['dados'].loc[idx_uh, 'DEFL']

                # TERM
                if value['tipo'] == 'TERM':

                    if key == 'CADUNIDT':

                        for idx in value['dados'].index:
                            idx_term = [self.term[x].Codigo for x in range(len(self.term))].index(int(value['dados'].loc[idx, 'NUM']))
                            self.term[idx_term].UnidadeGeradoraT[int(value['dados'].loc[idx, 'IND'])] = {'Capacidade': float(value['dados'].loc[idx, 'POT']),
                                                                                                        'GerMinAcion': float(value['dados'].loc[idx, 'POTMIN']),
                                                                                                        'TOn': float(value['dados'].loc[idx, 'TON']),
                                                                                                        'TOff': float(value['dados'].loc[idx, 'TOFF']),
                                                                                                        'CCold': float(value['dados'].loc[idx, 'CCOLD']) if not np.isnan(value['dados'].loc[idx, 'CCOLD']) else None,
                                                                                                        'CStd': float(value['dados'].loc[idx, 'CSTD']) if not np.isnan(value['dados'].loc[idx, 'CSTD']) else None,
                                                                                                        'RUp': float(value['dados'].loc[idx, 'RUP']) if not np.isnan(value['dados'].loc[idx, 'RUP']) else None,
                                                                                                        'RDown': float(value['dados'].loc[idx, 'RDOWN']) if not np.isnan(value['dados'].loc[idx, 'RDOWN']) else None,
                                                                                                        'FlagGer': int(value['dados'].loc[idx, 'FLAG']) if not np.isnan(value['dados'].loc[idx, 'FLAG']) else None,
                                                                                                        'NumMaxOsc': int(value['dados'].loc[idx, 'NO']) if not np.isnan(value['dados'].loc[idx, 'NO']) else None,
                                                                                                        'FlagUE': float(value['dados'].loc[idx, 'EQU']) if not np.isnan(value['dados'].loc[idx, 'EQU']) else None,
                                                                                                        'RTrans': float(value['dados'].loc[idx, 'RTRANS']) if not np.isnan(value['dados'].loc[idx, 'RTRANS']) else None}

                            self.term[idx_term].UnidadeGeradoraT[int(value['dados'].loc[idx, 'IND'])]['Operacao'] = self.conf.DiscTemporalT[['DI', 'HI', 'MI']].copy()
                            self.term[idx_term].UnidadeGeradoraT[int(value['dados'].loc[idx, 'IND'])]['Operacao']['LimMinOper'] = np.nan
                            self.term[idx_term].UnidadeGeradoraT[int(value['dados'].loc[idx, 'IND'])]['Operacao']['LimMaxOper'] = np.nan
                            self.term[idx_term].UnidadeGeradoraT[int(value['dados'].loc[idx, 'IND'])]['Operacao']['Custo'] = np.nan

                    elif key == 'UT':

                        for idx in value['dados'].index:
                            idx_term = [self.term[x].Codigo for x in range(len(self.term))].index(int(value['dados'].loc[idx, 'NUM']))
                            tipo = 'RAMPA' if int(value['dados'].loc[idx, 'FLAGREST']) == 1 else 'LIMITE'
                            di = value['dados'].loc[idx, 'DI']
                            hi = value['dados'].loc[idx, 'HI']
                            mi = value['dados'].loc[idx, 'MI']
                            gmin_decr = float(value['dados'].loc[idx, 'GMIN/DEC'])
                            gmax_cres = float(value['dados'].loc[idx, 'GMAX/CRES'])
                            if tipo == 'RAMPA':
                                gini = float(value['dados'].loc[idx, 'GINI'])
                            else:
                                gini = np.nan
                            self.term[idx_term].RestricaoGeracaoT.loc[(self.term[idx_term].RestricaoGeracaoT['DI'] == di) &
                                                                     (self.term[idx_term].RestricaoGeracaoT['HI'] == hi) &
                                                                     (self.term[idx_term].RestricaoGeracaoT['MI'] == mi),
                                                                     ['TIPO', 'GMIN/DECR', 'GMAX/CRES', 'GINI']] = [tipo, gmin_decr, gmax_cres, gini]

                    elif key == 'MT':
                        term_list = sorted(set(list(value['dados']['NUM'])))
                        for code in term_list:
                            idx_term = [self.term[x].Codigo for x in range(len(self.term))].index(code)
                            self.term[idx_term].ManutencaoT = self.term[idx_term].ManutencaoT.append(value['dados'][value['dados']['NUM'] == code][['NUMGRUPO', 'DI', 'HI', 'MI', 'DF', 'HF', 'MF']], ignore_index=True)

                    elif key == 'PTOPER USIT':

                        for idx in value['dados'].index:
                            idx_term = [self.term[x].Codigo for x in range(len(self.term))].index(int(value['dados'].loc[idx, 'NUM']))
                            di = value['dados'].loc[idx, 'DI']
                            hi = value['dados'].loc[idx, 'HI']
                            mi = value['dados'].loc[idx, 'MI']
                            valor = float(value['dados'].loc[idx, 'VALOR'])
                            self.term[idx_term].GeracaoFixaT.loc[(self.term[idx_term].GeracaoFixaT['DI'] == di) &
                                                                (self.term[idx_term].GeracaoFixaT['HI'] == hi) &
                                                                (self.term[idx_term].GeracaoFixaT['MI'] == mi), 'GERACAO'] = valor

                    elif key == 'INIT':

                        for idx in value['dados'].index:
                            idx_term = [self.term[x].Codigo for x in range(len(self.term))].index(int(value['dados'].loc[idx, 'NUM']))
                            self.term[idx_term].UnidadeGeradoraT[int(value['dados'].loc[idx, 'IND'])]['StatusIni'] = 'LIGADA' if int(value['dados'].loc[idx, 'STATUS']) == 1 else 'DESLIGADA'
                            self.term[idx_term].UnidadeGeradoraT[int(value['dados'].loc[idx, 'IND'])]['GerIni'] = float(value['dados'].loc[idx, 'GERINI'])
                            self.term[idx_term].UnidadeGeradoraT[int(value['dados'].loc[idx, 'IND'])]['TempoPermStatus'] = float(value['dados'].loc[idx, 'TEMPO']) if int(value['dados'].loc[idx, 'MH']) == 0 else (float(value['dados'].loc[idx, 'TEMPO'])+0.5)
                            self.term[idx_term].UnidadeGeradoraT[int(value['dados'].loc[idx, 'IND'])]['Trajetoria'] = 'ACIONAMENTO' if int(value['dados'].loc[idx, 'AD']) == 1 else 'DESLIGAMENTO' if int(value['dados'].loc[idx, 'AD']) == 2 else self.term[idx_term].UnidadeGeradoraT[int(value['dados'].loc[idx, 'IND'])]['StatusIni']

                    elif key == 'OPER':

                        for idx in value['dados'].index:
                            idx_term = [self.term[x].Codigo for x in range(len(self.term))].index(int(value['dados'].loc[idx, 'NUM']))
                            di = value['dados'].loc[idx, 'DI']
                            hi = value['dados'].loc[idx, 'HI']
                            mi = value['dados'].loc[idx, 'MI']
                            linf = 0 if np.isnan(value['dados'].loc[idx, 'LINF']) else float(value['dados'].loc[idx, 'LINF'])
                            lsup = self.term[idx_term].UnidadeGeradoraT[int(value['dados'].loc[idx, 'IND'])]['Capacidade'] if np.isnan(value['dados'].loc[idx, 'LSUP']) else float(value['dados'].loc[idx, 'LSUP'])
                            custo = float(value['dados'].loc[idx, 'CUSTO'])
                            self.term[idx_term].UnidadeGeradoraT[int(value['dados'].loc[idx, 'IND'])]['Operacao'].loc[(self.term[idx_term].UnidadeGeradoraT[int(value['dados'].loc[idx, 'IND'])]['Operacao']['DI'] == di) &
                                                                                                  (self.term[idx_term].UnidadeGeradoraT[int(value['dados'].loc[idx, 'IND'])]['Operacao']['HI'] == hi) &
                                                                                                  (self.term[idx_term].UnidadeGeradoraT[int(value['dados'].loc[idx, 'IND'])]['Operacao']['MI'] == mi),
                                                                                                  ['LimMinOper', 'LimMaxOper', 'Custo']] = [linf, lsup, custo]

                    elif key == 'RAMP':

                        aux = [(int(value['dados'].loc[i, 'NUM']), int(value['dados'].loc[i, 'INDUNI'])) for i in value['dados'].index]
                        ind_uni = set(aux)
                        for x in ind_uni:
                            idx_term = [self.term[x].Codigo for x in range(len(self.term))].index(x[0])
                            self.term[idx_term].UnidadeGeradoraT[x[1]]['RampaAcionamento'] = value['dados'][(value['dados']['NUM'] == x[0]) &
                                                                                                                (value['dados']['INDUNI'] == x[1]) &
                                                                                                                (value['dados']['TRAJ'] == 'A')][['POT', 'TEMPO', 'FLAG']].reset_index(drop=True)
                            self.term[idx_term].UnidadeGeradoraT[x[1]]['RampaDesligamento'] = value['dados'][(value['dados']['NUM'] == x[0]) &
                                                                                                                (value['dados']['INDUNI'] == x[1]) &
                                                                                                                (value['dados']['TRAJ'] == 'D')][['POT', 'TEMPO', 'FLAG']].reset_index(drop=True)

                    # elif key == 'CADCONF':  # em construção
                    #
                    #     for idx in value['dados'].index:
                    #         idx_term = [self.term[x].Codigo for x in range(len(self.term))].index(int(value['dados'].loc[idx, 'NUM']))
                    #                    #
                    # elif key == 'CADMIN':  # em construção
                    #
                    #     for idx in value['dados'].index:
                    #         idx_term = [self.term[x].Codigo for x in range(len(self.term))].index(int(value['dados'].loc[idx, 'NUM']))
                    #

                # SIST
                if value['tipo'] == 'SIST':

                    if key == 'DP':

                        subs = set(value['dados']['NUMSUB'])
                        for idx in subs:
                            idx_sist = [self.sist[i].Codigo for i in range(len(self.sist))].index(idx)
                            self.sist[idx_sist].CargaT = value['dados'][value['dados']['NUMSUB'] == idx][['DI', 'HI', 'MI', 'CARGA']].reset_index(drop=True)

                    elif key == 'CI':

                        aux = [(int(value['dados'].loc[i, 'SS/BUS']), int(value['dados'].loc[i, 'NUM'])) for i in value['dados'].index]
                        sub_num = set(aux)
                        for x in sub_num:
                            idx_sist = [self.sist[x].Codigo for x in range(len(self.sist))].index(x[0])
                            self.sist[idx_sist].ContratoImportacaoT[x[1]] = self.conf.DiscTemporalT[['DI', 'HI', 'MI']].copy()
                            self.sist[idx_sist].ContratoImportacaoT[x[1]]['LimInf'] = np.nan
                            self.sist[idx_sist].ContratoImportacaoT[x[1]]['LimSup'] = np.nan
                            self.sist[idx_sist].ContratoImportacaoT[x[1]]['Custo'] = np.nan
                            self.sist[idx_sist].ContratoImportacaoT[x[1]]['EnergIni'] = np.nan

                        for idx in value['dados'].index:
                            idx_sist = [self.sist[x].Codigo for x in range(len(self.sist))].index(value['dados'].loc[idx, 'SS/BUS'])
                            num = value['dados'].loc[idx, 'NUM']
                            di = self.sist[idx_sist].ContratoImportacaoT[num].loc[0, 'DI'] if value['dados'].loc[idx, 'DI'].strip() == 'I' else int(value['dados'].loc[idx, 'DI'])
                            hi = self.sist[idx_sist].ContratoImportacaoT[num].loc[0, 'HI'] if np.isnan(value['dados'].loc[idx, 'HI']) else int(value['dados'].loc[idx, 'HI'])
                            mi = self.sist[idx_sist].ContratoImportacaoT[num].loc[0, 'MI'] if np.isnan(value['dados'].loc[idx, 'MI']) else int(value['dados'].loc[idx, 'MI'])
                            linf = float(value['dados'].loc[idx, 'LINF'])
                            lsup = float(value['dados'].loc[idx, 'LSUP'])
                            custo = float(value['dados'].loc[idx, 'CUSTO'])
                            ini = None if np.isnan(value['dados'].loc[idx, 'INI']) else float(value['dados'].loc[idx, 'INI'])

                            self.sist[idx_sist].ContratoImportacaoT[num].loc[(self.sist[idx_sist].ContratoImportacaoT[num]['DI'] == di) &
                                                                            (self.sist[idx_sist].ContratoImportacaoT[num]['HI'] == hi) &
                                                                            (self.sist[idx_sist].ContratoImportacaoT[num]['MI'] == mi), ['LimInf', 'LimSup', 'Custo', 'EnergIni']] = [linf, lsup, custo, ini]

                    elif key == 'CE':

                        aux = [(int(value['dados'].loc[i, 'SS/BUS']), int(value['dados'].loc[i, 'NUM'])) for i in value['dados'].index]
                        sub_num = set(aux)
                        for x in sub_num:
                            idx_sist = [self.sist[x].Codigo for x in range(len(self.sist))].index(x[0])
                            self.sist[idx_sist].ContratoExportacaoT[x[1]] = self.conf.DiscTemporalT[['DI', 'HI', 'MI']].copy()
                            self.sist[idx_sist].ContratoExportacaoT[x[1]]['LimInf'] = np.nan
                            self.sist[idx_sist].ContratoExportacaoT[x[1]]['LimSup'] = np.nan
                            self.sist[idx_sist].ContratoExportacaoT[x[1]]['Custo'] = np.nan
                            self.sist[idx_sist].ContratoExportacaoT[x[1]]['EnergIni'] = np.nan

                        for idx in value['dados'].index:
                            idx_sist = [self.sist[x].Codigo for x in range(len(self.sist))].index(value['dados'].loc[idx, 'SS/BUS'])
                            num = value['dados'].loc[idx, 'NUM']
                            di = self.sist[idx_sist].ContratoExportacaoT[num].loc[0, 'DI'] if value['dados'].loc[idx, 'DI'].strip() == 'I' else int(value['dados'].loc[idx, 'DI'])
                            hi = self.sist[idx_sist].ContratoExportacaoT[num].loc[0, 'HI'] if np.isnan(value['dados'].loc[idx, 'HI']) else int(value['dados'].loc[idx, 'HI'])
                            mi = self.sist[idx_sist].ContratoExportacaoT[num].loc[0, 'MI'] if np.isnan(value['dados'].loc[idx, 'MI']) else int(value['dados'].loc[idx, 'MI'])
                            linf = float(value['dados'].loc[idx, 'LINF'])
                            lsup = float(value['dados'].loc[idx, 'LSUP'])
                            custo = float(value['dados'].loc[idx, 'CUSTO'])
                            ini = None if np.isnan(value['dados'].loc[idx, 'INI']) else float(value['dados'].loc[idx, 'INI'])

                            self.sist[idx_sist].ContratoExportacaoT[num].loc[(self.sist[idx_sist].ContratoExportacaoT[num]['DI'] == di) &
                                                                            (self.sist[idx_sist].ContratoExportacaoT[num]['HI'] == hi) &
                                                                            (self.sist[idx_sist].ContratoExportacaoT[num]['MI'] == mi), ['LimInf', 'LimSup', 'Custo', 'EnergIni']] = [linf, lsup, custo, ini]

                    # Pode ser um dado temporal, mas o deck vem um valor fixo
                    elif key == 'CD':

                        for idx in value['dados'].index:
                            idx_sist = [self.sist[i].Codigo for i in range(len(self.sist))].index(value['dados'].loc[idx, 'NUMSUB'])
                            self.sist[idx_sist].CustoDeficit = float(value['dados'].loc[idx, 'CDEF'])

        # Forward fill hidr values
        for idx_hidr, uhe in enumerate(all_hidr):
            if uhe.Codigo in sorted(set(vazoes['NUM'])):
                all_hidr[idx_hidr].VazIncPrevistaT = all_hidr[idx_hidr].VazIncPrevistaT.ffill(axis=0)
            all_hidr[idx_hidr].VolumeEsperaT = all_hidr[idx_hidr].VolumeEsperaT.fillna(100.)
            all_hidr[idx_hidr].DesvioAguaT = all_hidr[idx_hidr].DesvioAguaT.ffill(axis=0)
            try:
                mont_list = list(all_hidr[idx_hidr].DefAntMontanteT.keys())
                for mont in mont_list:
                    all_hidr[idx_hidr].DefAntMontanteT[mont]['Defl'] = all_hidr[idx_hidr].DefAntMontanteT[mont]['Defl'].ffill(axis=0).sort_index(ascending=False)
                    all_hidr[idx_hidr].DefAntMontanteT[mont]['Defl'].index = range(0, all_hidr[idx_hidr].DefAntMontanteT[mont]['Defl'].shape[0])
            except:
                pass

            try:
                for var in all_hidr[idx_hidr].RestricaoOperacaoT.keys():
                    all_hidr[idx_hidr].RestricaoOperacaoT[var]['Restricao'] = all_hidr[idx_hidr].RestricaoOperacaoT[var]['Restricao'].ffill(axis=0)
            except:
                pass

        for uhe in all_hidr:
            if uhe.Status == 'EX':
                self.hidr.append(uhe)

        # Forward fill term values
        for idx_term, ute in enumerate(self.term):
            self.term[idx_term].RestricaoGeracaoT = self.term[idx_term].RestricaoGeracaoT.ffill(axis=0)
            self.term[idx_term].GeracaoFixaT = self.term[idx_term].GeracaoFixaT.ffill(axis=0)
            for k, v in self.term[idx_term].UnidadeGeradoraT.items():
                self.term[idx_term].UnidadeGeradoraT[k]['Operacao'] = self.term[idx_term].UnidadeGeradoraT[k]['Operacao'].ffill(axis=0)

        # Forward fill sist values
        for idx_sist, sis in enumerate(self.sist):
            if self.sist[idx_sist].Sigla != 'FC':
                for k in self.sist[idx_sist].ContratoImportacaoT.keys():
                    self.sist[idx_sist].ContratoImportacaoT[k] = self.sist[idx_sist].ContratoImportacaoT[k].ffill(axis=0)
                for k in self.sist[idx_sist].ContratoExportacaoT.keys():
                    self.sist[idx_sist].ContratoExportacaoT[k] = self.sist[idx_sist].ContratoExportacaoT[k].ffill(axis=0)
            self.sist[idx_sist].CargaT = self.sist[idx_sist].CargaT.ffill(axis=0)

        # Forward fill interc values
        for idx_interc, intercambio in enumerate(self.interc):
            self.interc[idx_interc].LimiteT = self.interc[idx_interc].LimiteT.ffill(axis=0)

    # endregion

    # region Read files
    def management_read_files(self) -> tuple:

        for i, file in enumerate(self.arquivos):
            file_name = file['arquivo']

            if file_name == 'dadvaz.dat':
                init_date, config_vazoes, vazoes = self.read_dadvaz_file(name_file=file_name)

            elif file_name == 'entdados.dat':
                self.read_entdados_file(name_file=file_name)

            elif file_name == 'mapcut.dat':
                pass

            elif file_name == 'cortdeco.dat':
                pass

            elif file_name == 'hidr.dat':
                hidr_data = self.read_hidr_file(name_file=file_name)

            elif file_name == 'operuh.dat':
                self.read_operuh_file(name_file=file_name)

            elif file_name == 'deflant.dat':
                self.read_deflant_file(name_file=file_name)

            elif file_name == 'termdat.dat':
                self.read_termdat_file(name_file=file_name)

            elif file_name == 'operut.dat':
                self.read_operut_file(name_file=file_name)

            elif file_name == 'desselet.dat':
                pass

            elif file_name == 'ils_tri.dat':
                pass

            elif file_name == 'areacont.dat':
                self.read_areacont_file(name_file=file_name)

            elif file_name == 'respot.dat':
                self.read_respot_file(name_file=file_name)

            elif file_name == 'mlt.dat':
                pass

            elif file_name == 'ptoper.dat':
                self.read_ptoper_file(name_file=file_name)

            elif file_name == 'infofcf.dat':
                pass

            elif file_name == 'metas.dat':
                pass

            elif file_name == 'renovaveis.dat':
                self.read_renovaveis_file(name_file=file_name)

            elif file_name == 'rampas.dat':
                self.read_rampas_file(name_file=file_name)

            elif file_name == 'rstlpp.dat':
                self.read_restlpp_file(name_file=file_name)

            elif file_name == 'restseg.dat':
                self.read_restseg_file(name_file=file_name)

            elif file_name == 'respotele.dat':
                pass

            elif file_name == 'renovaveis.dat':
                pass

        return hidr_data, vazoes, init_date

    def read_dessem_arq_file(self, name_file: str) -> List[dict]:

        path_file = os.path.join(self.diretorio, name_file)

        output = list()
        with open(path_file) as file:
            lines = file.readlines()
            self.titulo = lines[3][50:]
            for line in lines[4:]:
                aux = dict()
                if line[0] != '&':
                    aux['nome'] = line[:10].strip()
                    aux['descricao'] = line[10:49].strip()
                    aux['arquivo'] = line[49:].strip()

                    output.append(aux)

        file.close()

        return output

    def read_dadvaz_file(self, name_file: str) -> tuple:

        path_file = os.path.join(self.diretorio, name_file)

        with open(path_file) as file:
            lines = file.readlines()

            # uhe_list = [int(x) for x in lines[6].split()]
            # vazoes = {k: dict() for k in uhe_list}

            hour, day, month, year = [int(x) for x in lines[9].split()]
            init_date = datetime(year, month, day, hour)
            weekday, semFCF, nr_weeks, pre_interesse = [int(x) for x in lines[12].split()]
            config_vazoes = {'diaSemana': weekday, 'semFCF': semFCF, 'numSemanas': nr_weeks, 'preInteresse': pre_interesse}

            cont = 16
            for line in lines[cont:]:
                if line.strip() == 'FIM':
                    break
                cont += 1

            df_lines = lines[16:cont]
            df_lines_nocoment = ''.join([x for x in df_lines if x[0] != '&'])
            names = ['NUM', 'NOME', 'ITP', 'DI', 'HI', 'MI', 'DF', 'HF', 'MF', 'VAZAO']
            colspecs = [(0, 3), (4, 16), (19, 20), (24, 26), (27, 29), (30, 31), (32, 34), (35, 37), (38, 39), (44, 53)]

            with open(NamedTemporaryFile().name, 'w+') as f:
                print(df_lines_nocoment, file=f)
            f.close()

            vazoes = pd.read_fwf(f.name, names=names, colspecs=colspecs)
            vazoes = vazoes.dropna(axis=0, how='all')
            vazoes = vazoes.fillna(0).astype({'NUM': int, 'ITP': int, 'DI': int, 'HI': int, 'MI': int,
                                              'HF': int, 'MF': int, 'VAZAO': float})
            return init_date, config_vazoes, vazoes

    def read_entdados_file(self, name_file: str) -> None:

        path_file = os.path.join(self.diretorio, name_file)

        with open(path_file) as file:
            lines = file.readlines()

            for registro, config in config_entdados.items():

                lines_registro = ''.join([x for x in lines if x[:config['colspecs'][0][1]].strip() == registro])

                with open(NamedTemporaryFile().name, 'w+') as f:
                    print(lines_registro, file=f)
                f.close()

                config['dados'] = pd.read_fwf(f.name, names=list(config['colunas'].keys()), colspecs=config['colspecs'])
                config['dados'] = config['dados'].dropna(axis=0, how='all')
                fill_values = {list(config['colunas'].keys())[i]: config['default'][i] for i in range(len(config['default'])) if config['default'][i]}
                config['dados'] = config['dados'].fillna(value=fill_values) # .astype(config['colunas'])

    def read_ptoper_file(self, name_file: str) -> None:

        path_file = os.path.join(self.diretorio, name_file)

        with open(path_file) as file:
            lines = file.readlines()

            for registro, config in config_ptoper.items():

                lines_registro = ''.join([x for x in lines if x[:config['colspecs'][0][1]].strip() == registro])

                with open(NamedTemporaryFile().name, 'w+') as f:
                    print(lines_registro, file=f)
                f.close()

                config['dados'] = pd.read_fwf(f.name, names=list(config['colunas'].keys()), colspecs=config['colspecs'])
                config['dados'] = config['dados'].dropna(axis=0, how='all')
                fill_values = {list(config['colunas'].keys())[i]: config['default'][i] for i in range(len(config['default'])) if config['default'][i]}
                config['dados'] = config['dados'].fillna(value=fill_values)  # .astype(config['colunas'])

    def read_hidr_file(self, name_file: str):

        path_file = os.path.join(self.diretorio, name_file)
        file = open(path_file, "rb")
        nreg = 320

        cadastro = []

        i = 0
        while i < nreg:
            cadastro.append(hidr())
            iusi = len(cadastro) - 1
            cadastro[iusi].Codigo = i+1
            cadastro[iusi].Nome = struct.unpack('12s', file.read(12))[0].strip().decode('utf-8')
            cadastro[iusi].Posto = struct.unpack('i', file.read(4))[0]
            cadastro[iusi].Bdh = struct.unpack('8s', file.read(8))[0]
            cadastro[iusi].Sist = struct.unpack('i', file.read(4))[0]
            cadastro[iusi].Empr = struct.unpack('i', file.read(4))[0]
            cadastro[iusi].Jusante = struct.unpack('i', file.read(4))[0]
            cadastro[iusi].Desvio = struct.unpack('i', file.read(4))[0]
            cadastro[iusi].VolMin = struct.unpack('f', file.read(4))[0]
            cadastro[iusi].VolMax = struct.unpack('f', file.read(4))[0]
            cadastro[iusi].VolMinVert = struct.unpack('f', file.read(4))[0]
            cadastro[iusi].VolMinDesv = struct.unpack('f', file.read(4))[0]
            cadastro[iusi].CotaMin = struct.unpack('f', file.read(4))[0]
            cadastro[iusi].CotaMax = struct.unpack('f', file.read(4))[0]
            cadastro[iusi].PolCotaVol = list(struct.unpack('5f', bytearray(file.read(20))))
            cadastro[iusi].PolCotaArea = list(struct.unpack('5f', bytearray(file.read(20))))
            cadastro[iusi].CoefEvapMensal = list(struct.unpack('12i', bytearray(file.read(48))))
            cadastro[iusi].NumConjMaq = struct.unpack('i', file.read(4))[0]
            cadastro[iusi].MaqporConj = list(struct.unpack('5i', bytearray(file.read(20))))
            cadastro[iusi].PEfporConj = list(struct.unpack('5f', bytearray(file.read(20))))

            cadastro[iusi].CF_HBQT = []
            cadastro[iusi].CF_HBQG = []
            cadastro[iusi].CF_HBPT = []
            for j in range(5):
                cadastro[iusi].CF_HBQT.append(list(struct.unpack('5f', bytearray(file.read(20)))))
            for j in range(5):
                cadastro[iusi].CF_HBQG.append(list(struct.unpack('5f', bytearray(file.read(20)))))
            for j in range(5):
                cadastro[iusi].CF_HBPT.append(list(struct.unpack('5f', bytearray(file.read(20)))))

            cadastro[iusi].AltEfetConj = list(struct.unpack('5f', bytearray(file.read(20))))
            cadastro[iusi].VazEfetConj = list(struct.unpack('5i', bytearray(file.read(20))))
            cadastro[iusi].ProdEsp = struct.unpack('f', file.read(4))[0]
            cadastro[iusi].PerdaHid = struct.unpack('f', file.read(4))[0]
            cadastro[iusi].NumPolVNJ = struct.unpack('i', file.read(4))[0]

            cadastro[iusi].PolVazNivJus = []
            for j in range(5):
                cadastro[iusi].PolVazNivJus.append(list(struct.unpack('6f', bytearray(file.read(24)))))

            cadastro[iusi].CotaRefNivelJus = list(struct.unpack('6f', bytearray(file.read(24))))
            cadastro[iusi].CFMed = struct.unpack('f', file.read(4))[0]
            cadastro[iusi].InfVertCanalFuga = struct.unpack('i', file.read(4))[0]
            cadastro[iusi].FatorCargaMax = struct.unpack('f', file.read(4))[0]
            cadastro[iusi].FatorCargaMin = struct.unpack('f', file.read(4))[0]
            cadastro[iusi].VazMin = struct.unpack('i', file.read(4))[0]
            cadastro[iusi].UnidBase = struct.unpack('i', file.read(4))[0]
            cadastro[iusi].TipoTurb = struct.unpack('i', file.read(4))[0]
            cadastro[iusi].RepresConj = struct.unpack('i', file.read(4))[0]
            cadastro[iusi].TEIF = struct.unpack('f', file.read(4))[0]
            cadastro[iusi].IP = struct.unpack('f', file.read(4))[0]
            cadastro[iusi].TipoPerda = struct.unpack('i', file.read(4))[0]
            cadastro[iusi].Data = struct.unpack('8s', file.read(8))[0].strip().decode('utf-8')
            cadastro[iusi].Observ = struct.unpack('43s', file.read(43))[0]
            cadastro[iusi].VolRef = struct.unpack('f', file.read(4))[0]
            cadastro[iusi].TipoReg = struct.unpack('c', file.read(1))[0]

            i = i + 1

        file.close()

        return cadastro

    def read_operuh_file(self, name_file: str) -> None:

        path_file = os.path.join(self.diretorio, name_file)

        with open(path_file,  errors='ignore') as file:
            lines = file.readlines()

            for registro, config in config_operuh.items():

                lines_registro = ''.join([x for x in lines if x[:config['colspecs'][0][1]].strip() == registro])

                with open(NamedTemporaryFile().name, 'w+') as f:
                    print(lines_registro, file=f)
                f.close()

                config['dados'] = pd.read_fwf(f.name, names=list(config['colunas'].keys()), colspecs=config['colspecs'])
                config['dados'] = config['dados'].dropna(axis=0, how='all')
                fill_values = {list(config['colunas'].keys())[i]: config['default'][i] for i in range(len(config['default'])) if config['default'][i]}
                config['dados'] = config['dados'].fillna(value=fill_values) # .astype(config['colunas'])

    def read_termdat_file(self, name_file: str) -> None:

        path_file = os.path.join(self.diretorio, name_file)

        with open(path_file) as file:
            lines = file.readlines()

            for registro, config in config_termdat.items():

                lines_registro = ''.join([x for x in lines if x[:config['colspecs'][0][1]].strip() == registro])

                with open(NamedTemporaryFile().name, 'w+') as f:
                    print(lines_registro, file=f)
                f.close()

                config['dados'] = pd.read_fwf(f.name, names=list(config['colunas'].keys()), colspecs=config['colspecs'])
                config['dados'] = config['dados'].dropna(axis=0, how='all')
                fill_values = {list(config['colunas'].keys())[i]: config['default'][i] for i in range(len(config['default'])) if config['default'][i]}
                config['dados'] = config['dados'].fillna(value=fill_values)  # .astype(config['colunas'])

    def read_operut_file(self, name_file: str) -> None:

        path_file = os.path.join(self.diretorio, name_file)

        with open(path_file) as file:
            lines = file.readlines()

            for registro, config in config_operut.items():

                if registro == 'INIT' or registro == 'OPER':

                    pos_init = [i for i, x in enumerate(lines) if x.strip() == registro][0]
                    for i, x in enumerate(lines[pos_init:]):
                        if x.strip() == 'FIM':
                            pos_final = pos_init + i
                            break

                    lines_registro = ''.join([x for x in lines[pos_init+1:pos_final] if x[0] != '&'])

                    with open(NamedTemporaryFile().name, 'w+') as f:
                        print(lines_registro, file=f)
                    f.close()

                    config['dados'] = pd.read_fwf(f.name, names=list(config['colunas'].keys()), colspecs=config['colspecs'])
                    config['dados'] = config['dados'].dropna(axis=0, how='all')
                    fill_values = {list(config['colunas'].keys())[i]: config['default'][i] for i in
                                   range(len(config['default'])) if config['default'][i]}
                    config['dados'] = config['dados'].fillna(value=fill_values)  # .astype(config['colunas'])
                    config['dados'].insert(loc=0, column='MNE', value=registro)

                else:

                    lines_registro = ''.join([x for x in lines if x[:config['colspecs'][0][1]].strip() == registro])

                    with open(NamedTemporaryFile().name, 'w+') as f:
                        print(lines_registro, file=f)
                    f.close()

                    config['dados'] = pd.read_fwf(f.name, names=list(config['colunas'].keys()), colspecs=config['colspecs'])
                    config['dados'] = config['dados'].dropna(axis=0, how='all')
                    fill_values = {list(config['colunas'].keys())[i]: config['default'][i] for i in range(len(config['default'])) if config['default'][i]}
                    config['dados'] = config['dados'].fillna(value=fill_values)  # .astype(config['colunas'])

    def read_areacont_file(self, name_file: str) -> None:

        path_file = os.path.join(self.diretorio, name_file)

        with open(path_file) as file:
            lines = file.readlines()

            for registro, config in config_areacont.items():

                pos_init = [i for i, x in enumerate(lines) if x.strip() == registro][0]
                for i, x in enumerate(lines[pos_init:]):
                    if x.strip() == 'FIM':
                        pos_final = pos_init + i
                        break

                lines_registro = ''.join([x for x in lines[pos_init+1:pos_final] if x[0] != '&'])

                with open(NamedTemporaryFile().name, 'w+') as f:
                    print(lines_registro, file=f)
                f.close()

                config['dados'] = pd.read_fwf(f.name, names=list(config['colunas'].keys()), colspecs=config['colspecs'])
                config['dados'] = config['dados'].dropna(axis=0, how='all')
                fill_values = {list(config['colunas'].keys())[i]: config['default'][i] for i in
                               range(len(config['default'])) if config['default'][i]}
                config['dados'] = config['dados'].fillna(value=fill_values)  # .astype(config['colunas'])
                config['dados'].insert(loc=0, column='MNE', value=registro)

    def read_respot_file(self, name_file: str) -> None:

        path_file = os.path.join(self.diretorio, name_file)

        with open(path_file) as file:
            lines = file.readlines()

            for registro, config in config_respot.items():

                lines_registro = ''.join([x for x in lines if x[:config['colspecs'][0][1]].strip() == registro])

                with open(NamedTemporaryFile().name, 'w+') as f:
                    print(lines_registro, file=f)
                f.close()

                config['dados'] = pd.read_fwf(f.name, names=list(config['colunas'].keys()), colspecs=config['colspecs'])
                config['dados'] = config['dados'].dropna(axis=0, how='all')
                fill_values = {list(config['colunas'].keys())[i]: config['default'][i] for i in range(len(config['default'])) if config['default'][i]}
                config['dados'] = config['dados'].fillna(value=fill_values)  # .astype(config['colunas'])

    def read_deflant_file(self, name_file: str) -> None:

        path_file = os.path.join(self.diretorio, name_file)

        with open(path_file) as file:
            lines = file.readlines()

            for registro, config in config_deflant.items():

                lines_registro = ''.join([x for x in lines if x[:config['colspecs'][0][1]].strip() == registro])

                with open(NamedTemporaryFile().name, 'w+') as f:
                    print(lines_registro, file=f)
                f.close()

                config['dados'] = pd.read_fwf(f.name, names=list(config['colunas'].keys()), colspecs=config['colspecs'])
                config['dados'] = config['dados'].dropna(axis=0, how='all')
                fill_values = {list(config['colunas'].keys())[i]: config['default'][i] for i in range(len(config['default'])) if config['default'][i]}
                config['dados'] = config['dados'].fillna(value=fill_values)  # .astype(config['colunas'])

    def read_restseg_file(self, name_file: str) -> None:

        path_file = os.path.join(self.diretorio, name_file)

        with open(path_file) as file:
            lines = file.readlines()

            for registro, config in config_restseg.items():

                lines_registro = ''.join([x for x in lines if x[:config['colspecs'][0][1]].strip() == registro])

                with open(NamedTemporaryFile().name, 'w+') as f:
                    print(lines_registro, file=f)
                f.close()

                config['dados'] = pd.read_fwf(f.name, names=list(config['colunas'].keys()), colspecs=config['colspecs'])
                config['dados'] = config['dados'].dropna(axis=0, how='all')
                fill_values = {list(config['colunas'].keys())[i]: config['default'][i] for i in range(len(config['default'])) if config['default'][i]}
                config['dados'] = config['dados'].fillna(value=fill_values)  # .astype(config['colunas'])

    def read_restlpp_file(self, name_file: str) -> None:

        path_file = os.path.join(self.diretorio, name_file)

        with open(path_file) as file:
            lines = file.readlines()

            for registro, config in config_rstlpp.items():

                lines_registro = ''.join([x for x in lines if x[:config['colspecs'][0][1]].strip() == registro])

                with open(NamedTemporaryFile().name, 'w+') as f:
                    print(lines_registro, file=f)
                f.close()

                config['dados'] = pd.read_fwf(f.name, names=list(config['colunas'].keys()), colspecs=config['colspecs'])
                config['dados'] = config['dados'].dropna(axis=0, how='all')
                fill_values = {list(config['colunas'].keys())[i]: config['default'][i] for i in range(len(config['default'])) if config['default'][i]}
                config['dados'] = config['dados'].fillna(value=fill_values)  # .astype(config['colunas'])

    def read_rampas_file(self, name_file: str) -> None:

        path_file = os.path.join(self.diretorio, name_file)

        with open(path_file) as file:
            lines = file.readlines()

            for registro, config in config_rampas.items():

                if registro == 'RAMP':

                    pos_init = [i for i, x in enumerate(lines) if x.strip() == registro][0]
                    for i, x in enumerate(lines[pos_init:]):
                        if x.strip() == 'FIM':
                            pos_final = pos_init + i
                            break

                    lines_registro = ''.join([x for x in lines[pos_init+1:pos_final] if x[0] != '&'])

                    with open(NamedTemporaryFile().name, 'w+') as f:
                        print(lines_registro, file=f)
                    f.close()

                    config['dados'] = pd.read_fwf(f.name, names=list(config['colunas'].keys()), colspecs=config['colspecs'])
                    config['dados'] = config['dados'].dropna(axis=0, how='all')
                    fill_values = {list(config['colunas'].keys())[i]: config['default'][i] for i in
                                   range(len(config['default'])) if config['default'][i]}
                    config['dados'] = config['dados'].fillna(value=fill_values)  # .astype(config['colunas'])
                    config['dados'].insert(loc=0, column='MNE', value=registro)

    def read_renovaveis_file(self, name_file: str) -> None:

        path_file = os.path.join(self.diretorio, name_file)

        with open(path_file) as file:
            lines = file.readlines()

            for registro, config in config_renovaveis.items():

                lines_registro = ''.join([x for x in lines if x.split(';')[0].strip() == registro])

                with open(NamedTemporaryFile().name, 'w+') as f:
                    print(lines_registro, file=f)
                f.close()

                config['dados'] = pd.read_csv(f.name, names=list(config['colunas'].keys()), sep=';')
                config['dados'] = config['dados'].dropna(axis=0, how='all')
                fill_values = {list(config['colunas'].keys())[i]: config['default'][i] for i in range(len(config['default'])) if config['default'][i]}
                config['dados'] = config['dados'].fillna(value=fill_values)  # .astype(config['colunas'])
    # endregion