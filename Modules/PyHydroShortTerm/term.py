import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta as delta


class term(object):

    Codigo = None
    Nome = None
    Sist = None
    DataInicioOper = None
    NumUnidades = None
    UnidadeGeradoraT = None
    RestricaoGeracaoT = None
    ManutencaoT = None
    GeracaoFixaT = None

    # Thermal plants parameters calculus
    def parameters_calculus_run(self, data, init_date: datetime) -> list:

        for i, ute in enumerate(data):

            for idx, unid in enumerate(ute.UnidadeGeradoraT):

                data[i].UnidadeGeradoraT[unid]['CapacidadeT'] = self.maintenance_parameters_calculus(data=data[i], unid=unid, ref_date=init_date)

        return data

    def maintenance_parameters_calculus(self, data, unid: int, ref_date: datetime) -> pd.DataFrame:

        capacidade = data.UnidadeGeradoraT[unid]['Operacao'][['DI', 'HI', 'MI']].copy()

        for idx in capacidade.index:

            value_capacidade = data.UnidadeGeradoraT[unid]['Capacidade']

            for i in data.ManutencaoT.index:

                # Interval Maintenance
                if data.ManutencaoT.loc[i, 'DI'] < capacidade.loc[0, 'DI']:
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
                if data.ManutencaoT.loc[i, 'DF'] < capacidade.loc[0, 'DI']:
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
                if capacidade.loc[idx, 'DI'] < capacidade.loc[0, 'DI']:
                    new_month = ref_date.month + 1
                    new_year = ref_date.year
                    if new_month == 13:
                        new_month = 1
                        new_year = new_year + 1
                    init_date = datetime(new_year, new_month, int(capacidade.loc[idx, 'DI']),
                                         int(capacidade.loc[idx, 'HI']), int(capacidade.loc[idx, 'MI']*30))
                else:
                    init_date = datetime(ref_date.year, ref_date.month, int(capacidade.loc[idx, 'DI']),
                                         int(capacidade.loc[idx, 'HI']), int(capacidade.loc[idx, 'MI']*30))

                if idx < capacidade.index[-1]:
                    if capacidade.loc[idx+1, 'DI'] < capacidade.loc[0, 'DI']:
                        new_month = ref_date.month + 1
                        new_year = ref_date.year
                        if new_month == 13:
                            new_month = 1
                            new_year = new_year + 1
                        end_date = datetime(new_year, new_month, int(capacidade.loc[idx+1, 'DI']),
                                            int(capacidade.loc[idx+1, 'HI']), int(capacidade.loc[idx+1, 'MI']*30))
                    else:
                        end_date = datetime(ref_date.year, ref_date.month, int(capacidade.loc[idx+1, 'DI']),
                                            int(capacidade.loc[idx+1, 'HI']), int(capacidade.loc[idx+1, 'MI']*30))
                else:
                    if capacidade.loc[idx, 'DI'] < capacidade.loc[0, 'DI']:
                        new_month = ref_date.month + 1
                        new_year = ref_date.year
                        if new_month == 13:
                            new_month = 1
                            new_year = new_year + 1
                        end_date = datetime(new_year, new_month, int(capacidade.loc[idx, 'DI']),
                                            23, 30)
                    else:
                        end_date = datetime(ref_date.year, ref_date.month, int(capacidade.loc[idx, 'DI']),
                                            23, 30)

                length = int((end_date - init_date).total_seconds() / 60 / 30)
                interval_stage = [init_date + delta(minutes=30*i) for i in range(length)]

                # Test
                condition = [True if x in interval_stage else False for x in interval_maintenance]

                true_count = sum(condition)

                if (true_count >= int(len(interval_stage) / 2)) and (int(data.ManutencaoT.loc[i, 'NUMGRUPO']) == unid):
                    value_capacidade = 0.

            capacidade.loc[idx, 'CAPACIDADE'] = value_capacidade

        return capacidade
