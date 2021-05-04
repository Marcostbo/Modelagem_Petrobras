from Modules.NewaveFcf.cortes import Cortes
import os
import pandas as pd
import numpy as np


def convert_EARM_hydro(hidr, coef, indep) -> pd.DataFrame:

    columns = [1, 6, 7, 5, 10, 12, 2, 11, 3, 4, 8, 9]

    coef = pd.DataFrame(data=np.round(coef, 2), columns=columns)
    indep = pd.DataFrame(data=np.round(indep, 2).reshape((len(indep), 1)), columns=['indep'])
    data = pd.concat([coef, indep], axis=1)
    data.drop_duplicates(inplace=True)

    new_data = data.loc[~(data == 0).all(axis=1)]

    ree_codes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    fcf_NW = pd.DataFrame(data=[])
    earm_max_ree = dict()
    for ree in ree_codes:
        earm_max_ree[ree] = sum([x.EarmMax for x in hidr if x.Ree == ree])

    for idx, uhe in enumerate(hidr):
        ree = uhe.Ree
        fator = uhe.EarmMax/earm_max_ree[ree]
        fcf_NW[idx] = fator*new_data[ree]

    fcf_NW['indep'] = new_data['indep']

    fcf_NW = fcf_NW.round(2)
    fcf_NW.drop_duplicates(inplace=True, ignore_index=True)

    return fcf_NW


def main(hidr, month: int):

    cortes_path = os.path.join(os.environ['DESSEM_DECK_PATH'], 'cortes_FCF_NW', 'cortes.dat')
    cortesh_path = os.path.join(os.environ['DESSEM_DECK_PATH'], 'cortes_FCF_NW', 'cortesh.dat')

    cortes = Cortes(cortes_path, cortesh_path)
    cortes.leitura(periodo=month)

    # print("Termo Independente: {}".format(cortes.rhs[1]))
    # print("Coeficiente Cortes EARM: \n{}".format(cortes.piv[1]))
    # print("Coeficiente Cortes EAF:  \n{}".format(cortes.pih[1].reshape([cortes.cortesh.nsis, cortes.parp_max])))
    # print("Coeficiente Cortes GNL : \n{}".format(cortes.pig[1].reshape([cortes.cortesh.nsbm, cortes.parp_max])))

    fcf_NW = convert_EARM_hydro(hidr=hidr, coef=cortes.piv, indep=cortes.rhs)

    return fcf_NW
