import numpy as np


class fcf(object):
    nr_cortes = 0
    coef_vol = list()
    termo_i = list()
    estagio = 0

    def __init__(self, estagio):
        self.nr_cortes = 0
        self.coef_vol = []
        self.termo_i = []
        self.estagio = estagio

    def calcAB(self, ncol, nr_var_hidr):
        if self.nr_cortes == 0:
            A = np.array([]).reshape(0, ncol)
            B = np.array([]).reshape(0, 1)
        else:
            nr_hidr = len(self.coef_vol[0])
            A = np.zeros((self.nr_cortes, ncol))
            B = np.zeros((self.nr_cortes, 1))
            for icor in range(self.nr_cortes):
                for i in range(nr_hidr):
                    A[icor, nr_var_hidr*i] = self.coef_vol[icor][i]
                    A[icor, -1] = -1
                B[icor, 0] = -self.termo_i[icor]

            AB = np.concatenate((A, B), axis=1)
            AB = np.around(AB, decimals=0)
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

    def geraFCF(self, vol, custo, inc_vol):
        self.nr_cortes += 1
        self.coef_vol.append(inc_vol)
        self.termo_i.append(custo - np.dot(inc_vol, vol))
