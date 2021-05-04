import numpy as np


class fcf(object):
    nr_cortes = 0
    coef_ea = list()
    termo_i = list()
    estagio = 0

    def __init__(self, estagio):
        self.nr_cortes = 0
        self.coef_ea = []
        self.termo_i = []
        self.estagio = estagio

    def calcAB(self, Aeq):
        if self.nr_cortes == 0:
            A = None
            B = None
        else:
            ncol = Aeq.shape[1]
            nr_sist = len(self.coef_ea[0])
            A = np.zeros((self.nr_cortes, ncol))
            B = np.zeros((self.nr_cortes, 1))
            for icor in range(self.nr_cortes):
                for isist in range(nr_sist):
                    A[icor, isist] = self.coef_ea[icor][isist]
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

    def geraFCF(self, ea, custo, inc_ea):
        self.nr_cortes += 1
        self.coef_ea.append(inc_ea)
        self.termo_i.append(custo - np.dot(inc_ea, ea))
