""" Modulo que contem a classe para leitura do arquivo de cortes """

import numpy as np
from Modules.NewaveFcf import cortesh


class Cortes(object):
    """ Classe Cortes para leitura dos Cortes de Benders """

    def __init__(self, file_name_cortes: str, file_name_cortesh: str):
        """
        Inicializacao da classe para leitura do arquivo de cortes

        :param file_name_cortes: caminho completo para o arquivo de cortes
        :param file_name_cortesh: caminho completo para o arquivo de cabeçalho

        """

        self.file_name_cortes = file_name_cortes
        self.file_name_cortesh = file_name_cortesh

        # Inicializa a classe para leitura do cabeçalho
        self.cortesh = cortesh.CortesH(self.file_name_cortesh)
        self.cortesh.leitura()

        # Ordem Máxima do PARP
        self.parp_max = int(np.max(self.cortesh.mord))

        # Numero de períodos que possuem cortes registrados no arquivo de cortes
        # numero de períodos pre + numero de periodos de planejamento + numero de periodos pos
        # - mes inicial do periodo de planejamento - 1
        # O ultimo período do estudo não possui cortes (funcao de custo futuro)
        self.periodos = self.cortesh.npre + self.cortesh.nper + self.cortesh.npst - self.cortesh.mesi

        self.num_reg = self.periodos * self.cortesh.nsim * 24

        # Estrutura para Armazenamento dos Cortes
        self.ireg = np.zeros([self.num_reg], dtype=np.dtype('i4'))
        self.iter = np.zeros([self.num_reg], dtype=np.dtype('i4'))
        self.isim = np.zeros([self.num_reg], dtype=np.dtype('i4'))
        self.dummy = np.zeros([self.num_reg], dtype=np.dtype('i4'))
        self.rhs = np.zeros([self.num_reg], dtype=np.dtype('f8'))

        self.piv = np.zeros([self.num_reg, self.cortesh.nsis], dtype=np.dtype('f8'))
        self.pih = np.zeros(
            [self.num_reg, self.cortesh.nsis * self.parp_max], dtype=np.dtype('f8'))
        self.pig = np.zeros(
            [self.num_reg, self.cortesh.nsbm * self.cortesh.npmc * self.cortesh.lagmax], dtype=np.dtype('f8'))

    def leitura(self, periodo):
        """ Leitura do Arquivo de Cortes """
        if self.cortesh.mesi > periodo:
            raise ValueError()

        # cortes = None

        # Lê o arquivo de cortes para a memoria -- vetor numpy
        with open(self.file_name_cortes, 'rb') as f:

            # Registro para Leitura do Arquivo de COrtes
            tam_reg = int((self.cortesh.lrec - 4 * 4 - 8 * 1)/8)
            struct = np.dtype(
                [
                    ('ireg', 'i4', 1),
                    ('iter', 'i4', 1),
                    ('isim', 'i4', 1),
                    ('dummy', 'i4', 1),
                    ('rhs', 'f8', 1),
                    ('ccorte', 'f8', tam_reg)
                ]
            )

            # Leitura do arquivo de Cortes
            cortes = np.fromfile(f, dtype=struct)

        # Inicia p Mapeamento dos Cortes do Periodo
        last_cut = self.cortesh.iptreg[periodo-1]

        icut = -1
        while last_cut > 0:

            icut += 1
            # Proximo Corte
            next_cut = int(cortes['ireg'][last_cut-1])

            # Termo Independente
            self.rhs[icut] = cortes['rhs'][last_cut-1]

            # Coeficiente do Multiplicador de Lagrange das Restricoes de EARM
            ini = 0
            fim = self.cortesh.nsis
            self.piv[icut, :] = cortes['ccorte'][last_cut-1, ini:fim]

            # Coeficiente do Multiplicador de Lagrange das Restricoes de EAF
            ini = fim
            fim = ini + self.cortesh.nsis * self.parp_max
            self.pih[icut, :] = cortes['ccorte'][last_cut-1, ini:fim]

            # Coeficientes do Multiplicador de Lagrange das Restricoes de GNL
            ini = fim
            fim = ini + self.cortesh.npmc * self.cortesh.nsbm * self.cortesh.lagmax
            self.pig[icut, :] = cortes['ccorte'][last_cut-1, ini:fim]

            # Atualiza ultimo corte lido
            last_cut = next_cut
        print("OK! Leitura do arquivo CORTES.")
