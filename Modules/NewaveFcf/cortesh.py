""" Modulo que contem a classe para leitura dos arquivos com os cortes de benders """
import numpy as np

# Tamanho de cada Registro do Arquivo de Cortes
TAM_REG = 11520


class CortesH(object):
    """ Classe para Leitura do Arquivo de Cabecalho dos Cortes """

    def __init__(self, file_name: str):
        """
        Inicializacao da Classe
        :param file_name: string com o caminho completo para o cortesh

        """
        self.file_name = file_name
        self.versao = ''  # Versão do modelo NEWAVE
        self.lrec = None  # Tamanho dos registros do arquivo de cortes
        self.lrece = None  # Tamanho dos registros do arquivo de estados
        self.nsis = None  # número de REEs
        self.npre = None  # número de períodos do estático inicial
        self.nper = None  # número de períodos de planejamento
        self.npst = None  # número de períodos do estático final
        self.npea = None  #
        self.nconf = None  # número de configurações
        self.nsim = None  # número de simulações forward
        self.npmc = None  # número de patamares de carga
        self.anoi = None  # ano inicial do período de planejamento
        self.mesi = None  # mês inicial do período de planejamento
        self.lagmax = None  # lag máximo adotada para o despacho antecipado
        self.mecaver = None  # Tipo de mecanismo de aversão adotado
        self.nsbm = None  # número de subsistemas / submercados
        self.nnsbm = None  # Número total de subsistemas/submercados (reais e fictícios)

        self.iptreg = []  # Numero do ultimo registro de cortes de cada periodo
        self.mord = []  # Ordem do processo PARP escolhido para cada REE, periodo e configuracao
        self.pconf = []  # Configuracao válida para cada periodo
        self.fpeng = []  # Duracao do Patamar

    def leitura(self):
        """ Leitura do Arquivo de Cabecalho dos Cortes """

        with open(self.file_name, 'rb') as f:

            # Leitura das informacoes do Primeiro Registro
            self.versao = np.fromfile(f, dtype=np.dtype('i4'), count=1)[0]
            self.lrec = np.fromfile(f, dtype=np.dtype('i4'), count=1)[0]
            self.lrece = np.fromfile(f, dtype=np.dtype('i4'), count=1)[0]
            self.nsis = np.fromfile(f, dtype=np.dtype('i4'), count=1)[0]
            self.npre = np.fromfile(f, dtype=np.dtype('i4'), count=1)[0]
            self.nper = np.fromfile(f, dtype=np.dtype('i4'), count=1)[0]
            self.npst = np.fromfile(f, dtype=np.dtype('i4'), count=1)[0]
            self.npea = np.fromfile(f, dtype=np.dtype('i4'), count=1)[0]
            self.nconf = np.fromfile(f, dtype=np.dtype('i4'), count=1)[0]
            self.nsim = np.fromfile(f, dtype=np.dtype('i4'), count=1)[0]
            self.npmc = np.fromfile(f, dtype=np.dtype('i4'), count=1)[0]
            self.anoi = np.fromfile(f, dtype=np.dtype('i4'), count=1)[0]
            self.mesi = np.fromfile(f, dtype=np.dtype('i4'), count=1)[0]
            self.lagmax = np.fromfile(f, dtype=np.dtype('i4'), count=1)[0]
            self.mecaver = np.fromfile(f, dtype=np.dtype('i4'), count=1)[0]
            self.nsbm = np.fromfile(f, dtype=np.dtype('i4'), count=1)[0]
            self.nnsbm = np.fromfile(f, dtype=np.dtype('i4'), count=1)[0]
            # Pula para o proximo registro
            f.seek(TAM_REG * 4, 0)

            # Leitura das Informacoes do Segundo Registro
            # Numero do ultimo registro de cortes de cada periodo
            nreg = self.npre + self.nper + self.npst
            self.iptreg = np.fromfile(f, dtype=np.dtype('i4'), count=nreg)
            f.seek((TAM_REG - nreg) * 4, 1)

            # Leitura das Informacoes do Terceiro Registro
            # Ordem do processo PARP escolhido para cada REE, periodo e configuracao
            nreg = (self.nper + 2 * self.npea) * self.nsis
            self.mord = np.fromfile(f, dtype=np.dtype('i4'), count=nreg)
            f.seek((TAM_REG - nreg) * 4, 1)

            # Leitura das Informacoes do Quarto Registro
            # Configuracao válida para cada periodo
            nreg = self.npre + self.nper + self.npst
            self.pconf = np.fromfile(f, dtype=np.dtype('i4'), count=nreg)
            f.seek((TAM_REG - nreg) * 4, 1)

            # Leitura das Informacoes do Quinto Registro
            # Duracao do Patamar
            nreg = self.nper * self.npmc
            self.fpeng = np.fromfile(f, dtype=np.dtype('f8'), count=nreg)

        print("OK! Leitura do arquivo CORTESH.")
