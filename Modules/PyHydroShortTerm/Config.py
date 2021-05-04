# Associa o dia da semana a um número
number_to_weekdays = {1: 'Sab', 2: 'Dom', 3: 'Seg', 4: 'Ter', 5: 'Qua', 6: 'Qui', 7: 'Sex'}

# Configuração dos blocos existentes no arquivo "entdados.dat"
config_entdados = {
    'RIVAR':
        {'descricao': 'Restrições internas "soft" de variação para variáveis do problema',
         'tipo': 'CONFIG',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'PARA': int, 'COD': int, 'PENAL': float},
         'colspecs': [(0, 5), (7, 10), (11, 14), (15, 17), (19, 29)],
         'default': [None, None, None, None, 1e-8]
         },
    'TM':
        {'descricao': 'Discretização temporal do estudo e representação da rede elétrica',
         'tipo': 'CONFIG',
         'dados': None,
         'colunas': {'MNE': str, 'DI': int, 'HI': int, 'MI': int, 'Duracao': float, 'rede': int, 'Patamar': str},
         'colspecs': [(0, 2), (4, 6), (9, 11), (14, 15), (19, 24), (29, 30), (33, 39)],
         'default': [None, None, None, None, 1, None, None]
         },
    'AG':
        {'descricao': 'Número de estágios da PDD',
         'tipo': 'CONFIG',
         'dados': None,
         'colunas': {'MNE': str, 'NEST': int},
         'colspecs': [(0, 2), (3, 6)],
         'default': [None, 1]
         },
    'GP':
        {'descricao': 'Tolerância para convergência',
         'tipo': 'CONFIG',
         'dados': None,
         'colunas': {'MNE': str, 'TOL': float, 'TOLINT': float},
         'colspecs': [(0, 2), (4, 14), (15, 25)],
         'default': [None, 1., 0.00001]
         },
    'NI':
        {'descricao': 'Número máximo de iterações',
         'tipo': 'CONFIG',
         'dados': None,
         'colunas': {'MNE': str, 'FLAG': int, 'NITER': int},
         'colspecs': [(0, 2), (4, 5), (9, 12)],
         'default': [None, 0, 1]
         },
    'TX':
        {'descricao': 'Taxa de juros anual',
         'tipo': 'CONFIG',
         'dados': None,
         'colunas': {'MNE': str, 'TX': float},
         'colspecs': [(0, 2), (4, 14)],
         'default': [None, None]
         },
    'SIST':
        {'descricao': 'Definição dos subsistemas',
         'tipo': 'SIST',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'SIG': str, 'FLAG': int, 'NOME': str},
         'colspecs': [(0, 6), (7, 9), (10, 12), (13, 14), (16, 26)],
         'default': [None, None, None, None, None]
         },
    'IA':
        {'descricao': 'Intercâmbio de energia entre subsistemas',
         'tipo': 'INTERC',
         'dados': None,
         'colunas': {'MNE': str, 'SS1': str, 'SS2': str, 'DI': int, 'HI': int, 'MI': int,
                     'DF': str, 'HF': int, 'MF': int, 'SS1->SS2': float, 'SS2->SS1': float},
         'colspecs': [(0, 2), (4, 6), (9, 11), (13, 15), (16, 18), (19, 20), (21, 23), (24, 26),
                      (27, 28), (29, 39), (39, 49)],
         'default': [None, None, None, None, None, None, None, None, None, 9e10, 9e10]
         },
    'REE':
        {'descricao': 'Definição de reservatórios equivalentes',
         'tipo': 'OUTROS',
         'dados': None,
         'colunas': {'MNE': str, 'NUMREE': int, 'NUMSUBS': int, 'NOME': str},
         'colspecs': [(0, 3), (6, 8), (9, 11), (12, 22)],
         'default': [None, None, None, None]
         },
    'UH':
        {'descricao': 'Definição das usinas hidrelétricas',
         'tipo': 'HIDR',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'NOME': str, 'NUMREE': int, 'VINI': float, 'FLAGEVAP': int,
                     'DI': int, 'HI': int, 'MI': int, 'VMOR': float, 'FLAGPROD': int, 'FLAGBANG': int},
         'colspecs': [(0, 2), (4, 7), (9, 21), (24, 26), (29, 39), (39, 40), (41, 43), (44, 46), (47, 48),
                      (59, 59), (64, 65), (69, 70)],
         'default': [None, None, None, None, None, 0, None, 0, 0, None, 0, 0]
         },
    'UT':
        {'descricao': 'Definição das usinas termelétricas',
         'tipo': 'TERM',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'NOME': str, 'NUMSUB': int, 'FLAGREST': int, 'DI': int, 'HI': int, 'MI': int,
                     'DF': str, 'HF': int, 'MF': int, 'UNIREST': int, 'GMIN/DEC': float, 'GMAX/CRES': float, 'GINI': float},
         'colspecs': [(0, 2), (4, 7), (9, 21), (22, 24), (25, 26), (27, 29), (30, 32), (33, 34), (35, 37), (38, 40),
                      (41, 42), (46, 47), (47, 57), (57, 67), (67, 77)],
         'default': [None, None, None, None, None, None, None, None, None, None, None, 0, None,  None, None]
         },
    'USIE':
        {'descricao': 'Definição das usinas elevatórias',
         'tipo': 'HIDR',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int,  'NUMSUB': int,  'NOME': str, 'NUMMONT': int, 'NUMJUS': int, 'QMIN': float,
                     'QMAX': float, 'TX': float},
         'colspecs': [(0, 4), (5, 8), (9, 11), (14, 26), (29, 32), (34, 37), (39, 49), (49, 59), (59, 69)],
         'default': [None, None, None, None, None, None, 0, None, None]
         },
    'DE':
        {'descricao': 'Demandas e cargas especiais',
         'tipo': 'OUTROS',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'DI': int, 'HI': int, 'MI': int, 'DF': int, 'HF': int, 'MF': int,
                     'DEM': float, 'DESC': str},
         'colspecs': [(0, 2), (4, 7), (8, 10), (11, 13), (14, 15), (16, 18), (19, 21), (22, 23), (24, 34), (35, 45)],
         'default': [None, None, None, None, None, None, None, None, None, None]
         },
    'CI':
        {'descricao': 'Contratos de importação de energia',
         'tipo': 'SIST',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'NOME': str, 'SS/BUS': int, 'FLAG': int,
                     'DI': int, 'HI': int, 'MI': int, 'DF': str, 'HF': int, 'MF': int,
                     'UNIREST': int, 'LINF': float, 'LSUP': float, 'CUSTO': float, 'INI': float},
         'colspecs': [(0, 2), (3, 6), (7, 17), (18, 23), (23, 24), (25, 27), (28, 30), (31, 32), (33, 35), (36, 38),
                      (39, 40), (41, 42), (43, 53), (53, 63), (63, 73), (73, 83)],
         'default': [None, None, None, None, None, None, None, None, None, None, None, 0, 0, 9e10, None, None]
         },
    'CE':
        {'descricao': 'Contratos de exportação de energia',
         'tipo': 'SIST',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'NOME': str, 'SS/BUS': int, 'FLAG': int,
                     'DI': int, 'HI': int, 'MI': int, 'DF': str, 'HF': int, 'MF': int,
                     'UNIREST': int, 'LINF': float, 'LSUP': float, 'CUSTO': float, 'INI': float},
         'colspecs': [(0, 2), (3, 6), (7, 17), (18, 23), (23, 24), (25, 27), (28, 30), (31, 32), (33, 35), (36, 38),
                      (39, 40), (41, 42), (43, 53), (53, 63), (63, 73), (73, 83)],
         'default': [None, None, None, None, None, None, None, None, None, None, None, 0, 0, 9e10, None, None]
         },
    'DP':
        {'descricao': 'Carga de energia',
         'tipo': 'SIST',
         'dados': None,
         'colunas': {'MNE': str, 'NUMSUB': int, 'DI': int, 'HI': int, 'MI': int, 'DF': str, 'HF': int, 'MF': int,
                     'CARGA': float},
         'colspecs': [(0, 2), (4, 6), (8, 10), (11, 13), (14, 15), (16, 18), (19, 21), (22, 23), (24, 34)],
         'default': [None, None, None, None, None, None, None, None, None]
         },
    'CD':
        {'descricao': 'Custo de déficit de energia',
         'tipo': 'SIST',
         'dados': None,
         'colunas': {'MNE': str, 'NUMSUB': int, 'NUMSEG': int, 'DI': int, 'HI': int, 'MI': int, 'DF': str, 'HF': int, 'MF': int,
                     'CDEF': float, 'PROF': float},
         'colspecs': [(0, 2), (4, 5), (6, 8), (9, 11), (12, 14), (15, 16), (17, 19), (20, 22), (23, 24), (25, 35), (35, 45)],
         'default': [None, None, None, None, None, None, None, None, None, None, None]
         },
    'VE':
        {'descricao': 'Volume de espera',
         'tipo': 'HIDR',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'DI': int, 'HI': int, 'MI': int, 'DF': str, 'HF': int, 'MF': int, 'VOL': float},
         'colspecs': [(0, 2), (4, 7), (8, 10), (11, 13), (14, 15), (16, 18), (19, 21), (22, 23), (24, 34)],
         'default': [None, None, None, None, None, None, None, None, 100.]
         },
    'TVIAG':
        {'descricao': 'Tempo de viagem da água',
         'tipo': 'HIDR',
         'dados': None,
         'colunas': {'MNE': str, 'USIMON': int, 'USIJUS': int, 'TIPOJUS': str, 'TEMPO': int, 'TIPOTVIAG': int},
         'colspecs': [(0, 6), (6, 9), (10, 13), (14, 15), (19, 22), (24, 25)],
         'default': [None, None, None, None, 0, None]
         },
    'DA':
        {'descricao': 'Taxa de desvio da água',
         'tipo': 'HIDR',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'DI': int, 'HI': int, 'MI': int, 'DF': str, 'HF': int, 'MF': int,
                     'TX': float, 'OBS': str},
         'colspecs': [(0, 2), (4, 7), (8, 10), (11, 13), (14, 15), (16, 18), (19, 21), (22, 23), (24, 34), (35, 47)],
         'default': [None, None, None, None, None, None, None, None, 0., None]
         },
    'FP':
        {'descricao': 'Dados para discretização da função de produção hidrelétrica',
         'tipo': 'HIDR',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'TIPO': int, 'NTURB': int, 'NVOL': int, 'FLAGCONC': int, 'FLAGMQ': int,
                     'DELTAV': float, 'TOL': float},
         'colspecs': [(0, 2), (3, 6), (7, 8), (10, 13), (15, 18), (20, 21), (24, 25), (29, 39), (39, 49)],
         'default': [None, None, 2, 5, 5, None, None, 100., 0.02]
         },
    'EZ':
        {'descricao': 'Vínculo hidráulico entre subsistemas',
         'tipo': 'SIST',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'PERC': float},
         'colspecs': [(0, 2), (4, 7), (9, 14)],
         'default': [None, None, None]
         },
    'AC':
        {'descricao': 'Alterações de cadastro',
         'tipo': 'HIDR',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'PARAM': str, 'VALOR': str},
         'colspecs': [(0, 2), (4, 7), (9, 15), (19, 76)],
         'default': [None, None, None, None]
         },
    'MH':
        {'descricao': 'Manutenção programada das usinas hidrelétricas',
         'tipo': 'HIDR',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'INDGRUPO': int, 'INDUNI': int,
                     'DI': int, 'HI': int, 'MI': int, 'DF': str, 'HF': int, 'MF': int, 'FLAG': int},
         'colspecs': [(0, 2), (4, 7), (9, 11), (12, 14), (14, 16), (17, 19), (20, 21), (22, 24), (25, 27), (28, 29), (30, 31)],
         'default': [None, None, None, None, None, None, None, None, None, None, None]
         },
    'MT':
        {'descricao': 'Manutenção programada das usinas termelétricas',
         'tipo': 'TERM',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'NUMGRUPO': int,
                     'DI': int, 'HI': int, 'MI': int, 'DF': str, 'HF': int, 'MF': int, 'FLAG': int},
         'colspecs': [(0, 2), (4, 7), (8, 11), (13, 15), (16, 18), (19, 20), (21, 23), (24, 26), (27, 28), (29, 30)],
         'default': [None, None, None, None, None, None, None, None, None, 1]
         },
    'RE':
        {'descricao': 'Restrições elétricas especiais',
         'tipo': 'ELET',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'DI': int, 'HI': int, 'MI': int, 'DF': str, 'HF': int, 'MF': int},
         'colspecs': [(0, 2), (4, 7), (9, 11), (12, 14), (15, 16), (17, 19), (20, 22), (23, 24)],
         'default': [None, None, None, None, None, None, None, None]
         },
    'LU':
        {'descricao': 'Limites das restrições elétricas',
         'tipo': 'ELET',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'DI': int, 'HI': int, 'MI': int, 'DF': str, 'HF': int, 'MF': int,
                     'LINF': float, 'LSUP': float},
         'colspecs': [(0, 2), (4, 7), (8, 10), (11, 13), (14, 15), (16, 18), (19, 21), (22, 23), (24, 34), (34, 44)],
         'default': [None, None, None, None, None, None, None, None, -9e10, 9e10]
         },
    'FH':
        {'descricao': 'Fator de participação das usinas hidroelétricas na restrição elétrica',
         'tipo': 'ELET',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'DI': int, 'HI': int, 'MI': int, 'DF': str, 'HF': int, 'MF': int,
                     'NUSI': int, 'NCONJ': int, 'FATOR': float},
         'colspecs': [(0, 2), (4, 7), (8, 10), (11, 13), (14, 15), (16, 18), (19, 21), (22, 23), (24, 27), (27, 29), (34, 44)],
         'default': [None, None, None, None, None, None, None, None, None, None, 0.]
         },
    'FT':
        {'descricao': 'Fator de participação das usinas termelétricas na restrição elétrica',
         'tipo': 'ELET',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'DI': int, 'HI': int, 'MI': int, 'DF': str, 'HF': int, 'MF': int,
                     'NUSI': int, 'FATOR': float},
         'colspecs': [(0, 2), (4, 7), (8, 10), (11, 13), (14, 15), (16, 18), (19, 21), (22, 23), (24, 27), (34, 44)],
         'default': [None, None, None, None, None, None, None, None, None, 0.]
         },
    'FI':
        {'descricao': 'Fator de participação de intercâmbio na restrição elétrica',
         'tipo': 'ELET',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'DI': int, 'HI': int, 'MI': int, 'DF': str, 'HF': int, 'MF': int,
                     'MNEDE': str, 'MNEPARA': str, 'FATOR': float},
         'colspecs': [(0, 2), (4, 7), (8, 10), (11, 13), (14, 15), (16, 18), (19, 21), (22, 23), (24, 26), (29, 31),
                      (34, 44)],
         'default': [None, None, None, None, None, None, None, None, None, None, 0.]
         },
    'FE':
        {'descricao': 'Fator de participação dos contratos de importação/exportação na restrição elétrica',
         'tipo': 'ELET',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'DI': int, 'HI': int, 'MI': int, 'DF': str, 'HF': int, 'MF': int,
                     'MNEDE': str, 'MNEPARA': str, 'FATOR': float},
         'colspecs': [(0, 2), (4, 7), (8, 10), (11, 13), (14, 15), (16, 18), (19, 21), (22, 23), (24, 27), (34, 44)],
         'default': [None, None, None, None, None, None, None, None, None, 0.]
         },
    'FR':
        {'descricao': 'Fator de participação das fontes renováveis eólicas na restrição elétrica',
         'tipo': 'ELET',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'DI': int, 'HI': int, 'MI': int, 'DF': str, 'HF': int, 'MF': int,
                     'NEOL': int, 'FATOR': float},
         'colspecs': [(0, 2), (4, 9), (10, 12), (13, 15), (16, 17), (18, 20), (21, 23), (24, 25), (26, 31), (36, 46)],
         'default': [None, None, None, None, None, None, None, None, None, 0.]
         },
    'FC':
        {'descricao': 'Fator de participação da demana/cargas especiais na restrição elétrica',
         'tipo': 'ELET',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'DI': int, 'HI': int, 'MI': int, 'DF': str, 'HF': int, 'MF': int,
                     'NDEM': int, 'FATOR': float},
         'colspecs': [(0, 2), (4, 7), (10, 12), (13, 15), (16, 17), (18, 20), (21, 23), (24, 25), (26, 29), (36, 46)],
         'default': [None, None, None, None, None, None, None, None, None, 0.]
         },
    'IT':
        {'descricao': 'Coeficientes da régua 11 de Itaipu',
         'tipo': 'ITAIPU',
         'dados': None,
         'colunas': {'MNE': str, 'REE': int, 'COEF1': int, 'COEF2': int, 'COEF3': int, 'COEF4': int, 'COEF5': int},
         'colspecs': [(0, 2), (4, 6), (9, 24), (24, 39), (39, 54), (54, 69), (69, 84)],
         'default': [None, None, None, None, None, None, None]
         },
    'RI':
        {'descricao': 'Limites para gerações 50/60 Hz de Itaipu',
         'tipo': 'ITAIPU',
         'dados': None,
         'colunas': {'MNE': str, 'DI': int, 'HI': int, 'MI': int, 'DF': str, 'HF': int, 'MF': int,
                     'LINF50': float, 'LSUP50': float, 'LINF60': float, 'LSUP60': float, 'ANDE': float},
         'colspecs': [(0, 2), (8, 10), (11, 13), (14, 15), (16, 18), (19, 21), (22, 23), (26, 36), (36, 46), (46, 56),
                      (56, 66), (66, 76)],
         'default': [None, None, None, None, None, None, None, None, None, None, None, None, None, None]
         },
}

# Configuração arquivo "ptoper.dat"
config_ptoper = {
    'PTOPER USIT':
        {'descricao': 'Definição do ponto de operação de usinas termelétricas (fixa geração térmica)',
         'tipo': 'TERM',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'VAR': str,
                     'DI': int, 'HI': int, 'MI': int, 'DF': str, 'HF': int, 'MF': int, 'VALOR': float},
         'colspecs': [(0, 13), (14, 17), (18, 24), (25, 27), (28, 30), (31, 32), (33, 35), (36, 38), (39, 40),
                      (41, 52)],
         'default': [None, None, None, None, None, None, None, None, None, None]
         }
}

# Configuração arquivo "ptoper.dat"
config_operuh = {
    'OPERUH REST':
        {'descricao': 'Definição das restrições operativas das usinas hidrelétricas (restrição)',
         'tipo': 'HIDR',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'TIP': str, 'FLAG': int, 'JUST': str, 'VALORINI': float},
         'colspecs': [(0, 13), (14, 19), (21, 22), (24, 25), (27, 39), (40, 50)],
         'default': [None, None, None, 1, None, None]
         },
    'OPERUH ELEM':
        {'descricao': 'Definição das restrições operativas das usinas hidrelétricas (elemento)',
         'tipo': 'HIDR',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'NUSI': int, 'NOME': str, 'COD': int, 'FATOR': str},
         'colspecs': [(0, 13), (14, 19), (20, 23), (25, 37), (40, 42), (43, 48)],
         'default': [None, None, None, None, None, None]
         },
    'OPERUH LIM':
        {'descricao': 'Definição das restrições operativas das usinas hidrelétricas (limite)',
         'tipo': 'HIDR',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int,
                     'DI': str, 'HI': int, 'MI': int, 'DF': str, 'HF': int, 'MF': int, 'LINF': float, 'LSUP': float},
         'colspecs': [(0, 13), (14, 19), (20, 22), (23, 25), (26, 27), (28, 30), (31, 33), (34, 35), (38, 48), (48, 58)],
         'default': [None, None, None, None, None, None, None, None, 0., None]
         },
    'OPERUH VAR':
        {'descricao': 'Definição das restrições operativas das usinas hidrelétricas (variação)',
         'tipo': 'HIDR',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int,
                     'DI': str, 'HI': int, 'MI': int, 'DF': str, 'HF': int, 'MF': int,
                     'RampDecrPerc': float, 'RampAcrescPerc': float, 'RampDecrAbs': float, 'RampAcrescAbs': float},
         'colspecs': [(0, 13), (14, 19), (19, 21), (22, 24), (25, 26), (27, 29), (30, 32), (33, 34), (37, 47),
                      (47, 57), (57, 67), (67, 77)],
         'default': [None, None, None, None, None, None, None, None, None, None, None, None]
         },
}

# Configuração arquivo "termdat.dat"
config_termdat = {
    'CADUSIT':
        {'descricao': 'Cadastro das usinas termelétricas',
         'tipo': 'TERM',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'NOME': str, 'NSUB': int,
                     'ANO': int, 'MES': int, 'DIA': int, 'HORA': int, 'MI': int, 'NUNI': int},
         'colspecs': [(0, 7), (8, 11), (12, 24), (25, 27), (28, 32), (33, 35), (36, 38), (39, 41), (42, 43), (45, 48)],
         'default': [None, None, None, None, None, None, None, None, None, None]
         },
    'CADUNIDT':
        {'descricao': 'Cadastro das unidades das usinas termelétricas',
         'tipo': 'TERM',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'IND': int,
                     'ANO': int, 'MES': int, 'DIA': int, 'HORA': int, 'MI': int,
                     'POT': float, 'POTMIN': float, 'TON': int, 'TOFF': int, 'CCOLD': float, 'CSTD': float,
                     'RUP': float, 'RDOWN': float, 'FLAG': int, 'NO': int, 'EQU': int, 'RTRANS': float},
         'colspecs': [(0, 8), (9, 12), (12, 15), (16, 20), (21, 23), (24, 26), (27, 29), (30, 31), (33, 43), (44, 54),
                      (55, 60), (61, 66), (67, 77), (89, 99), (100, 110), (111, 121), (122, 123), (124, 126), (127, 130), (131, 141)],
         'default': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
         },
    'CADCONF':
        {'descricao': 'Cadastro da configuração das usinas termelétricas',
         'tipo': 'TERM',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'INDEQUIV': int, 'INDUNI': int},
         'colspecs': [(0, 7), (8, 11), (12, 15), (16, 19)],
         'default': [None, None, None, None]
         },
    'CADMIN':
        {'descricao': f'''Cadastro da relação de quantidade de unidades reais disponíveis mínimas para acionamento
                        da unidade equivalente''',
         'tipo': 'TERM',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'INDEQUIV': int, 'NMIN': int},
         'colspecs': [(0, 6), (8, 11), (12, 15), (16, 19)],
         'default': [None, None, None, None]
         },
}

# Configuração arquivo "operut.dat"
config_operut = {
    'UCTERM':
        {'descricao': 'Flag para tratamento de unit commitment térmico',
         'tipo': 'CONFIG',
         'dados': None,
         'colunas': {'MNE': str, 'FLAG': int},
         'colspecs': [(0, 6), (7, 8)],
         'default': [None, None]
         },
    'REGRANPTV':
        {'descricao': 'Flag para definir valores default para a função de produção hidrelétrica',
         'tipo': 'CONFIG',
         'dados': None,
         'colunas': {'MNE': str, 'REGRA': int, 'FLAG': int, 'NPONTOS': int},
         'colspecs': [(0, 9), (10, 11), (14, 16), (18, 20)],
         'default': [None, None, None, None]
         },
    'TOLERILH':
        {'descricao': 'Flag para ativar tolerância nas equações de atendimento a demanda por ilha',
         'tipo': 'CONFIG',
         'dados': None,
         'colunas': {'MNE': str, 'FLAG': int},
         'colspecs': [(0, 8), (9, 10)],
         'default': [None, 0]
         },
    'MILPIN':
        {'descricao': 'Flag para processar o problema inteiro mesmo que inviável',
         'tipo': 'CONFIG',
         'dados': None,
         'colunas': {'MNE': str},
         'colspecs': [(0, 6)],
         'default': [None]
         },
    'AJUSTEFCF':
        {'descricao': 'Flag para habilitar ajuste da FCF',
         'tipo': 'CONFIG',
         'dados': None,
         'colunas': {'MNE': str},
         'colspecs': [(0, 9)],
         'default': [None]
         },
    'UCTSER':
        {'descricao': 'Flag para desabilitar processamento paralelo do pacote de otimização',
         'tipo': 'CONFIG',
         'dados': None,
         'colunas': {'MNE': str},
         'colspecs': [(0, 6)],
         'default': [None]
         },
    'AVLCMO':
        {'descricao': 'Flag para habilitar impressão dos arquivos de avaliação do calculo do CMO',
         'tipo': 'CONFIG',
         'dados': None,
         'colunas': {'MNE': str, 'FLAG': int},
         'colspecs': [(0, 6), (7, 8)],
         'default': [None, 1]
         },
    'ENGOLIMENTO':
        {'descricao': 'Flag para habilitar o Engolimento Máximo',
         'tipo': 'CONFIG',
         'dados': None,
         'colunas': {'MNE': str, 'FLAG': int, 'TIPO': int},
         'colspecs': [(0, 11), (12, 13), (14, 15)],
         'default': [None, 1, 2]
         },
    'UCTPAR':
        {'descricao': 'Flag para habilitar número de núcleos para o processamento paralelo do pacote de otimização',
         'tipo': 'CONFIG',
         'dados': None,
         'colunas': {'MNE': str, 'FLAG': int, 'TIPO': int},
         'colspecs': [(0, 6), (7, 9)],
         'default': [None, 4]
         },
    'CPXPRESLV':
        {'descricao': 'Flag para desabilitar o pré-processamento do pacote de otimização',
         'tipo': 'CONFIG',
         'dados': None,
         'colunas': {'MNE': str},
         'colspecs': [(0, 9)],
         'default': [None]
         },
    'FLGUCTERM':
        {'descricao': 'Flag para ativação de variáveis de folga para as restrições de geração térmica mínima de acionamento',
         'tipo': 'CONFIG',
         'dados': None,
         'colunas': {'MNE': str},
         'colspecs': [(0, 9)],
         'default': [None]
         },
    'UCTBUSLOC':
        {
            'descricao': 'Flag para ativar a restrição de Busca Local',
            'tipo': 'CONFIG',
            'dados': None,
            'colunas': {'MNE': str},
            'colspecs': [(0, 9)],
            'default': [None]
            },
    'PINT':
        {
            'descricao': 'Flag para ativar a metodologia de Pontos Interiores',
            'tipo': 'CONFIG',
            'dados': None,
            'colunas': {'MNE': str},
            'colspecs': [(0, 4)],
            'default': [None]
            },
    'UCTHEURFP':
        {'descricao': f'''Flag para ativar a metodologia Feasibility Pump com Busca Local e Fixação de Variáveis de Status, 
                        resolvendo-se os problemas lineares pelo método de Pontos Interiores''',
         'tipo': 'CONFIG',
         'dados': None,
         'colunas': {'MNE': str, 'NREL': int, 'NMIN': int},
         'colspecs': [(0, 9), (10, 13), (14, 17)],
         'default': [None, None, None]
         },
    'CONSTDADOS':
        {'descricao': 'Flag para ativar a Consistência dos dados',
         'tipo': 'CONFIG',
         'dados': None,
         'colunas': {'MNE': str, 'FLAG': int},
         'colspecs': [(0, 10), (11, 12)],
         'default': [None, None]
         },
    'INIT':
        {'descricao': 'Condições iniciais das unidades',
         'tipo': 'TERM',
         'dados': None,
         'colunas': {'NUM': int, 'NOME': str, 'IND': int, 'STATUS': int, 'GERINI': float, 'TEMPO': int,
                     'MH': int, 'AD': int},
         'colspecs': [(0, 3), (4, 16), (18, 21), (24, 26), (29, 39), (41, 46), (48, 49), (51, 52)],
         'default': [None, None, None, None, None, None, 0, 0]
         },
    'OPER':
        {'descricao': 'Limites e condições operativas das unidades',
         'tipo': 'TERM',
         'dados': None,
         'colunas': {'NUM': int, 'NOME': str, 'IND': int,
                     'DI': str, 'HI': int, 'MI': int, 'DF': str, 'HF': int, 'MF': int,
                     'LINF': float, 'LSUP': float, 'CUSTO': float},
         'colspecs': [(0, 3), (4, 16), (16, 19), (20, 22), (23, 25), (26, 27), (28, 30), (31, 33), (34, 35), (36, 46),
                      (46, 56), (56, 66)],
         'default': [None, None, None, None, None, None, None, None, None, 0, None, None]
         },
}

# Configuração arquivo "areacont.dat"
config_areacont = {
    'AREA':
        {'descricao': 'Cadastro das Áreas de Reserva de Potência (definição das áreas)',
         'tipo': 'RESPOT',
         'dados': None,
         'colunas': {'NUM': int, 'NOME': str},
         'colspecs': [(0, 3), (9, 49)],
         'default': [None, None]
         },
    'USINA':
        {'descricao': 'Cadastro das Áreas de Reserva de Potência (usina)',
         'tipo': 'RESPOT',
         'dados': None,
         'colunas': {'NUM': int, 'CONJ': int, 'TIPO': str, 'NUMMNE': str, 'NOME': str},
         'colspecs': [(0, 3), (4, 5), (7, 8), (9, 12), (14, 54)],
         'default': [None, None, None, None, None]
         },
}

# Configuração arquivo "respot.dat"
config_respot = {
    'RP':
        {'descricao': 'Reserva de potência por área (identificação das áreas)',
         'tipo': 'RESPOT',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int,
                     'DI': str, 'HI': int, 'MI': int, 'DF': str, 'HF': int, 'MF': int, 'OBS': str},
         'colspecs': [(0, 2), (4, 7), (9, 11), (12, 14), (15, 16), (17, 19), (20, 22), (23, 24), (30, 70)],
         'default': [None, None, None, None, None, None, None, None, None]
         },
    'LM':
        {'descricao': 'Reserva de potência por área (valores mínimos da reserva de potência)',
         'tipo': 'RESPOT',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int,
                     'DI': str, 'HI': int, 'MI': int, 'DF': str, 'HF': int, 'MF': int, 'RESMIN': float},
         'colspecs': [(0, 2), (4, 7), (9, 11), (12, 14), (15, 16), (17, 19), (20, 22), (23, 24), (25, 35)],
         'default': [None, None, None, None, None, None, None, None, None]
         },
}

# Configuração arquivo "deflant.dat"
config_deflant = {
    'DEFANT':
        {'descricao': 'Defluência das usinas hidrelétricas antes do estudo (para uso com tempo de viagem)',
         'tipo': 'HIDR',
         'dados': None,
         'colunas': {'MNE': str, 'NUMMON': int, 'NUMJUS': int, 'TIPO': str,
                     'DI': str, 'HI': int, 'MI': int, 'DF': str, 'HF': int, 'MF': int, 'DEFL': float},
         'colspecs': [(0, 6), (9, 12), (14, 17), (19, 20), (24, 26), (27, 30), (30, 31), (32, 34), (35, 37),
                      (38, 39), (44, 54)],
         'default': [None, None, None, None, None, None, None, None, None, None, None]
         }
}

# Configuração arquivo "restseg.dat"
config_restseg = {
    'TABSEG INDICE':
        {'descricao': 'Restrição de segurança (índice da restrição)',
         'tipo': 'SEGUR',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'DESC': str},
         'colspecs': [(0, 13), (14, 19), (20, 30)],
         'default': [None, None, None, None]
         },
    'TABSEG TABELA':
        {'descricao': 'Restrição de segurança (equação de fluxo por tabela)',
         'tipo': 'SEGUR',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'TIPO1': str, 'TIPO2': str, 'NUNVAR': int, 'PERC': float},
         'colspecs': [(0, 13), (14, 19), (20, 26), (27, 33), (34, 39), (40, 45)],
         'default': [None, None, None, None, None, None]
         },
    'TABSEG LIMITE':
        {'descricao': 'Restrição de segurança (limite dos parâmetros)',
         'tipo': 'SEGUR',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'LIM1': float, 'LIM2': float, 'LIM3': float},
         'colspecs': [(0, 13), (14, 19), (20, 30), (31, 41), (42, 52)],
         'default': [None, None, None, None, None]
         },
    'TABSEG CELULA':
        {'descricao': 'Restrição de segurança (limite de cada intervalo dos parâmetros)',
         'tipo': 'SEGUR',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'LIM': float, 'INF1': float, 'SUP1': float,
                     'INF2': float, 'SUP2': float, 'INF3': float, 'SUP3': float},
         'colspecs': [(0, 13), (14, 19), (20, 30), (31, 32), (36, 46), (48, 58), (60, 70), (72, 82), (84, 94), (96, 106)],
         'default': [None, None, None, None, None, None, None, None, None, None]
         },
}

# Configuração arquivo "rstlpp.dat"
config_rstlpp = {
    'RSTSEG':
        {'descricao': 'Restrições de segurança - funções lineares por parte LPP (definição)',
         'tipo': 'SEGUR',
         'dados': None,
         'colunas': {'MNE': str, 'NOME': str, 'NUMLPP': int, 'FLAG': int, 'NUM': int, 'CHAVE': str, 'IDENT': int, 'DESC': str},
         'colspecs': [(0, 6), (7, 14), (15, 19), (19, 20), (20, 24), (25, 30), (31, 36), (37, 77)],
         'default': [None, None, None, None, None, None, None, None]
         },
    'ADICRS':
        {'descricao': 'Restrições de segurança - funções lineares por parte LPP (adição mais de uma restrição controlada)',
         'tipo': 'SEGUR',
         'dados': None,
         'colunas': {'MNE': str, 'NUMLPP': int, 'CHAVE': str, 'IDENT': int},
         'colspecs': [(0, 6), (7, 14), (15, 19), (19, 20), (25, 30), (31, 36), (37, 77)],
         'default': [None, None, None, None]
         },
    'PARAM':
        {'descricao': 'Restrições de segurança - funções lineares por parte LPP (definição dos parâmetros)',
         'tipo': 'SEGUR',
         'dados': None,
         'colunas': {'MNE': str, 'NUMLPP': int, 'CHAVE': str,  'IDENT': str},
         'colspecs': [(0, 5), (6, 10), (11, 16), (17, 22)],
         'default': [None, None, None, None]
         },
    'VPARAM':
        {'descricao': 'Restrições de segurança - funções lineares por parte LPP (valores dos parâmetros)',
         'tipo': 'SEGUR',
         'dados': None,
         'colunas': {'MNE': str, 'NUMLPP': int, 'NUMCURVA': int, 'INF1': float, 'SUP1': float, 'INF2': float, 'SUP2': float},
         'colspecs': [(0, 5), (6, 10), (11, 13), (14, 24), (25, 35), (36, 46), (47, 57)],
         'default': [None, None, None, None, None, None, None]
         },
    'RESLPP':
        {'descricao': 'Restrições de segurança - funções lineares por parte LPP (LPP para cada valor do parâmetro)',
         'tipo': 'SEGUR',
         'dados': None,
         'colunas': {'MNE': str, 'NUMLPP': int, 'NUMCURVA': int, 'INDCORTE': int, 'COEFANG': float, 'COEFLIN': float,
                     'COEFANG2': float, 'COEFANG3': float, 'COEFANG4': float},
         'colspecs': [(0, 6), (7, 11), (12, 13), (14, 15), (16, 26), (27, 37), (38, 48), (49, 59), (60, 70)],
         'default': [None, None, None, None, None, None, None, None, None]
         },
}

# Configuração arquivo "rampas.dat"
config_rampas = {
    'RAMP':
        {'descricao': 'Trajetória de acionamento/desligamento das usidades térmicas',
         'tipo': 'TERM',
         'dados': None,
         'colunas': {'NUM': int, 'INDUNI': int, 'IMP': str, 'TRAJ': str, 'POT': float, 'TEMPO': int, 'FLAG': int},
         'colspecs': [(0, 3), (4, 7), (13, 14), (17, 18), (20, 30), (31, 36), (37, 38)],
         'default': [None, None, None, None, None, None, None]
         }
}

# Configuração arquivo "renovaveis.dat"
config_renovaveis = {
    'EOLICA':
        {'descricao': 'Dados das usinas eólicas (configuração)',
         'tipo': 'RENOV',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'NOME': str, 'POTMAX': float, 'FATOR': float, 'FLAG': int},
         'default': [None, None, None, None, None, None]
         },
    'EOLICABARRA':
        {'descricao': 'Localização elétrica das usinas eólicas (barra)',
         'tipo': 'RENOV',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'BARRA': int},
         'default': [None, None, None]
         },
    'EOLICASUBM':
        {'descricao': 'Localização elétrica das usinas eólicas (submercado)',
         'tipo': 'RENOV',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int, 'SUBM': str},
         'default': [None, None, None]
         },
    'EOLICA-GERACAO':
        {'descricao': 'Dados usinas eólicas (geração)',
         'tipo': 'RENOV',
         'dados': None,
         'colunas': {'MNE': str, 'NUM': int,
                     'DI': str, 'HI': int, 'MI': int, 'DF': str, 'HF': int, 'MF': int,
                     'GERACAO': float},
         'default': [None, None, None, None, None, None, None, None, None]
         },
}

# List of all files
list_files_data = [config_entdados, config_ptoper, config_operuh, config_termdat,
                   config_operut, config_areacont, config_respot, config_deflant,
                   config_restseg, config_rstlpp, config_rampas, config_renovaveis]


# Contraints codes for hydro varaibles
constraints_variable_codes = {
    1: 'CotaMontante',
    2: 'VolArm_%VU',
    3: 'VazTurb',
    4: 'VazVert',
    5: 'VazDesv',
    6: 'VazDefTotal',
    7: 'Geracao',
    8: 'VazBomb',
    9: 'VazAfl'
}
