class otimosist(object):

    Nome = None
    EArm = None
    EVert = None
    GHidr = None
    Exc = None
    Deficit = None
    GTerm = None
    CMO = None
    ValorAgua = None
    DemLiq = None
    OutroUsos = None

    def __init__(self, Nome):
        self.Nome = Nome


class otimoterm(object):
    GT = None
    GT_MAX = None
    Nome = None
    Sistema = None

    def __init__(self):
        pass

class otimointerc(object):
    INT = None
    INT_MAX = None
    De = None
    Para = None
    Nome = None
