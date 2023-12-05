from Database import *

class Condensation:
    def __init__(self):
        self.wl = Water().mw
        self.name = Condensation

class PolyCondensation:
    def __init__(self):
        self.wl = Water().mw
        self.name = PolyCondensation

class Etherification:
    def __init__(self):
        self.wl = 0
        self.name = Etherification

Reactions = ["Condensation", "Etherification", "PolyCondensation"]
End_Metrics = ["Amine Value", "Acid Value", "OH Value", "Epoxide Value", '% EHC', 'Iodine Value', "1° TAV", "2° TAV", "3° TAV", '% Cl']
Num_Samples = ["1000", "2500", "5000", "10000", "25000", "50000", "100000"]


class reactive_groups:
    def __init__(self):
        self.NH2 = ['COOH', 'COC', 'Cl','aB_unsat']
        self.NH = ['COC', 'Cl', 'COOH', 'COHH', 'aB_unsat']
        self.N = ['Cl']
        self.POH = ['COOH', 'COC']
        self.SOH = ['COOH', 'COC']
        self.COOH = ['NH2', 'NH', 'POH', 'COC', 'SOH']
        self.COC = ['NH2', 'POH', 'SOH', 'COOH', 'NH']
        self.COHH = ['NH2']
        self.imine = ['CC_Nuc']
        self.Cl = ['NH2', 'NH']
        self.aB_unsat = ['NH2', 'NH']
        self.CC_3 = None
        self.CC_2 = None
        self.CC_1 = None
        self.CC_Nuc = ['imine']
        self.COOC = None
        self.CONH = None
        self.EHC = ['OH']

class NH2:
    def __init__(self):
        self.COOH = "CONH"
        self.COOH_wl = 18.01528
        self.COOH_wl_id = "H2O"
        self.COC = 'NH'
        self.COC_wl = 0
        self.COC_wl_id = 'None'
        self.COC_2 = 'SOH'
        self.COC_2_wl = 0
        self.COC_2_wl_id = 'None'
        self.Cl = 'NH'
        self.Cl_wl = 36.458
        self.Cl_wl_id = "HCl"
        self.aB_unsat = 'NH'
        self.aB_unsat_wl = 0
        self.aB_unsat_wl_id = 'None'
        self.COHH = 'imine'
        self.COHH_wl = 18.0158
        self.COHH_wl_id = "H2O"

class CC_Nuc:
    def __init__(self):
        self.imine = 'NH'
        self.imine_wl = 0
        self.imine_wl_id = None

class imine:
    def __init__(self):
        self.CC_Nuc = 'NH'
        self.CC_Nuc_wl = 0
        self.CC_Nuc_wl_id = None

class COHH:
    def __init__(self):
        self.NH2 = "imine"
        self.NH2_wl = 18.01528
        self.NH2_wl_id = "H2O"
        self.NH = "N"
        self.NH_wl = 18.01528
        self.NH_wl_id = "H2O"

class aB_unsat:
    def __init__(self):
        self.NH2 = 'NH'
        self.NH2_wl = 0
        self.NH2_wl_id = 'None'
        self.NH = 'N'
        self.NH_wl = 0
        self.NH_wl_id = 'None'

class COOH:
    def __init__(self):
        self.NH2 = "CONH"
        self.NH2_wl = 18.01528
        self.NH2_wl_id = "H2O"
        self.NH = "CONH"
        self.NH_wl = 18.01528
        self.NH_wl_id = "H2O"
        self.POH = 'COOC'
        self.POH_wl = 18.01528
        self.POH_wl_id = "H2O"
        self.SOH = 'COOC'
        self.SOH_wl = 18.01528
        self.SOH_wl_id = "H2O"
        self.COC = 'COCOH'
        self.COC_wl = 0
        self.COC_wl_id = 'None'

class NH:
    def __init__(self):
        self.COC = 'N'
        self.COC_wl = 0
        self.COC_wl_id = 'None'
        self.COC_2 = 'SOH'
        self.COC_2_wl = 0
        self.COC_2_wl_id = 'None'
        self.Cl = 'N'
        self.Cl_wl = 36.458
        self.Cl_wl_id = "HCl"
        self.COHH = 'CN'
        self.COHH_wl = 18.0158
        self.COHH_wl_id = "H2O"
        self.COOH = 'COHN'
        self.COOH_wl = 18.0158
        self.COOH_wl_id = "H2O"
        self.aB_unsat = 'N'
        self.aB_unsat_wl = 0
        self.aB_unsat_wl_id = 'None'


class COC:
    def __init__(self):
        self.NH2 = 'NH'
        self.NH2_wl = 0
        self.NH2_wl_id = 'None'
        self.NH2_2 = 'SOH'
        self.NH2_2_wl = 0
        self.NH2_2_wl_id = 'None'
        self.POH = 'SOH'
        self.POH_wl = 0
        self.POH_wl_id = 'None'
        self.SOH = 'SOH'
        self.SOH_wl = 0
        self.SOH_wl_id = 'None'
        self.COOH = 'COCOH'
        self.COOH_wl = 0
        self.COOH_wl_id = 'None'
        self.NH = 'N'
        self.NH_wl = 0
        self.NH_wl_id = 'None'
        self.NH_2 = 'SOH'
        self.NH_2_wl = 0
        self.NH_2_wl_id = 'None'

class POH:
    def __init__(self):
        self.COOH = 'COOC'
        self.COOH_wl = 18.01528
        self.COOH_wl_id = "H2O"
        self.COC = 'SOH'
        self.COC_wl = 0
        self.COC_wl_id = 'None'


class SOH:
    def __init__(self):
        self.COOH = 'COOC'
        self.COOH_wl = 18.01528
        self.COOH_wl_id = "H2O"
        self.COC = 'SOH'
        self.COC_wl = 0
        self.COC_wl_id = 'None'

class Cl:
    def __init__(self):
        self.NH2 = 'NH'
        self.NH2_wl = 36.458
        self.NH2_wl_id = 'HCl'
        self.NH = 'N'
        self.NH_wl = 36.458
        self.NH_wl_id = 'HCl'

class EHC:
    def __init__(self):
        self.OH = 'COC'
        self.OH_wl = 58.44
        self.OH_wl_id = 'NaCl'








