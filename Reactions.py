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
End_Metrics = ["Amine Value", "Acid Value", "OH Value", "Epoxide Value", '% EHC']
Num_Samples = ["1000", "2500", "5000", "10000", "100000"]


class reactive_groups:
    def __init__(self):
        self.NH2 = ['COOH', 'COC', 'Cl']
        self.NH = ['COC', 'Cl']
        self.POH = ['COOH', 'COC']
        self.SOH = ['COOH', 'COC']
        self.COOH = ['NH2', 'POH', 'COC', 'SOH']
        self.COC = ['NH2', 'POH', 'SOH', 'COOH', 'NH']
        self.Cl = ['NH2', 'NH']
        self.COOC = None

class NH2:
    def __init__(self):
        self.COOH = "CONH"
        self.COOH_wl = Water().mw
        self.COOH_wl_id = "Water"
        self.COC = 'NH'
        self.COC_wl = 0
        self.COC_wl_id = 'None'
        self.COC_2 = 'SOH'
        self.COC_2_wl = 0
        self.COC_2_wl_id = 'None'
        self.Cl = 'NH'
        self.Cl_wl = 36.458
        self.Cl_wl_id = "HCl"

class COOH:
    def __init__(self):
        self.NH2 = "CONH"
        self.NH2_wl = Water().mw
        self.NH2_wl_id = "Water"
        self.POH = 'COOC'
        self.POH_wl = Water().mw
        self.POH_wl_id = 'Water'
        self.SOH = 'COOC'
        self.SOH_wl = Water().mw
        self.SOH_wl_id = 'Water'
        self.COC = 'COCOH'
        self.COC_wl = 0

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
        self.NH = 'N'
        self.NH_wl = 0
        self.NH_wl_id = 'None'
        self.NH_2 = 'SOH'
        self.NH_2_wl = 0
        self.NH_2_wl_id = 'None'


class POH:
    def __init__(self):
        self.COOH = 'COOC'
        self.COOH_wl = Water().mw
        self.COOH_wl_id = 'Water'
        self.COC = 'SOH'
        self.COC_wl = 0
        self.COC_wl_id = 'None'

class SOH:
    def __init__(self):
        self.COOH = 'COOC'
        self.COOH_wl = Water().mw
        self.COOH_wl_id = 'Water'
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






