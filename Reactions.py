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
        self.NH = ['COC']
        self.POH = ['COOH', 'COC']
        self.SOH = ['COOH', 'COC']
        self.COOH = ['NH2', 'POH', 'COC', 'SOH']
        self.COC = ['NH2', 'POH', 'SOH', 'COOH', 'NH']
        self.HPCOH = ['COC']
        self.Cl = ['NH2']
        self.COOC = None

class NH2:
    def __init__(self):
        self.COOH = "CONH"
        self.COOH_wl = Water().mw
        self.COOH_wl_id = "Water"
        self.COC = 'COCNH'
        self.COC_wl = 0
        self.COC_wl_id = 'None'
        self.Cl = 'ClNH'
        self.Cl_wl = 0

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
        self.COC = 'COCN'
        self.COC_wl = 0
        self.COC_wl_id = 'None'

class COC:
    def __init__(self):
        self.NH2 = 'COCNH'
        self.NH2_wl = 0
        self.NH2_wl_id = 'None'
        self.POH = 'HPCOH'
        self.POH_wl = 0
        self.POH_wl_id = 'None'
        self.SOH = 'HPCOH'
        self.SOH_wl = 0
        self.SOH_wl_id = 'None'
        self.COOH = 'COCOH'
        self.COOH_wl = 0
        self.NH = 'COCN'
        self.NH_wl = 0
        self.HPCOH = 'HPCOH'
        self.HPCOH_wl = 0
        self.HPCOH_wl_id = 'None'

class POH:
    def __init__(self):
        self.COOH = 'COOC'
        self.COOH_wl = Water().mw
        self.COOH_wl_id = 'Water'
        self.COC = 'HPCOH'
        self.COC_wl = 0
        self.COC_wl_id = 'None'

class SOH:
    def __init__(self):
        self.COOH = 'COOC'
        self.COOH_wl = Water().mw
        self.COOH_wl_id = 'Water'
        self.COC = 'HPCOH'
        self.COC_wl = 0
        self.COC_wl_id = 'None'

class HPCOH:
    def __init__(self):
        self.COC = 'HPCOH'
        self.COC_wl = 0
        self.COC_wl_id = 'None'







