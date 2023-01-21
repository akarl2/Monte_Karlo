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
End_Metrics = ["Amine_Value", "Acid_Value", "OH_Value"]
Num_Samples = ["1000", "2500", "5000", "10000", "100000"]


class reactive_groups:
    def __init__(self):
        self.NH2 = ['COOH', 'COC', 'Cl']
        self.NH = ['COC']
        self.OH = ['COOH', 'COC']
        self.COOH = ['NH2', 'OH', 'COC']
        self.COC = ['NH2', 'OH', 'COOH', 'NH']

class NH2:
    def __init__(self):
        self.COOH = "CONH"
        self.COOH_wl = Water().mw
        self.COC = 'COCNH'
        self.COC_wl = 0
        self.Cl = 'ClNH'
        self.Cl_wl = 0


class COOH:
    def __init__(self):
        self.NH2 = "CONH"
        self.NH2_wl = Water().mw
        self.OH = 'COOC'
        self.OH_wl = Water().mw
        self.COC = 'COCOH'
        self.COC_wl = 0

class NH:
    def __init__(self):
        self.COC = 'COCN'
        self.COC_wl = 0

class COC:
    def __init__(self):
        self.NH2 = 'COCNH'
        self.NH2_wl = 0
        self.OH = 'COOC'
        self.OH_wl = 0
        self.COOH = 'COCOH'
        self.COOH_wl = 0
        self.NH = 'COCN'
        self.NH_wl = 0

class OH:
    def __init__(self):
        self.COOH = 'COOC'
        self.COOH_wl = Water().mw
        self.COC = 'COOC'
        self.COC_wl = 0



