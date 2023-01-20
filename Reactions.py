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
        self.COC = 'COCNH'
        self.Cl = 'ClNH'

class COOH:
    def __init__(self):
        self.NH2 = "CONH"
        self.OH = 'COOC'
        self.COC = 'COCOH'

class NH:
    def __init__(self):
        self.COC = 'COCN'

class COC:
    def __init__(self):
        self.NH2 = 'COCNH'
        self.OH = 'COOC'
        self.COOH = 'COCOH'
        self.NH = 'COCN'

class OH:
    def __init__(self):
        self.COOH = 'COOC'
        self.COC = 'COOC'



