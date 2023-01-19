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
        self.NH = ['COOH', 'COC']
        self.OH = ['COOH', 'COC']
        self.COOH = ['NH₂', 'OH', 'COC']
        self.COC = ['NH₂', 'OH', 'COOH', 'NH']


