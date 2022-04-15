class Epichlorohydrin:
    def __init__(self):
        self.name = "Epichlorohydrin"
        self.sn = "Epi"
        self.formula = "C3H5ClO"
        self.mw = 92.52
        self.density = 1.18
        self.prg = 1
        self.srg = 1
        self.tg = self.prg + self.srg
        self.cg = S_Hydroxyl

class Butanol:
    def __init__(self):
        self.name = "Butanol"
        self.sn = "BuOH"
        self.formula = "C4H10O"
        self.mw = 74.121
        self.density = 0.810
        self.prg = 1
        self.srg = 0
        self.tg = self.prg + self.srg

class Glycerol:
    def __init__(self):
        self.name = "Glycerol"
        self.sn = "Gly"
        self.formula = "C3H8O"
        self.mw = 92.09382
        self.density = 1.26
        self.prg = P_Hydroxyl
        self.srg = S_Hydroxyl
        self.comp = (self.prg, self.srg, self.prg)
        self.mass = self.mw

class C181:
    def __init__(self):
        self.name = "C181"
        self.sn = "C181"
        self.formula = "C18H34O2"
        self.mw = 282.47
        self.density = 0.895
        self.prg = Carboxyl
        self.mass = self.mw
        self.comp = self.prg

class Water:
    def __init__(self):
        self.name = "Water"
        self.sn = "H2O"
        self.formula = "H2O"
        self.mw = 18.015
        self.density = 1.000
        self.prg = 1
        self.srg = 0
        self.tg = self.prg + self.srg

class Ethanol:
    def __init__(self):
        self.name = "Ethanol"
        self.sn = "EtOH"
        self.formula = "C2H6O"
        self.mw = 46.0688
        self.density = 0.789
        self.prg = 1
        self.srg = 0
        self.tg = self.prg + self.srg

class Methanol:
    def __init__(self):
        self.name = "Methanol"
        self.sn = "MeOH"
        self.formula = "C2H5O"
        self.mw = 30.0469
        self.density = 0.789
        self.prg = 1
        self.srg = 0
        self.tg = self.prg + self.srg

class Acetone:
    def __init__(self):
        self.name = "Acetone"
        self.sn = "AcOH"
        self.formula = "C2H3O"
        self.mw = 42.0367
        self.density = 0.789
        self.prg = 0
        self.srg = 0
        self.tg = self.prg + self.srg

class AceticAcid:
    def __init__(self):
        self.name = "Acetic Acid"
        self.sn = "AA"
        self.formula = "CH3COOH"
        self.mw = 60.052
        self.density = 1.05
        self.prg = 1
        self.srg = 0
        self.tg = self.prg + self.srg

class FormicAcid:
    def __init__(self):
        self.name = "Formic Acid"
        self.sn = "FA"
        self.formula = "CH2O2"
        self.mw = 46.03
        self.density = 1.22
        self.prg = 1
        self.srg = 0
        self.tg = self.prg + self.srg

class PAA:
    def __init__(self):
        self.name = "PAA"
        self.sn = "PAA"
        self.formula = "C2H4O3"
        self.mw = 76.0514
        self.density = 104
        self.prg = 1
        self.srg = 0
        self.tg = self.prg + self.srg

class DETA:
    def __init__(self):
        self.name = "DETA"
        self.sn = "DETA"
        self.formula = "C4H13N3"
        self.mw = 103.169
        self.density = 0.955
        self.prg = 2
        self.srg = 1
        self.tg = self.prg + self.srg

class P_Hydroxyl:
    def __init__(self):
        self.name = "P-Hydroxyl"
        self.OH = 1
        self.wt = 17.007
        self.rxn = (Carboxyl, Epoxide, self)

class S_Hydroxyl:
    def __init__(self):
        self.name = "P-Hydroxyl"
        self.OH = 1
        self.wt = 17.007
        self.rxn = (Carboxyl, Epoxide)

class Carboxyl:
    def __init__(self):
        self.name = "Carbonyl"
        self.COOH = 1
        self.rxn = (P_Hydroxyl, S_Hydroxyl, Epoxide, P_Amine)

class P_Amine:
    def __init__(self):
        self.name = "Carbonyl"
        self.rxn = (Epoxide, Carboxyl)

class S_Amine:
    def __init__(self):
        self.name = "Carbonyl"
        self.rxn = (Epichlorohydrin, Carboxyl)

class Epoxide:
    def __init__(self):
        self.name = "Epoxide"
        self.rxn = (P_Hydroxyl, S_Hydroxyl, P_Amine, S_Amine, Carboxyl)

class P_Ester:
    def __init__(self):
        self.name = "P_Ester"
        self.rxn = P_Amine

class S_Ester:
    def __init__(self):
        self.name = "S_Ester"
        self.rxn = P_Amine


