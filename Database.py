from inspect import isclass
import Database

#Create a dictionary of all the above classes
reactantsA = ["Epichlorohydrin","Butanol", "Glycerol", "PPG425", "Sorbitol", "Propylene_Glycol", "Pentaerythritol", "Butanediol", "Trimethylolpropane", "C181", "Water", "Ethanol", "Methanol", "Acetone", "Acetic Acid", "Formic_Acid", "PAA", "DETA", "Adipic_Acid", "Cyclohexanedimethanol","Ethylhexanol", "Pripol", "Isostearic_Acid", "DEA", "Lysinol", "LTETA"]
reactantsB = reactantsA

#------------------------------------------Alcohols------------------------------#

class Butanol:
    def __init__(self):
        self.name = "Butanol"
        self.sn = "BuOH"
        self.formula = "C4H10O"
        self.mw = 74.121
        self.density = 0.810
        self.prgmw = self.mw
        self.srgmw = 0
        self.comp = self.mw
        self.mass = self.mw
        self.numgroups = 1

class Glycerol:
    def __init__(self):
        self.name = "Glycerol"
        self.sn = "Gly"
        self.formula = "C3H6O3"
        self.mw = 92.09382
        self.density = 1.26
        self.prgmw = 31.03392
        self.srgmw = 30.02598
        self.comp = (self.prgmw, self.srgmw, self.prgmw)
        self.mass = self.mw
        self.rg = "Alcohol"
        self.numgroups = len(set(self.comp))
        self.prgk = 1
        self.srgk = 1
        self.crgk = 0
        self.trgk = 0
        self.prgID = "OH"
        self.srgID = "OH"
        self.trgID = "None"

class PPG425:
    def __init__(self):
        self.name = "PPG425"
        self.sn = "PPG425"
        self.formula = "C3H6O3"
        self.mw = 425
        self.density = 1.26
        self.prgmw = 212.5
        self.srgmw = 0
        self.comp = (self.prgmw, self.prgmw)
        self.mass = self.mw
        self.numgroups = len(set(self.comp))

class Pripol:
    def __init__(self):
        self.name = "Pripol"
        self.sn = "Pripol"
        self.formula = "C3H6O3"
        self.mw = 542
        self.density = 1.26
        self.prgmw = 271.014
        self.srgmw = 0
        self.comp = (self.prgmw, self.prgmw)
        self.mass = self.mw
        self.numgroups = len(set(self.comp))

class Sorbitol:
    def __init__(self):
        self.name = "Sorbitol"
        self.sn = "Sor"
        self.formula = "C6H14O6"
        self.mw = 182.17
        self.density = 1.49
        self.prgmw = 31.03392
        self.srgmw = 30.02598
        self.comp = (self.prgmw, self.srgmw, self.srgmw, self.srgmw, self.srgmw, self.prgmw)
        self.mass = self.mw
        self.numgroups = len(set(self.comp))
        self.prgk = 1
        self.srgk = 1
        self.crgk = 0
        self.trgk = 0
        self.prgID = "OH"
        self.srgID = "OH"
        self.trgID = "None"

class Propylene_Glycol:
    def __init__(self):
        self.name = "Propylene_Glycol"
        self.sn = "PG"
        self.formula = "C3H8O2"
        self.mw = 76.095
        self.density = 1.04
        self.prgmw = self.mw / 2
        self.srgmw = 0
        self.comp = (self.prgmw, self.prgmw)
        self.mass = self.mw
        self.numgroups = len(set(self.comp))

class Pentaerythritol:
    def __init__(self):
        self.name = "Pentaerythritol"
        self.sn = "Pen"
        self.formula = "C5H12O4"
        self.mw = 136.15
        self.prgmw = 34.0375
        self.srgmw = 0
        self.density = 1.40
        self.comp = (self.prgmw,self.prgmw,self.prgmw,self.prgmw)
        self.mass = self.mw
        self.numgroups = len(set(self.comp))

class Butanediol:
    def __init__(self):
        self.name = "1,4-Butanediol"
        self.sn = "1,4-BDO"
        self.formula = "C4H10O2"
        self.mw = 90.122
        self.density = 1.0171
        self.prgmw = 45.061
        self.srgmw = 0
        self.comp = (self.prgmw, self.prgmw)
        self.mass = self.mw
        self.numgroups = len(set(self.comp))

class Trimethylolpropane:
    def __init__(self):
        self.name = "Trimethylolpropane"
        self.sn = "TMP"
        self.formula = "C6H14O3"
        self.mw = 134.17
        self.density = 1.08
        self.prgmw = 44.723
        self.srgmw = 44.722
        self.comp = (self.prgmw, self.srgmw, self.prgmw)
        self.mass = self.mw
        self.numgroups = len(set(self.comp))

class Cyclohexanedimethanol:
    def __init__(self):
        self.name = "Cyclohexanedimethanol"
        self.sn = "CHDM"
        self.formula = "C8H16O2"
        self.mw = 144.21
        self.density = 1.02
        self.prgmw = 72.105
        self.srgmw = 0
        self.comp = (self.prgmw, self.prgmw)
        self.mass = self.mw
        self.numgroups = len(set(self.comp))

class Ethylhexanol:
    def __init__(self):
        self.name = "2-Ethylhexanol"
        self.sn = "2-EH"
        self.formula = "C8H18O"
        self.mw = 130.23
        self.density = 0.833
        self.mass = self.mw
        self.comp = self.prg
        self.numgroups = 1

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
        self.prgmw = 30.0469
        self.srgmw = 0
        self.comp = self.prgmw
        self.mass = self.mw
        self.numgroups = 1

#-------------------------------------------Amines----------------------------#
class DETA:
    def __init__(self):
        self.name = "DETA"
        self.sn = "DETA"
        self.formula = "C4H13N3"
        self.mw = 103.169
        self.density = 0.955
        self.prgmw = 51.5845
        self.srgmw = 0
        self.comp = (self.prgmw, self.srgmw, self.prgmw)
        self.mass = self.mw
        self.rg = "Amine"
        self.numgroups = len(set(self.comp))
        self.prgID = "NH₂"
        self.Nprg = 2
        self.prgk = 1
        self.cprgID = "NH"
        self.Ncprg = 0
        self.cprgk = 0
        self.srgID = "NH"
        self.Nsrg = 1
        self.srgk = 0
        self.csrgID = "N"
        self.Ncsrg = 0
        self.csrgk = 0
        self.trgID = None
        self.Ntrg = 0
        self.trgk = 0
        self.ctrgID = None
        self.Nctrg = 0
        self.ctrgk = 0
        self.dist = [[self.prgID], [self.srgID], [self.prgID]]

class LTETA:
    def __init__(self):
        self.name = "LTETA"
        self.sn = "LTETA"
        self.formula = "C6H18N4"
        self.mw = 146.234
        self.density = 0.982
        self.prgmw = 30.049
        self.srgmw = 43.068
        self.comp = (self.prgmw, self.srgmw, self.srgmw,self.prgmw)
        self.mass = self.mw
        self.rg = "Amine"
        self.numgroups = len(set(self.comp))
        self.prgID = "NH₂"
        self.prgk = 1
        self.cprgID = "NH"
        self.cprgk = 0
        self.srgID = "NH"
        self.srgk = 0
        self.csrgID = "N"
        self.csrgk = 0
        self.trgID = None
        self.trgk = 0
        self.ctrgID = None
        self.ctrgk = 0


class DEA:
    def __init__(self):
        self.name = "Diethanolamine"
        self.sn = "DEA"
        self.formula = "C4H11NO2"
        self.mw = 105.14
        self.density = 0.955
        self.prgmw = 43.068
        self.srgmw = 31.034
        self.comp = (self.srgmw, self.prgmw, self.srgmw)
        self.compid = ("Alcohol", "Amine", "Alcohol")
        self.mass = self.mw
        self.rg = "Amine"
        self.numgroups = len(set(self.comp))
        self.prgID = "NH₂"
        self.prgk = 1
        self.cprgID = "NH"
        self.cprgk = 0
        self.srgID = "OH"
        self.srgk = 0
        self.csrgID = "O"
        self.csrgk = 0
        self.trgID = None
        self.trgk = 0
        self.ctrgID = None
        self.ctrgk = 0
        self.dist = [[self.prgID, self.srgID, self.prgID]]

class Lysinol:
    def __init__(self):
        self.name = "Lysinol"
        self.sn = "Lys"
        self.formula = "C6H16NO2"
        self.mw = 132.20
        self.density = 1.1
        self.prgmw = 16.023
        self.srgmw = 100.159
        self.comp = (self.prgmw,self.prgmw,self.srgmw)
        self.compid = ("Amine","Amine","Alcohol")
        self.mass = self.mw
        self.rg = "Amine"

class Lysine:
    def __init__(self):
        self.name = "Lysine"
        self.sn = "Lys"
        self.mw = 146.19
        self.formula = "C6H14N2O2"
        self.density = 1.28
        self.prgmw = 72.1289
        self.srgmw = 21.04122
        self.trgmw = 45.01744
        self.comp = (self.prgmw,self.srgmw,self.trgmw)
        self.mass = self.mw
        self.prgk = 1
        self.srgk = 1
        self.trgk = 0
        self.crgk = 0
        self.prgID = "NH₂"
        self.srgID = "NH₂"
        self.trgID = "COOH"


#---------------------------------------------------------Acids------------------------------------------------#

class Adipic_Acid:
    def __init__(self):
        self.name = "Adipic Acid"
        self.sn = "AA"
        self.formula = "C6H10O4"
        self.mw = 146.14
        self.density = 1.36
        self.prgmw = 73.07
        self.srgmw = 0
        self.comp = (self.prgmw, self.prgmw)
        self.mass = self.mw
        self.rg = "Acid"

class Isostearic_Acid:
    def __init__(self):
        self.name = "Isostearic Acid"
        self.sn = "ISA"
        self.formula = "C18H36O2"
        self.mw = 284.48
        self.prgmw = 284.48
        self.srgmw = 0
        self.density = 0.93
        self.mass = self.mw
        self.comp = self.prgmw
        self.numgroups = 1
        self.mass = self.mw
        self.name = "C181"
        self.sn = "C181"
        self.formula = "C18H34O2"
        self.mw = 282.47
        self.prgmw = 282.47
        self.srgmw = 0
        self.density = 0.895
        self.mass = self.mw
        self.comp = self.prgmw
        self.rg = "Acid"
        self.mass = self.mw
        self.prgID = "COOH"
        self.prgk = 1
        self.cprgID = None
        self.cprgk = 0
        self.srgID = None
        self.srgk = 0
        self.csrgID = None
        self.csrgk = 0
        self.trgID = None
        self.trgk = 0
        self.ctrgID = None
        self.ctrgk = 0
        self.dist = [[self.prgID]]

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

class C181:
    def __init__(self):
        self.name = "C181"
        self.sn = "C181"
        self.formula = "C18H34O2"
        self.mw = 282.47
        self.prgmw = 282.47
        self.srgmw = 0
        self.density = 0.895
        self.mass = self.mw
        self.comp = self.prgmw
        self.rg = "Acid"
        self.mass = self.mw
        self.prgID = "COOH"
        self.Nprg = 1
        self.prgk = 1
        self.cprgID = None
        self.Ncprg = 0
        self.cprgk = 0
        self.srgID = None
        self.Nsrg = 0
        self.srgk = 0
        self.csrgID = None
        self.Ncsrg = 0
        self.csrgk = 0
        self.trgID = None
        self.Ntrg = 0
        self.trgk = 0
        self.ctrgID = None
        self.Nctrg = 0
        self.ctrgk = 0
        self.dist = [[self.prgID]]


#-------------------------------------Other--------------------------------------#

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

class Water:
    def __init__(self):
        self.name = "Water"
        self.sn = "H2O"
        self.formula = "H2O"
        self.mw = 18.01528
        self.density = 1.000
        self.prg = 1
        self.srg = 0
        self.tg = self.prg + self.srg

class Epichlorohydrin:
    def __init__(self):
        self.name = "Epichlorohydrin"
        self.sn = "Epi"
        self.formula = "C3H5ClO"
        self.mw = 92.52
        self.density = 1.18
        self.prgmw = 92.52
        self.srgmw = 0
        self.comp = self.prgmw
        self.numgroups = 1
        self.prgID = "COC"
        self.srgID = "Cl"

class Clear:
    def __init__(self):
        self.name = "Clear"

Reactants = [x for x in dir(Database) if isclass(getattr(Database, x))]











