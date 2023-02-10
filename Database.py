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
        self.prgID = "POH"
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

class Glycerol:
    def __init__(self):
        self.name = "Glycerol"
        self.sn = "Gly"
        self.formula = "C3H6O3"
        self.mw = 92.09382
        self.density = 1.26
        self.mass = self.mw
        self.prgID = "POH"
        self.prgk = 1
        self.cprgID = None
        self.cprgk = 0
        self.srgID = "SOH"
        self.srgk = 1
        self.csrgID = None
        self.csrgk = 0
        self.trgID = None
        self.trgk = 0
        self.ctrgID = None
        self.ctrgk = 0
        self.dist = [[self.prgID], [self.srgID], [self.prgID]]

class Neopentyl_glycol:
    def __init__(self):
        self.name = "Neopentyl_glycol"
        self.sn = "NPG"
        self.formula = "C5H12O2"
        self.mw = 104.148
        self.density = 1.26
        self.mass = self.mw
        self.prgID = "POH"
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
        self.dist = [[self.prgID], [self.prgID]]

class Castor_Oil:
    def __init__(self):
        self.name = "Castor_Oil"
        self.sn = "Castor_Oil"
        self.formula = "C3H6O3"
        self.mw = 1032.52
        self.density = 1.26
        self.mass = self.mw
        self.prgID = "SOH"
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
        self.dist = [[self.prgID], [self.prgID], [self.prgID]]

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
        self.mass = self.mw
        self.prgID = "POH"
        self.prgk = 1
        self.cprgID = None
        self.cprgk = 0
        self.srgID = "SOH"
        self.srgk = 1
        self.csrgID = None
        self.csrgk = 0
        self.trgID = None
        self.trgk = 0
        self.ctrgID = None
        self.ctrgk = 0
        self.dist = [[self.prgID], [self.srgID], [self.srgID], [self.srgID], [self.srgID], [self.prgID]]

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
        self.prgID = "POH"
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
        self.dist = [[self.prgID], [self.prgID], [self.prgID]]

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

class Bisphenol_A:
    def __init__(self):
        self.name = "Bisphenol_A"
        self.sn = "BPA"
        self.formula = "C15H16O2"
        self.mw = 228.29
        self.density = 1.2
        self.mass = self.mw
        self.prgID = "POH"
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
        self.dist = [[self.prgID], [self.prgID]]

#-------------------------------------------Amines----------------------------#
class DETA:
    def __init__(self):
        self.name = "DETA"
        self.sn = "DETA"
        self.formula = "C4H13N3"
        self.mw = 103.169
        self.density = 0.955
        self.mass = self.mw
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
        self.dist = [[self.prgID], [self.srgID], [self.prgID]]

class LTEPA:
    def __init__(self):
        self.name = "LTEPA"
        self.sn = "LTEPA"
        self.formula = "C4H13N3"
        self.mw = 189.307
        self.density = 0.955
        self.mass = self.mw
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
        self.dist = [[self.prgID], [self.srgID], [self.srgID], [self.srgID], [self.prgID]]

class AETETA:
    def __init__(self):
        self.name = "AETETA"
        self.sn = "AETETA"
        self.formula = "C4H13N3"
        self.mw = 189.307
        self.density = 0.955
        self.mass = self.mw
        self.prgID = "NH₂"
        self.prgk = 1
        self.cprgID = "NH"
        self.cprgk = 0
        self.srgID = "NH"
        self.srgk = 0
        self.csrgID = "N"
        self.csrgk = 0
        self.trgID = "N"
        self.trgk = 0
        self.ctrgID = None
        self.ctrgk = 0
        self.dist = [[self.prgID], [self.trgID], [self.prgID], [self.srgID], [self.prgID]]

class AEPEEDA:
    def __init__(self):
        self.name = "AEPEEDA"
        self.sn = "AEPEEDA"
        self.formula = "C4H13N3"
        self.mw = 216.35
        self.density = 0.955
        self.mass = self.mw
        self.prgID = "NH₂"
        self.prgk = 1
        self.cprgID = "NH"
        self.cprgk = 0
        self.srgID = "NH"
        self.srgk = 0
        self.csrgID = "N"
        self.csrgk = 0
        self.trgID = "N"
        self.trgk = 0
        self.ctrgID = None
        self.ctrgk = 0
        self.dist = [[self.prgID], [self.trgID], [self.trgID], [self.srgID], [self.prgID]]

class PEDETA:
    def __init__(self):
        self.name = "PEDETA"
        self.sn = "PEDETA"
        self.formula = "C4H13N3"
        self.mw = 216.35
        self.density = 0.955
        self.mass = self.mw
        self.prgID = "NH₂"
        self.prgk = 1
        self.cprgID = "NH"
        self.cprgk = 0
        self.srgID = "NH"
        self.srgk = 0
        self.csrgID = "N"
        self.csrgk = 0
        self.trgID = "N"
        self.trgk = 0
        self.ctrgID = None
        self.ctrgk = 0
        self.dist = [[self.srgID], [self.trgID], [self.srgID], [self.srgID], [self.prgID]]

class PEHA:
    def __init__(self):
        self.name = "PEHA"
        self.sn = "PEHA"
        self.formula = "C10H26N6"
        self.mw = 232.376
        self.density = 0.955
        self.mass = self.mw
        self.prgID = "NH₂"
        self.prgk = 1
        self.cprgID = "NH"
        self.cprgk = 0
        self.srgID = "NH"
        self.srgk = 0
        self.csrgID = "N"
        self.csrgk = 0
        self.trgID = "N"
        self.trgk = 0
        self.ctrgID = None
        self.ctrgk = 0
        self.dist = [[self.prgID], [self.srgID], [self.srgID], [self.srgID], [self.srgID], [self.prgID]]


class LTETA:
    def __init__(self):
        self.name = "LTETA"
        self.sn = "LTETA"
        self.formula = "C6H18N4"
        self.mw = 146.234
        self.density = 0.982
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
        self.dist = [[self.prgID], [self.srgID], [self.srgID], [self.prgID]]

class Branched_TETA:
    def __init__(self):
        self.name = "Branched TETA"
        self.sn = "BTETA"
        self.formula = "C6H18N4"
        self.mw = 146.234
        self.density = 0.982
        self.prgID = "NH₂"
        self.prgk = 1
        self.cprgID = "NH"
        self.cprgk = 0
        self.srgID = "NH"
        self.srgk = 0
        self.csrgID = "N"
        self.csrgk = 0
        self.trgID = "N"
        self.trgk = 0
        self.ctrgID = None
        self.ctrgk = 0
        self.dist = [[self.prgID], [self.trgID], [self.prgID], [self.prgID]]

class Bis_AEP:
    def __init__(self):
        self.name = "Bis-AEP"
        self.sn = "Bis-AEP"
        self.formula = "C6H16N4"
        self.mw = 168.24
        self.density = 0.982
        self.prgID = "NH₂"
        self.prgk = 1
        self.cprgID = "NH"
        self.cprgk = 0
        self.srgID = "NH"
        self.srgk = 0
        self.csrgID = "N"
        self.csrgk = 0
        self.trgID = "N"
        self.trgk = 0
        self.ctrgID = None
        self.ctrgk = 0
        self.dist = [[self.prgID], [self.trgID], [self.trgID], [self.prgID]]

class PEEDA:
    def __init__(self):
        self.name = "PEEDA"
        self.sn = "PEEDA"
        self.formula = "C4H13N3"
        self.mw = 168.24
        self.density = 0.955
        self.prgID = "NH₂"
        self.prgk = 1
        self.cprgID = "NH"
        self.cprgk = 0
        self.srgID = "NH"
        self.srgk = 0
        self.csrgID = "N"
        self.csrgk = 0
        self.trgID = "N"
        self.trgk = 0
        self.ctrgID = None
        self.ctrgk = 0
        self.dist = [[self.srgID], [self.trgID], [self.srgID], [self.prgID]]

class Hydroxyethylpiperazine:
    def __init__(self):
        self.name = "Hydroxyethylpiperazine"
        self.sn = "HEP"
        self.formula = "C6H14N2O"
        self.mw = 130.1882
        self.density = 0.955
        self.prgID = "POH"
        self.prgk = 1
        self.cprgID = None
        self.cprgk = 0
        self.srgID = "NH"
        self.srgk = 0
        self.csrgID = None
        self.csrgk = 0
        self.trgID = "N"
        self.trgk = 0
        self.ctrgID = None
        self.ctrgk = 0
        self.dist = [[self.prgID], [self.srgID], [self.trgID]]

class DEA:
    def __init__(self):
        self.name = "Diethanolamine"
        self.sn = "DEA"
        self.formula = "C4H11NO2"
        self.mw = 105.14
        self.density = 0.955
        self.mass = self.mw
        self.prgID = "NH₂"
        self.prgk = 1
        self.cprgID = "NH"
        self.cprgk = 0
        self.srgID = "SOH"
        self.srgk = 0
        self.csrgID = None
        self.csrgk = 0
        self.trgID = None
        self.trgk = 0
        self.ctrgID = None
        self.ctrgk = 0
        self.dist = [[self.prgID], [self.srgID], [self.prgID]]

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
        self.prgID = "NH₂"
        self.prgk = 1
        self.cprgID = "NH"
        self.cprgk = 0
        self.srgID = "COOH"
        self.srgk = 0
        self.csrgID = None
        self.csrgk = 0
        self.trgID = None
        self.trgk = 0
        self.ctrgID = None
        self.ctrgk = 0
        self.dist = [[self.prgID], [self.srgID], [self.prgID]]

class PACM:
    def __init__(self):
        self.name = "PACM"
        self.sn = "PACM"
        self.formula = "C13H14N2"
        self.mw = 198.269
        self.density = 1.05
        self.prgID = "NH₂"
        self.prgk = 1
        self.cprgID = "NH"
        self.cprgk = 0
        self.srgID = None
        self.srgk = 0
        self.csrgID = None
        self.csrgk = 0
        self.trgID = None
        self.trgk = 0
        self.ctrgID = None
        self.ctrgk = 0
        self.dist = [[self.prgID], [self.prgID]]

class MXDA:
    def __init__(self):
        self.name = "MXDA"
        self.sn = "MXDA"
        self.formula = "C8H12N2"
        self.mw = 136.198
        self.density = 1.032
        self.prgID = "NH₂"
        self.prgk = 1
        self.cprgID = "NH"
        self.cprgk = 0
        self.srgID = None
        self.srgk = 0
        self.csrgID = None
        self.csrgk = 0
        self.trgID = None
        self.trgk = 0
        self.ctrgID = None
        self.ctrgk = 0
        self.dist = [[self.prgID], [self.prgID]]


#---------------------------------------------------------Acids------------------------------------------------#

class Adipic_Acid:
    def __init__(self):
        self.name = "Adipic Acid"
        self.sn = "AdAc"
        self.formula = "C6H10O4"
        self.mw = 146.14
        self.density = 1.36
        self.prgmw = 73.07
        self.srgmw = 0
        self.comp = (self.prgmw, self.prgmw)
        self.mass = self.mw
        self.rg = "Acid"
        self.numgroups = len(set(self.comp))
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
        self.dist = [[self.prgID],[self.prgID]]


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
        self.density = 0.895
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
        self.sn = "C18:1"
        self.formula = "C18H34O2"
        self.mw = 282.47
        self.density = 0.895
        self.mass = self.mw
        self.prgID = "COOH"
        self.prgk = 1
        self.cprgID = None
        self.cprgk = 0
        self.srgID = "CC_1"
        self.srgk = 0
        self.csrgID = None
        self.csrgk = 0
        self.trgID = None
        self.trgk = 0
        self.ctrgID = None
        self.ctrgk = 0
        self.dist = [[self.prgID], [self.srgID]]

class C14:
    def __init__(self):
        self.name = "C14"
        self.sn = "C14"
        self.formula = "C14H28O2"
        self.mw = 228.376
        self.density = 0.895
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

class C16:
    def __init__(self):
        self.name = "C16"
        self.sn = "C16"
        self.formula = "C16H32O2"
        self.mw = 256.43
        self.density = 0.895
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

class C18:
    def __init__(self):
        self.name = "C18"
        self.sn = "C18"
        self.formula = "C18H36O2"
        self.mw = 284.484
        self.density = 0.895
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

class C161:
    def __init__(self):
        self.name = "C161"
        self.sn = "C16:1"
        self.formula = "C16H30O2"
        self.mw = 254.414
        self.density = 0.895
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

class C182:
    def __init__(self):
        self.name = "C18"
        self.sn = "C18:2"
        self.formula = "C18H32O2"
        self.mw = 280.452
        self.density = 0.895
        self.mass = self.mw
        self.prgID = "COOH"
        self.prgk = 1
        self.cprgID = None
        self.cprgk = 0
        self.srgID = "CC_2"
        self.srgk = 0
        self.csrgID = None
        self.csrgk = 0
        self.trgID = "CC_1"
        self.trgk = 0
        self.ctrgID = None
        self.ctrgk = 0
        self.dist = [[self.prgID], [self.srgID], [self.trgID]]

class C183:
    def __init__(self):
        self.name = "C18"
        self.sn = "C18:3"
        self.formula = "C18H30O2"
        self.mw = 278.436
        self.density = 0.9164
        self.mass = self.mw
        self.prgID = "COOH"
        self.prgk = 1
        self.cprgID = None
        self.cprgk = 0
        self.srgID = "CC_3"
        self.srgk = 0
        self.csrgID = None
        self.csrgk = 0
        self.trgID = "CC_2"
        self.trgk = 0
        self.ctrgID = None
        self.ctrgk = 0
        self.dist = [[self.prgID], [self.srgID], [self.trgID]]

class Dimer:
    def __init__(self):
        self.name = "Dimer"
        self.sn = "Dimer"
        self.formula = "C18H30O2"
        self.mw = 581.347
        self.density = 0.9164
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
        self.dist = [[self.prgID], [self.prgID]]

class Trimer:
    def __init__(self):
        self.name = "Timer"
        self.sn = "Timer"
        self.formula = "C18H30O2"
        self.mw = 872.021
        self.density = 0.9164
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
        self.dist = [[self.prgID], [self.prgID], [self.prgID]]

class Azelaic_acid:
    def __init__(self):
        self.name = "Azelaic acid"
        self.sn = "AzAc"
        self.formula = "C9H16O4"
        self.mw = 188.22
        self.density = 1.443
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
        self.dist = [[self.prgID], [self.prgID]]

class Diethyl_maleate:
    def __init__(self):
        self.name = "Diethyl maleate"
        self.sn = "DEM"
        self.formula = "C8H12O4"
        self.mw = 172.108
        self.density = 1.01
        self.mass = self.mw
        self.prgID = "aB_unsat"
        self.prgk = 1
        self.cprgID = None
        self.cprgk = 0
        self.srgID = "COO"
        self.srgk = 0
        self.csrgID = None
        self.csrgk = 0
        self.trgID = None
        self.trgk = 0
        self.ctrgID = None
        self.ctrgk = 0
        self.dist = [[self.srgID], [self.prgID], [self.srgID]]

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
        self.prgID = "COC"
        self.prgk = 1
        self.cprgID = "SOH"
        self.cprgk = 0
        self.srgID = "Cl"
        self.srgk = 0
        self.csrgID = None
        self.csrgk = 0
        self.trgID = None
        self.trgk = 0
        self.ctrgID = None
        self.ctrgk = 0
        self.dist = [[self.prgID], [self.srgID]]

class Propylene_oxide:
    def __init__(self):
        self.name = "Propylene oxide"
        self.sn = "PO"
        self.formula = "C3H6O2"
        self.mw = 58.08
        self.density = 0.859
        self.prgID = "COC"
        self.prgk = 1
        self.cprgID = "SOH"
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

class HDODA:
    def __init__(self):
        self.name = "HDODA"
        self.sn = "HDODA"
        self.formula = "C12H18O4"
        self.mw = 226.27
        self.density = 1.12
        self.prgID = "aB_unsat"
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
        self.dist = [[self.prgID], [self.prgID]]

class CM67:
    def __init__(self):
        self.name = "CM67"
        self.sn = "CM67"
        self.formula = "C12H18O4"
        self.mw = 270
        self.density = 1.12
        self.prgID = "COC"
        self.prgk = 1
        self.cprgID = "SOH"
        self.cprgk = 0
        self.srgID = None
        self.srgk = 0
        self.csrgID = None
        self.csrgk = 0
        self.trgID = None
        self.trgk = 0
        self.ctrgID = None
        self.ctrgk = 0
        self.dist = [[self.prgID], [self.prgID]]

class BPADGE:
    def __init__(self):
        self.name = "BPADGE"
        self.sn = "BPADGE"
        self.formula = "C12H18O4"
        self.mw = 372
        self.density = 1.12
        self.prgID = "COC"
        self.prgk = 1
        self.cprgID = "SOH"
        self.cprgk = 0
        self.srgID = None
        self.srgk = 0
        self.csrgID = None
        self.csrgk = 0
        self.trgID = None
        self.trgk = 0
        self.ctrgID = None
        self.ctrgk = 0
        self.dist = [[self.prgID], [self.prgID]]



class Clear:
    def __init__(self):
        self.name = "Clear"

Reactants = [x for x in dir(Database) if isclass(getattr(Database, x))]












