
#Create a dictionary of all the above classes
reactantsA = ["Epichlorohydrin","Butanol", "Glycerol", "PPG425", "Sorbitol", "Propylene_Glycol", "Pentaerythritol", "Butanediol", "Trimethylolpropane", "C181", "Water", "Ethanol", "Methanol", "Acetone", "Acetic Acid", "Formic_Acid", "PAA", "DETA", "Adipic_Acid", "Cyclohexanedimethanol","Ethylhexanol", "Pripol", "Isostearic_Acid", "DEA", "Lysinol"]
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

class Propylene_Glycol:
    def __init__(self):
        self.name = "Propylene_Glycol"
        self.sn = "PG"
        self.formula = "C3H8O2"
        self.mw = 76.095
        self.density = 1.04
        self.prgmw = self.mw / 2
        self.srgmw = 0
        self.comp = (self.prgmw,self.prgmw)
        self.mass = self.mw

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

class Ethylhexanol:
    def __init__(self):
        self.name = "2-Ethylhexanol"
        self.sn = "2-EH"
        self.formula = "C8H18O"
        self.mw = 130.23
        self.density = 0.833
        self.prg = Carboxyl
        self.mass = self.mw
        self.comp = self.prg

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
        self.comp = (self.prgmw, self.prgmw)
        self.mass = self.mw
        self.rg = "Amine"

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

class DETA_LTETA:
    def __init__(self):
        self.name_1 = "DETA"
        self.name_2 = "LTETA"
        self.sn_1 = "DETA"
        self.sn_2 = "LTETA"
        self.formula_1 = "C4H13N3"
        self.formula_2 = "C6H18N4"
        self.mw_1 = 103.169
        self.mw_2 = 146.234
        self.density_1 = 0.955
        self.density_2 = 0.982
        self.prgmw_1 = 51.5845
        self.srgmw_1 = 0
        self.prgmw_2 = 30.049
        self.srgmw_2 = 43.068
        self.comp_1 = (self.prgmw_1, self.prgmw_1)
        self.comp_2 = (self.prgmw_2, self.srgmw_2, self.srgmw_2,self.prgmw_2)
        self.mass_1 = self.mw_1
        self.mass_2 = self.mw_2
        self.rg_1 = "Amine"
        self.rg_2 = "Amine"


class LTETA:
    def __init__(self):
        self.name = "Liner TETA"
        self.sn = "L-TETA"
        self.formula = "C6H18N4"
        self.mw = 146.234
        self.density = 0.982
        self.prgmw = 30.0492
        self.srgmw = 43.0678
        self.comp = (self.prgmw, self.srgmw, self.srgmw, self.prgmw)
        self.mass = self.mw
        self.rg = "Amine"

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
        self.prg = Carboxyl
        self.mass = self.mw
        self.comp = self.prg

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
        self.prg = Carboxyl
        self.mass = self.mw
        self.comp = self.prg
        self.rg = "Acid"


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

#-------------------------------Functional Groups--------------------------------#

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

class Ester:
    def __init__(self):
        self.name = "Ester"
        self.rxn = P_Amine

class Amide:
    def __init__(self):
        self.name = "Amide"








