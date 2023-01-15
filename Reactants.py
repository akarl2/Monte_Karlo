class R1Data:
    def __init__(self):
        self.dist = None
        self.sn = None
        self.comp = None
        self.mw = None
        self.name = None
        self.species = None
        self.mass = None
        self.moles = None
        self.prgID = None
        self.prgk = None
        self.cprgID = None
        self.cprgk = None
        self.srgID = None
        self.srgk = None
        self.csrgID = None
        self.csrgk = None
        self.trgID = None
        self.trgk = None
        self.ctrgID = None
        self.ctrgk = None
        self.Nprg = None
        self.Ncprg = None
        self.Nsrg = None
        self.Ncsrg = None
        self.Ntrg = None
        self.Nctrg = None

    def assign(self, name, mass, moles, prgID, prgk, cprgID, cprgk, srgID, srgk, csrgID, csrgk, trgID, trgk, ctrgID, ctrgk):
        self.name = name
        self.species = name.name
        self.sn = name.sn
        self.mw = name.mw
        self.mass = mass
        self.moles = moles
        self.prgID = [prgID, float(prgk)]
        self.cprgID =[cprgID, float(cprgk)]
        self.srgID = [srgID, float(srgk)]
        self.csrgID = [csrgID, float(csrgk)]
        self.trgID = [trgID, float(trgk)]
        self.ctrgID = [ctrgID, float(ctrgk)]
        self.dist = name.dist
        self.comp = [self.dist, [self.mw], [[self.sn, 1]]]

        for i in range(len(self.dist[0])):
            print(self.comp[0][i][0])
            if self.comp[0][i][0] == self.prgID[0]:
                self.comp[0][i] = self.prgID
            if self.comp[0][i][0] == self.srgID[0]:
                self.comp[0][i] = self.srgID
            if self.comp[0][i][0] == self.trgID[0]:
                self.comp[0][i] = self.trgID
        print(self.comp)

class R2Data:
    def __init__(self):
        self.dist = None
        self.sn = None
        self.comp = None
        self.mw = None
        self.name = None
        self.species = None
        self.mass = None
        self.moles = None
        self.prgID = None
        self.prgk = None
        self.cprgID = None
        self.cprgk = None
        self.srgID = None
        self.srgk = None
        self.csrgID = None
        self.csrgk = None
        self.trgID = None
        self.trgk = None
        self.ctrgID = None
        self.ctrgk = None
        self.Nprg = None
        self.Ncprg = None
        self.Nsrg = None
        self.Ncsrg = None
        self.Ntrg = None
        self.Nctrg = None

    def assign(self, name, mass, moles, prgID, prgk, cprgID, cprgk, srgID, srgk, csrgID, csrgk, trgID, trgk, ctrgID, ctrgk):
        self.name = name
        self.species = name.name
        self.sn = name.sn
        self.mw = name.mw
        self.mass = mass
        self.moles = moles
        self.prgID = [prgID, float(prgk)]
        self.cprgID =[cprgID, float(cprgk)]
        self.srgID = [srgID, float(srgk)]
        self.csrgID = [csrgID, float(csrgk)]
        self.trgID = [trgID, float(trgk)]
        self.ctrgID = [ctrgID, float(ctrgk)]
        self.dist = name.dist
        self.comp = [self.dist, [self.mw], [[self.sn, 1]]]

        for i in range(len(self.comp[0])):
            if self.comp[0][i][0] == self.prgID[0]:
                self.comp[0][i] = self.prgID
            if self.comp[0][i][0] == self.srgID[0]:
                self.comp[0][i]= self.srgID
            if self.comp[0][i][0] == self.trgID[0]:
                self.comp[0][i] = self.trgID
        print(self.comp)

class R3Data:
    def __init__(self):
        self.sn = None
        self.comp = None
        self.mw = None
        self.name = None
        self.species = None
        self.mass = None
        self.moles = None
        self.prgID = None
        self.prgk = None
        self.cprgID = None
        self.cprgk = None
        self.srgID = None
        self.srgk = None
        self.csrgID = None
        self.csrgk = None
        self.trgID = None
        self.trgk = None
        self.ctrgID = None
        self.ctrgk = None
        self.Nprg = None
        self.Ncprg = None
        self.Nsrg = None
        self.Ncsrg = None
        self.Ntrg = None
        self.Nctrg = None

    def assign(self, name, mass, moles, prgID, prgk, cprgID, cprgk, srgID, srgk, csrgID, csrgk, trgID, trgk, ctrgID, ctrgk):
        self.name = name
        self.species = name.name
        self.sn = name.sn
        self.mw = name.mw
        self.mass = mass
        self.moles = moles
        self.prgID = prgID
        self.Nprg = name.Nprg
        self.prgk = float(prgk)
        self.cprgID = cprgID
        self.Ncprg = name.Ncprg
        self.cprgk = float(cprgk)
        self.srgID = srgID
        self.Nsrg = name.Nsrg
        self.srgk = float(srgk)
        self.csrgID = csrgID
        self.Ncsrg = name.Ncsrg
        self.csrgk = float(csrgk)
        self.trgID = trgID
        self.Ntrg = name.Ntrg
        self.trgk = float(trgk)
        self.ctrgID = ctrgID
        self.Nctrg = name.Nctrg
        self.ctrgk = float(ctrgk)
        self.comp = [[[self.prgID, self.Nprg, self.prgk], [self.cprgID, self.Ncprg, self.cprgk], [self.srgID, self.Nsrg, self.srgk],
                      [self.csrgID, self.Ncsrg, self.csrgk], [self.trgID, self.Ntrg, self.trgk], [self.ctrgID, self.Nctrg, self.ctrgk]], [self.mw], [[self.sn, 1]]]
        #remove none and 0 from list
        for i in range(len(self.comp[0]) - 1, -1, -1):
            if self.comp[0][i][0] == "None":
                self.comp[0].pop(i)
        for i in range(len(self.comp[0]) - 1, -1, -1):
            if self.comp[0][i][2] == 0:
                self.comp[0].pop(i)
        print(self.comp)

class R4Data:
    def __init__(self):
        self.species = None
        self.name = None
        self.mass = None
        self.moles = None
        self.prgID = None
        self.prgk = None
        self.cprgID = None
        self.cprgk = None
        self.srgID = None
        self.srgk = None
        self.csrgID = None
        self.csrgk = None
        self.trgID = None
        self.trgk = None
        self.ctrgID = None
        self.ctrgk = None

    def assign(self, name, mass, moles, prgID, prgk, cprgID, cprgk, srgID, srgk, csrgID, csrgk, trgID, trgk, ctrgID, ctrgk):
        self.name = name
        self.species = name.name
        self.mass = mass
        self.moles = moles
        self.prgID = prgID
        self.prgk = prgk
        self.cprgID = cprgID
        self.cprgk = cprgk
        self.srgID = srgID
        self.srgk = srgk
        self.csrgID = csrgID
        self.csrgk = csrgk
        self.trgID = trgID
        self.trgk = trgk
        self.ctrgID = ctrgID
        self.ctrgk = ctrgk
class R5Data:
    def __init__(self):
        self.species = None
        self.name = None
        self.mass = None
        self.moles = None
        self.prgID = None
        self.prgk = None
        self.cprgID = None
        self.cprgk = None
        self.srgID = None
        self.srgk = None
        self.csrgID = None
        self.csrgk = None
        self.trgID = None
        self.trgk = None
        self.ctrgID = None
        self.ctrgk = None

    def assign(self, name, mass, moles, prgID, prgk, cprgID, cprgk, srgID, srgk, csrgID, csrgk, trgID, trgk, ctrgID, ctrgk):
        self.name = name
        self.species = name.name
        self.mass = mass
        self.moles = moles
        self.prgID = prgID
        self.prgk = prgk
        self.cprgID = cprgID
        self.cprgk = cprgk
        self.srgID = srgID
        self.srgk = srgk
        self.csrgID = csrgID
        self.csrgk = csrgk
        self.trgID = trgID
        self.trgk = trgk
        self.ctrgID = ctrgID
        self.ctrgk = ctrgk
class R6Data:
    def __init__(self):
        self.species = None
        self.name = None
        self.mass = None
        self.moles = None
        self.prgID = None
        self.prgk = None
        self.cprgID = None
        self.cprgk = None
        self.srgID = None
        self.srgk = None
        self.csrgID = None
        self.csrgk = None
        self.trgID = None
        self.trgk = None
        self.ctrgID = None
        self.ctrgk = None

    def assign(self, name, mass, moles, prgID, prgk, cprgID, cprgk, srgID, srgk, csrgID, csrgk, trgID, trgk, ctrgID, ctrgk):
        self.name = name
        self.species = name.name
        self.mass = mass
        self.moles = moles
        self.prgID = prgID
        self.prgk = prgk
        self.cprgID = cprgID
        self.cprgk = cprgk
        self.srgID = srgID
        self.srgk = srgk
        self.csrgID = csrgID
        self.csrgk = csrgk
        self.trgID = trgID
        self.trgk = trgk
        self.ctrgID = ctrgID
        self.ctrgk = ctrgk
class R7Data:
    def __init__(self):
        self.species = None
        self.name = None
        self.mass = None
        self.moles = None
        self.prgID = None
        self.prgk = None
        self.cprgID = None
        self.cprgk = None
        self.srgID = None
        self.srgk = None
        self.csrgID = None
        self.csrgk = None
        self.trgID = None
        self.trgk = None
        self.ctrgID = None
        self.ctrgk = None

    def assign(self, name, mass, moles, prgID, prgk, cprgID, cprgk, srgID, srgk, csrgID, csrgk, trgID, trgk, ctrgID, ctrgk):
        self.name = name
        self.species = name.name
        self.mass = mass
        self.moles = moles
        self.prgID = prgID
        self.prgk = prgk
        self.cprgID = cprgID
        self.cprgk = cprgk
        self.srgID = srgID
        self.srgk = srgk
        self.csrgID = csrgID
        self.csrgk = csrgk
        self.trgID = trgID
        self.trgk = trgk
        self.ctrgID = ctrgID
        self.ctrgk = ctrgk
class R8Data:
    def __init__(self):
        self.species = None
        self.name = None
        self.mass = None
        self.moles = None
        self.prgID = None
        self.prgk = None
        self.cprgID = None
        self.cprgk = None
        self.srgID = None
        self.srgk = None
        self.csrgID = None
        self.csrgk = None
        self.trgID = None
        self.trgk = None
        self.ctrgID = None
        self.ctrgk = None

    def assign(self, name, mass, moles, prgID, prgk, cprgID, cprgk, srgID, srgk, csrgID, csrgk, trgID, trgk, ctrgID, ctrgk):
        self.name = name
        self.species = name.name
        self.mass = mass
        self.moles = moles
        self.prgID = prgID
        self.prgk = prgk
        self.cprgID = cprgID
        self.cprgk = cprgk
        self.srgID = srgID
        self.srgk = srgk
        self.csrgID = csrgID
        self.csrgk = csrgk
        self.trgID = trgID
        self.trgk = trgk
        self.ctrgID = ctrgID
        self.ctrgk = ctrgk
class R9Data:
    def __init__(self):
        self.species = None
        self.name = None
        self.mass = None
        self.moles = None
        self.prgID = None
        self.prgk = None
        self.cprgID = None
        self.cprgk = None
        self.srgID = None
        self.srgk = None
        self.csrgID = None
        self.csrgk = None
        self.trgID = None
        self.trgk = None
        self.ctrgID = None
        self.ctrgk = None

    def assign(self, name, mass, moles, prgID, prgk, cprgID, cprgk, srgID, srgk, csrgID, csrgk, trgID, trgk, ctrgID, ctrgk):
        self.name = name
        self.species = name.name
        self.mass = mass
        self.moles = moles
        self.prgID = prgID
        self.prgk = prgk
        self.cprgID = cprgID
        self.cprgk = cprgk
        self.srgID = srgID
        self.srgk = srgk
        self.csrgID = csrgID
        self.csrgk = csrgk
        self.trgID = trgID
        self.trgk = trgk
        self.ctrgID = ctrgID
        self.ctrgk = ctrgk
class R10Data:
    def __init__(self):
        self.name = None
        self.mass = None
        self.moles = None
        self.prgID = None
        self.prgk = None
        self.cprgID = None
        self.cprgk = None
        self.srgID = None
        self.srgk = None
        self.csrgID = None
        self.csrgk = None
        self.trgID = None
        self.trgk = None
        self.ctrgID = None
        self.ctrgk = None

    def assign(self, name, mass, moles, prgID, prgk, cprgID, cprgk, srgID, srgk, csrgID, csrgk, trgID, trgk, ctrgID, ctrgk):
        self.name = name
        self.species = name.name
        self.mass = mass
        self.moles = moles
        self.prgID = prgID
        self.prgk = prgk
        self.cprgID = cprgID
        self.cprgk = cprgk
        self.srgID = srgID
        self.srgk = srgk
        self.csrgID = csrgID
        self.csrgk = csrgk
        self.trgID = trgID
        self.trgk = trgk
        self.ctrgID = ctrgID
        self.ctrgk = ctrgk
class R11Data:
    def __init__(self):
        self.species = None
        self.name = None
        self.mass = None
        self.moles = None
        self.prgID = None
        self.prgk = None
        self.cprgID = None
        self.cprgk = None
        self.srgID = None
        self.srgk = None
        self.csrgID = None
        self.csrgk = None
        self.trgID = None
        self.trgk = None
        self.ctrgID = None
        self.ctrgk = None

    def assign(self, name, mass, moles, prgID, prgk, cprgID, cprgk, srgID, srgk, csrgID, csrgk, trgID, trgk, ctrgID, ctrgk):
        self.name = name
        self.species = name.name
        self.mass = mass
        self.moles = moles
        self.prgID = prgID
        self.prgk = prgk
        self.cprgID = cprgID
        self.cprgk = cprgk
        self.srgID = srgID
        self.srgk = srgk
        self.csrgID = csrgID
        self.csrgk = csrgk
        self.trgID = trgID
        self.trgk = trgk
        self.ctrgID = ctrgID
        self.ctrgk = ctrgk
class R12Data:
    def __init__(self):
        self.species = None
        self.name = None
        self.mass = None
        self.moles = None
        self.prgID = None
        self.prgk = None
        self.cprgID = None
        self.cprgk = None
        self.srgID = None
        self.srgk = None
        self.csrgID = None
        self.csrgk = None
        self.trgID = None
        self.trgk = None
        self.ctrgID = None
        self.ctrgk = None

    def assign(self, name, mass, moles, prgID, prgk, cprgID, cprgk, srgID, srgk, csrgID, csrgk, trgID, trgk, ctrgID, ctrgk):
        self.name = name
        self.species = name.name
        self.mass = mass
        self.moles = moles
        self.prgID = prgID
        self.prgk = prgk
        self.cprgID = cprgID
        self.cprgk = cprgk
        self.srgID = srgID
        self.srgk = srgk
        self.csrgID = csrgID
        self.csrgk = csrgk
        self.trgID = trgID
        self.trgk = trgk
        self.ctrgID = ctrgID
        self.ctrgk = ctrgk
class R13Data:
    def __init__(self):
        self.species = None
        self.name = None
        self.mass = None
        self.moles = None
        self.prgID = None
        self.prgk = None
        self.cprgID = None
        self.cprgk = None
        self.srgID = None
        self.srgk = None
        self.csrgID = None
        self.csrgk = None
        self.trgID = None
        self.trgk = None
        self.ctrgID = None
        self.ctrgk = None

    def assign(self, name, mass, moles, prgID, prgk, cprgID, cprgk, srgID, srgk, csrgID, csrgk, trgID, trgk, ctrgID, ctrgk):
        self.name = name
        self.species = name.name
        self.mass = mass
        self.moles = moles
        self.prgID = prgID
        self.prgk = prgk
        self.cprgID = cprgID
        self.cprgk = cprgk
        self.srgID = srgID
        self.srgk = srgk
        self.csrgID = csrgID
        self.csrgk = csrgk
        self.trgID = trgID
        self.trgk = trgk
        self.ctrgID = ctrgID
        self.ctrgk = ctrgk
class R14Data:
    def __init__(self):
        self.species = None
        self.name = None
        self.mass = None
        self.moles = None
        self.prgID = None
        self.prgk = None
        self.cprgID = None
        self.cprgk = None
        self.srgID = None
        self.srgk = None
        self.csrgID = None
        self.csrgk = None
        self.trgID = None
        self.trgk = None
        self.ctrgID = None
        self.ctrgk = None

    def assign(self, name, mass, moles, prgID, prgk, cprgID, cprgk, srgID, srgk, csrgID, csrgk, trgID, trgk, ctrgID, ctrgk):
        self.name = name
        self.species = name.name
        self.mass = mass
        self.moles = moles
        self.prgID = prgID
        self.prgk = prgk
        self.cprgID = cprgID
        self.cprgk = cprgk
        self.srgID = srgID
        self.srgk = srgk
        self.csrgID = csrgID
        self.csrgk = csrgk
        self.trgID = trgID
        self.trgk = trgk
        self.ctrgID = ctrgID
        self.ctrgk = ctrgk

#Make all classes global
R1 = R1Data()


