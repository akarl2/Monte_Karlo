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
        self.ct = None

    def assign(self, name, mass, moles, prgID, prgk, cprgID, cprgk, srgID, srgk, csrgID, csrgk, trgID, trgk, ctrgID, ctrgk, ct):
        self.name = name
        self.species = name.name
        self.sn = name.sn
        self.mw = name.mw
        self.mass = mass
        self.moles = round(moles, 4)
        self.prgID = [prgID, float(prgk)]
        self.cprgID =[cprgID, float(cprgk)]
        self.srgID = [srgID, float(srgk)]
        self.csrgID = [csrgID, float(csrgk)]
        self.trgID = [trgID, float(trgk)]
        self.ctrgID = [ctrgID, float(ctrgk)]
        self.dist = name.dist
        self.ct = int(self.moles * int(ct))
        index = 0
        for i in self.dist:
            if self.dist[index][0] == self.prgID[0]:
                self.dist[index] = self.prgID
            if self.dist[index][0] == self.srgID[0]:
                self.dist[index] = self.srgID
            if self.dist[index][0] == self.trgID[0]:
                self.dist[index] = self.trgID
            index += 1
        self.comp = [self.dist, [self.mw], [[self.sn, 1]], [self.ct]]

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
        self.ct = None

    def assign(self, name, mass, moles, prgID, prgk, cprgID, cprgk, srgID, srgk, csrgID, csrgk, trgID, trgk, ctrgID, ctrgk, ct):
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
        self.ct = int(self.moles * int(ct))
        index = 0
        for i in self.dist:
            if self.dist[index][0] == self.prgID[0]:
                self.dist[index] = self.prgID
            if self.dist[index][0] == self.srgID[0]:
                self.dist[index] = self.srgID
            if self.dist[index][0] == self.trgID[0]:
                self.dist[index] = self.trgID
            index += 1
        self.comp = [self.dist, [self.mw], [[self.sn, 1]], [self.ct]]

class R3Data:
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
        self.ct = None

    def assign(self, name, mass, moles, prgID, prgk, cprgID, cprgk, srgID, srgk, csrgID, csrgk, trgID, trgk, ctrgID, ctrgk, ct):
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
        self.ct = int(self.moles * int(ct))
        index = 0
        for i in self.dist:
            if self.dist[index][0] == self.prgID[0]:
                self.dist[index] = self.prgID
            if self.dist[index][0] == self.srgID[0]:
                self.dist[index] = self.srgID
            if self.dist[index][0] == self.trgID[0]:
                self.dist[index] = self.trgID
            index += 1
        self.comp = [self.dist, [self.mw], [[self.sn, 1]], [self.ct]]

class R4Data:
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
        self.ct = None

    def assign(self, name, mass, moles, prgID, prgk, cprgID, cprgk, srgID, srgk, csrgID, csrgk, trgID, trgk, ctrgID, ctrgk, ct):
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
        self.ct = int(self.moles * int(ct))
        index = 0
        for i in self.dist:
            if self.dist[index][0] == self.prgID[0]:
                self.dist[index] = self.prgID
            if self.dist[index][0] == self.srgID[0]:
                self.dist[index] = self.srgID
            if self.dist[index][0] == self.trgID[0]:
                self.dist[index] = self.trgID
            index += 1
        self.comp = [self.dist, [self.mw], [[self.sn, 1]], [self.ct]]

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


