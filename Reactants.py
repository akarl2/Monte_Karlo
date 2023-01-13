class R1Data:
    def __init__(self):
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

    def assign(self, name, mass, moles, prgID, prgk, cprgID, cprgk, srgID, srgk, csrgID, csrgk, trgID, trgk, ctrgID, ctrgk):
        self.name = name
        self.species = name.name
        self.mw = name.mw
        self.mass = mass
        self.moles = moles
        self.prgID = prgID
        self.prgk = float(prgk)
        self.cprgID = cprgID
        self.cprgk = float(cprgk)
        self.srgID = srgID
        self.srgk = float(srgk)
        self.csrgID = csrgID
        self.csrgk = float(csrgk)
        self.trgID = trgID
        self.trgk = float(trgk)
        self.ctrgID = ctrgID
        self.ctrgk = float(ctrgk)
        React_test = [self.mw,[[self.prgID,self.prgk],[self.cprgID,self.cprgk],[self.srgID,self.srgk],[self.csrgID,self.csrgk],[self.trgID,self.trgk],[self.ctrgID,self.ctrgk]]]
        #remove none and 0 from list
        for i in range(len(React_test[1])-1,-1,-1):
            if React_test[1][i][0] == "None":
                React_test[1].pop(i)
        for i in range(len(React_test[1])-1,-1,-1):
            if React_test[1][i][1] == 0:
                React_test[1].pop(i)
        print(React_test)






class R2Data:
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
class R3Data:
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


