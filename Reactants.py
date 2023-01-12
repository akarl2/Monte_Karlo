class R1Data:
    def __init__(self):
        self.name = None
        self.mass = None
        self.moles = None

    def assign(self, name, mass, moles, prgID, prgK, cprgk, srgID, srgK, csrgk, trgID, trgK, ctrgk):
        self.name = name
        self.mass = mass
        self.moles = moles
        self.prgID = prgID
        self.prgK = prgK
        self.cprgk = cprgk
        self.srgID = srgID
        self.srgK = srgK
        self.csrgk = csrgk
        self.trgID = trgID
        self.trgK = trgK
        self.ctrgk = ctrgk
        print(self.name, self.mass, self.moles)



