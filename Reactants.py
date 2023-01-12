class R1Data:
    def __init__(self):
        self.name = None
        self.mass = None
        self.moles = None

    def assign(self, name, mass, moles):
        self.name = name
        self.mass = mass
        self.moles = moles
        print(self.name, self.mass, self.moles)



