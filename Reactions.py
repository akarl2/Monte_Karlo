from Database import *

class Esterification:
    def __init__(self):
        self.wl = Water().mw
        self.rp = Ester
        P_Hydroxyl.wl = 1.0079
        S_Hydroxyl.wl = 1.0079
        Carboxyl.wl = 17.007


class Etherification:
    def __init__(self, reactants, products, rate):
        self.reactants = reactants
        self.products = products
        self.rate = rate

