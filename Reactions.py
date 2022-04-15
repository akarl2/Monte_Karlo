from Database import *

class Esterification:
    def __init__(self):
        self.wl = Water().mw

class Etherification:
    def __init__(self, reactants, products, rate):
        self.reactants = reactants
        self.products = products
        self.rate = rate

