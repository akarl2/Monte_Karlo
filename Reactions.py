from Database import *

class Esterification:
    def __init__(self):
        self.wl = Water().mw
        self.rp = Ester
        P_Hydroxyl.wl = 1.0079
        S_Hydroxyl.wl = 1.0079
        Carboxyl.wl = 17.007


class Etherification:
    def __init__(self):
        self.wl = 0

class Amidation:
    def __init__(self):
        self.wl = Water().mw
        self.rp = Amide
        P_Hydroxyl.wl = 1.0079
        S_Hydroxyl.wl = 1.0079
        Carboxyl.wl = 17.007
