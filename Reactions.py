from Database import *

class Condensation:
    def __init__(self):
        self.wl = Water().mw
        self.name = Condensation

class PolyCondensation:
    def __init__(self):
        self.wl = Water().mw
        self.name = PolyCondensation

class Etherification:
    def __init__(self):
        self.wl = 0
        self.name = Etherification

Reactions = ["Condensation", "PolyCondensation", "Etherification"]
End_Metrics = ["Amine_Value", "Acid_Value", "OH_Value"]
