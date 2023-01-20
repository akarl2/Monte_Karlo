class NH2:
    def __init__(self):
        self.COOH = "CONH"
        self.COC = 'COCNH'
        self.Cl = 'ClNH'

reactive_group = 'NH2'
check_group = 'COOH'

def new_group(reactive_group, check_group):
    NG = getattr(eval(reactive_group + '()'), check_group)
    print(NG)

new_group(reactive_group, check_group)