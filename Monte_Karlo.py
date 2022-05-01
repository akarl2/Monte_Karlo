import collections
import random
import sys
import tkinter
import pandas
from Database import *
from Reactions import *
import itertools
import os

# Set pandas dataframe display
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)
pandas.set_option('display.width', 100)

# Converts string name to class name
def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

# specify reaction chemicals, reaction type and # of samples
a = Trimethylolpropane()
b = Epichlorohydrin()
rt = Etherification()
Samples = 1000

# Specify extend of reaction (EOR)
EOR = 1

# Starting material mass and moles
a.mass = 250
b.mass = 200

def simulate(a, b, rt, Samples, EOR, a_mass, b_mass):
    a = str_to_class(a)
    b = str_to_class(b)
    rt = str_to_class(rt)
    a.mol = round(a_mass / a.mw, 3) * Samples
    b.mol = round(b_mass / b.mw, 3) * Samples
    unreacted = b.mol - (EOR * b.mol)
    b.mol = EOR * b.mol
    bms = b.mol

    # Creates final product naRme(s) from starting material name(s)
    final_product_names = [a.sn, b.sn]
    final_product_names.extend([f"{a.sn}({1})_{b.sn}({str(i)})" for i in range(1, 1001)])

    # Creates final product molar masses from final product name(s)
    final_product_masses = ({a.sn: round(a.mw, 1), b.sn: round(b.mw, 1)})
    if rt.name != PolyCondensation:
        final_product_masses.update(
            {f"{a.sn}({1})_{b.sn}({str(i)})": round(a.mw + (i * b.mw - i * rt.wl), 1) for i in range(1, 1001)})
    elif rt.name == PolyCondensation:
        final_product_masses.update(
            {f"{a.sn}({i})_{b.sn}({str(i)})": round(i * a.mw + i * b.mw - (i + i - 1) * rt.wl, 2) for i in
             range(1, 1001)})
        final_product_masses.update(
            {f"{a.sn}({i - 1})_{b.sn}({str(i)})": round((i - 1) * a.mw + i * b.mw - (i + i - 2) * rt.wl, 1) for i in
             range(2, 1001)})
        final_product_masses.update(
            {f"{a.sn}({i})_{b.sn}({str(i - 1)})": round(i * a.mw + (i - 1) * b.mw - (i + i - 2) * rt.wl, 1) for i in
             range(2, 1001)})

    # Specify rate constants
    prgK = 1
    srgK = 1
    cgK = .06

    # Creates starting composition list
    composition = []
    try:
        for i in range(0, int(a.mol)):
            composition.extend(group for group in a.comp)
    except TypeError:
        for i in range(0, int(a.mol)):
            composition.append(a.mw)

    # Create weights from starting composition list
    weights = []
    for group in composition:
        if group == a.prgmw:
            weights.append(prgK)
        else:
            weights.append(srgK)

    # Reacts away b.mol until gone.  Still need to add different rate constants(weights)
    if rt.name != PolyCondensation:
        while b.mol >= 0:
            MC = random.choices(list(enumerate(composition)), weights=weights, k=1)[0]
            if MC[1] == a.prgmw or MC[1] == a.srgmw:
                composition[MC[0]] = round(MC[1] + b.mw - rt.wl, 4)
                b.mol -= 1
                weights[MC[0]] = cgK
            else:
                composition[MC[0]] = round(MC[1] + b.mw - rt.wl, 4)
                b.mol -= 1
            print(round(100 - (b.mol / bms * 100), 2))
    elif rt.name == PolyCondensation:
        while b.mol >= 0:
            MC = random.choices(list(enumerate(composition)), weights=weights, k=1)[0]
            if MC[1] == a.prgmw or MC[1] == a.srgmw:
                composition[MC[0]] = round(MC[1] + b.mw - rt.wl, 4)
                b.mol -= 1
                weights[MC[0]] = cgK
            elif MC[1] + a.mw < MC[1] + b.mw:
                composition[MC[0]] = round(MC[1] + a.mw - rt.wl, 4)
                try:
                    composition = [composition[x:x + len(a.comp)] for x in range(0, len(composition), len(a.comp))]
                    composition_tuple = [tuple(l) for l in composition]
                except TypeError:
                    composition = [composition[x:x + 1] for x in range(0, len(composition), 1)]
                    composition_tuple = [tuple(l) for l in composition]
                index = composition_tuple.index(a.comp)
                del composition_tuple[index]
                composition = list(itertools.chain(*composition_tuple))
                try:
                    weights = [weights[x:x + len(a.comp)] for x in range(0, len(weights), len(a.comp))]
                    weights_tuple = [tuple(l) for l in weights]
                except TypeError:
                    weights = [weights[x:x + 1] for x in range(0, len(weights), 1)]
                    weights_tuple = [tuple(l) for l in weights]
                del weights_tuple[index]
                weights = list(itertools.chain(*weights_tuple))
            elif MC[1] + a.mw > MC[1] + b.mw:
                composition[MC[0]] = round(MC[1] + b.mw - rt.wl, 4)
                b.mol -= 1
                weights[MC[0]] = cgK
            else:
                pass
            print(composition[MC[0]])
            # print(round(100-(b.mol/bms*100), 2))

    # Separates composition into compounds
    try:
        composition = [composition[x:x + len(a.comp)] for x in range(0, len(composition), len(a.comp))]
        composition_tuple = [tuple(l) for l in composition]
    except TypeError:
        composition = [composition[x:x + 1] for x in range(0, len(composition), 1)]
        composition_tuple = [tuple(l) for l in composition]

    # Tabulates final composition and converts to dataframe
    rxn_summary = collections.Counter(composition_tuple)
    RS = []
    for key in rxn_summary:
        MS = round(sum(key), 1)
        for item in final_product_masses:
            if MS == final_product_masses[item]:
                RS.append((item, rxn_summary[key], key))
    # Convert RS to dataframe
    rxn_summary_df = pandas.DataFrame(RS, columns=['Product', 'Count', 'Mass_Comp'])
    rxn_summary_df.set_index('Product', inplace=True)
    rxn_summary_df.loc[f"{b.sn}"] = [unreacted, b.mw]

    # Add columns to dataframe
    rxn_summary_df['Molar Mass'] = rxn_summary_df.index.map(final_product_masses.get)
    rxn_summary_df.sort_values(by=['Molar Mass'], ascending=True, inplace=True)
    rxn_summary_df['Mass'] = rxn_summary_df['Molar Mass'] * rxn_summary_df['Count']
    rxn_summary_df['(%)'] = round(rxn_summary_df['Mass'] / rxn_summary_df['Mass'].sum() * 100, 4)

    # Add EHC to dataframe if rt == Etherification
    if rt.name == Etherification:
        EHC = []
        for i in rxn_summary_df["Mass_Comp"]:
            try:
                for chain_weight in i:
                    EHCCount = 0
                    EHCCount += sum(chain_weight > max(a.comp) for chain_weight in i)
                EHC.append(((EHCCount * 35.453) / sum(i)) * 100)
            except TypeError:
                try:
                    EHC.append(35.453 / i * 100)
                except TypeError:
                    if sum(i) == a.mw:
                        EHC.append(0)
                    else:
                        EHC.append(35.453 / sum(i) * 100)
    rxn_summary_df['EHC'] = EHC
    rxn_summary_df['% EHC'] = (rxn_summary_df['EHC'] * rxn_summary_df['(%)']) / 100
    EHCp = round(rxn_summary_df['% EHC'].sum(), 4)

    print(rxn_summary_df)
    print(f'% EHC = {round(EHCp, 2)}')
    print(f'Theoretical WPE = {round((3545.3 / EHCp) - 36.4, 2)}')

    # Export rxn_summary_df to desktop as csv
    desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    rxn_summary_df.to_csv(desktop + '\\' + 'rxn_summary.csv')


simulate(a=Butanol, b=Epichlorohydrin, rt=Etherification, Samples=1000, EOR=1, a_mass=250, b_mass=250)

# ---------------------------------------------------User-Interface----------------------------------------------
# window = tkinter.Tk()
# window.title("Monte Karlo")
# window.geometry("{0}x{1}+0+0".format(window.winfo_screenwidth(), window.winfo_screenheight()))
# a.mass = tkinter.Entry(window)
# a.mass.grid(row=1, column=1)
# b.mass = tkinter.Entry(window)
# b.mass.grid(row=2, column=1)
# a = tkinter.Entry(window)
# a.grid(row=3, column=1)
# b = tkinter.Entry(window)
# b.grid(row=4, column=1)
#
# #Add button to tkinter
# button = tkinter.Button(window, text="Simulate", command=lambda: simulate(a.mass.get(), b.mass.get(), a.get(), b.get()))
# button.grid(row=5, column=0, padx=10, pady=10)
#
# # Add a label for the interactions entry
# tkinter.Label(window, text="Grams of A: ").grid(row=1, column=0)
# tkinter.Label(window, text="Grams of B: ").grid(row=2, column=0)
# tkinter.Label(window, text="Reactant A: ").grid(row=3, column=0)
# tkinter.Label(window, text="Reactant B: ").grid(row=4, column=0)
#
#
# window.mainloop()
