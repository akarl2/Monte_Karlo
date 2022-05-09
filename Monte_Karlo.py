import collections
import random
import sys
import tkinter
import pandas
from Database import *
from Reactions import *
import itertools
from pandastable import Table, TableModel, config
import statsmodels


# Set pandas dataframe display
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)
pandas.set_option('display.width', 100)

# Converts string name to class name
def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

def simulate(a, b, rt, Samples, EOR, a_mass, b_mass, PRGk, SRGk, CGRk):
    a = str_to_class(a)()
    b = str_to_class(b)()
    rt = str_to_class(rt)()
    EOR = float(EOR)
    a.mass = float(a_mass)
    b.mass = float(b_mass)
    a.mol = round(a.mass / a.mw, 3) * int(Samples)
    b.mol = round(b.mass / b.mw, 3) * int(Samples)
    unreacted = b.mol - (EOR * b.mol)
    b.mol = EOR * b.mol
    bms = b.mol

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

    #Creates a list with possible chain lengths
    cw = a.prgmw
    chain_lengths = [(0, a.prgmw)]
    for chain_length in range(2, 100, 2):
        cw = cw + b.mw-rt.wl
        chain_lengths.append((chain_length - 1, round(cw, 2)))
        cw = cw + a.mw-rt.wl
        chain_lengths.append((chain_length, round(cw, 2)))

    # Specify rate constants
    prgK = float(PRGk)
    srgK = float(SRGk)
    cgK = float(CGRk)

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
            sim_status(round(100 - (b.mol / bms * 100), 2))
    elif rt.name == PolyCondensation:
        while b.mol >= 0:
            MC = random.choices(list(enumerate(composition)), weights=weights, k=1)[0]
            index = next((i for i, v in enumerate(chain_lengths) if round(v[1], 1) == round(MC[1], 1)), None)
            if chain_lengths[index+1][1] - composition[MC[0]] == (round(a.mw-rt.wl,2)):
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
                weights[MC[0]] = cgK
            elif chain_lengths[index + 1][1] - composition[MC[0]] == (round(b.mw - rt.wl,2)):
                b.mol -= 1
                weights[MC[0]] = cgK
            print(b.mol)
            composition[MC[0]] = chain_lengths[index + 1][1]
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
        MS = round(sum(key))
        for item in final_product_masses:
            if MS == round(final_product_masses[item]):
                RS.append((item, rxn_summary[key], key))
    # Convert RS to dataframe
    rxn_summary_df = pandas.DataFrame(RS, columns=['Product', 'Count', 'Mass Distribution'])
    rxn_summary_df.set_index('Product', inplace=True)
    rxn_summary_df.loc[f"{b.sn}"] = [unreacted, b.mw]

    # Add columns to dataframe
    rxn_summary_df['Molar Mass'] = rxn_summary_df.index.map(final_product_masses.get)
    rxn_summary_df.sort_values(by=['Molar Mass'], ascending=True, inplace=True)
    rxn_summary_df['Mass'] = rxn_summary_df['Molar Mass'] * rxn_summary_df['Count']
    rxn_summary_df['Mol %'] = round(rxn_summary_df['Count'] / rxn_summary_df['Count'].sum() * 100, 4)
    rxn_summary_df['Wt %'] = round(rxn_summary_df['Mass'] / rxn_summary_df['Mass'].sum() * 100, 4)

    # Add ehc to dataframe if rt == Etherification
    if rt.name == Etherification:
        ehc = []
        for i in rxn_summary_df["Mass Distribution"]:
            try:
                EHCCount = 0
                EHCCount += sum(chain_weight > max(a.comp) for chain_weight in i)
                ehc.append(((EHCCount * 35.453) / sum(i)) * 100)
            except TypeError:
                try:
                    ehc.append(35.453 / i * 100)
                except TypeError:
                    if sum(i) == a.mw:
                        ehc.append(0)
                    else:
                        ehc.append(35.453 / sum(i) * 100)
        rxn_summary_df['ehc'] = ehc
        rxn_summary_df['% ehc'] = (rxn_summary_df['ehc'] * rxn_summary_df['Wt %']) / 100
        EHCp = round(rxn_summary_df['% ehc'].sum(), 4)
        update_percent_EHC(round(EHCp, 2))
        update_WPE(round((3545.3 / EHCp) - 36.4, 2))
    update_results(rxn_summary_df)

# ---------------------------------------------------User-Interface----------------------------------------------#
window = tkinter.Tk()
window.title("Monte Karlo")
window.geometry("{0}x{1}+0+0".format(window.winfo_screenwidth(), window.winfo_screenheight()))

window.configure(background="#00BFFF")
Mass_of_A = tkinter.Entry(window)
Mass_of_A.insert(0, "100")
Mass_of_A.grid(row=2, column=1)
Moles_of_A = tkinter.Entry(window)
Moles_of_A.grid(row=1, column=3)
Mass_of_B = tkinter.Entry(window)
Mass_of_B.insert(0, "100")
Mass_of_B.grid(row=4, column=1)
Moles_of_B = tkinter.Entry(window)
Moles_of_B.grid(row=2, column=3)
speciesA = tkinter.StringVar()
speciesA.set("Reactant A")
Reactant_A = tkinter.OptionMenu(window, speciesA, *reactantsA)
Reactant_A.grid(row=1, column=1)
speciesB = tkinter.StringVar()
speciesB.set("Reactant B")
Reactant_B = tkinter.OptionMenu(window, speciesB, *reactantsB)
Reactant_B.grid(row=3, column=1)
reaction_type = tkinter.StringVar()
reaction_type.set("Reaction Type")
Reaction_Type = tkinter.OptionMenu(window, reaction_type, *Reactions)
Reaction_Type.grid(row=5, column=1)
Samples = tkinter.Entry(window)
Samples.insert(0, "1000")
Samples.grid(row=6, column=1)
EOR = tkinter.Entry(window)
EOR.insert(0, 1)
EOR.grid(row=7, column=1)
Sim_status = tkinter.Entry(window)
Sim_status.grid(row=1, column=5)
Percent_EHC = tkinter.Entry(window)
Percent_EHC.grid(row=15, column=1)
Theoretical_WPE = tkinter.Entry(window)
Theoretical_WPE.grid(row=16, column=1)
PRGk = tkinter.Entry(window)
PRGk.insert(0, 1)
PRGk.grid(row=8, column=1)
SRGk = tkinter.Entry(window)
SRGk.insert(0, 1)
SRGk.grid(row=9, column=1)
CGRk = tkinter.Entry(window)
CGRk.insert(0, 1)
CGRk.grid(row=10, column=1)
results = tkinter.Text(window, height=20, width=50)
def show_results(rxn_summary_df):
    global results
    frame = tkinter.Frame(window)
    x = (window.winfo_screenwidth() - frame.winfo_reqwidth()) / 2
    y = (window.winfo_screenheight() - frame.winfo_reqheight()) / 2
    frame.place(x=x, y=y, anchor='center')
    results = Table(frame, dataframe=rxn_summary_df, showtoolbar=True, showstatusbar=True, showindex=True, width=x,
                    height=y, align='center')
    results.show()

#Replace results Table with new results
def update_results(rxn_summary_df):
    global results
    try:
        results.destroy()
        show_results(rxn_summary_df)
    except:
        show_results(rxn_summary_df)
        pass

def update_moles_A(self):
    a = str_to_class(speciesA.get())()
    molesA = float(Mass_of_A.get()) / float(a.mw)
    Moles_of_A.delete(0, 'end')
    Moles_of_A.insert(0, round(molesA,4))

def update_moles_B(self):
    b = str_to_class(speciesB.get())()
    molesB = float(Mass_of_B.get()) / float(b.mw)
    Moles_of_B.delete(0, 'end')
    Moles_of_B.insert(0, round(molesB,4))

#Run update_moles_A() when the user changes the value of Mass_of_A
Mass_of_A.bind("<KeyRelease>", update_moles_A)
Mass_of_B.bind("<KeyRelease>", update_moles_B)

def update_percent_EHC(Value):
    Percent_EHC.delete(0, tkinter.END)
    Percent_EHC.insert(0, Value)

def sim_status(Value):
    Sim_status.delete(0, tkinter.END)
    Sim_status.insert(0, Value)

def update_WPE(Value):
    Theoretical_WPE.delete(0, tkinter.END)
    Theoretical_WPE.insert(0, Value)

def sim_values():
    simulate(a=speciesA.get(), b=speciesB.get(), rt=reaction_type.get(), Samples=Samples.get(), EOR=EOR.get(),
             a_mass=Mass_of_A.get(), b_mass=Mass_of_B.get(), PRGk=PRGk.get(), SRGk=SRGk.get(), CGRk=CGRk.get())

# add button to simulate
button = tkinter.Button(window, text="Simulate", command=sim_values,width=15,bg="red")
button.grid(row=11, column=1)


# Add a label for the interactions entry
bg_color = '#00BFFF'
tkinter.Label(window, text="Grams of A: ", bg=bg_color).grid(row=2, column=0)
tkinter.Label(window, text="Moles of A: ", bg=bg_color).grid(row=1, column=2, padx=10)
tkinter.Label(window, text="Grams of B: ", bg=bg_color).grid(row=4, column=0)
tkinter.Label(window, text="Moles of B: ", bg=bg_color).grid(row=2, column=2, padx=10)
tkinter.Label(window, text="Reactant A: ", bg=bg_color).grid(row=1, column=0)
tkinter.Label(window, text="Reactant B: ", bg=bg_color).grid(row=3, column=0)
tkinter.Label(window, text="Reaction Type: ", bg=bg_color).grid(row=5, column=0)
tkinter.Label(window, text="# of Samples: ", bg=bg_color).grid(row=6, column=0)
tkinter.Label(window, text="Extent of Reaction (EOR): ", bg=bg_color).grid(row=7, column=0)
tkinter.Label(window, text="Simulation Status: ", bg=bg_color).grid(row=1, column=4)
tkinter.Label(window, text="% EHC: ", bg=bg_color).grid(row=15, column=0)
tkinter.Label(window, text="Theoretical WPE: ", bg=bg_color).grid(row=16, column=0)
tkinter.Label(window, text="Primary K: ", bg=bg_color).grid(row=8, column=0)
tkinter.Label(window, text="Secondary k: ", bg=bg_color).grid(row=9, column=0)
tkinter.Label(window, text="Child k: ", bg=bg_color).grid(row=10, column=0)




window.mainloop()
