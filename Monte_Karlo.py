import collections
import random
import sys
import tkinter
from tkinter import ttk, messagebox
import pandas
from Database import *
from Reactions import *
import itertools
from pandastable import Table, TableModel, config
import statsmodels
import math

# Set pandas dataframe display
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)
pandas.set_option('display.width', 100)

running = False
# Runs the simulation
def simulate(a, b, rt, samples, eor, a_mass, b_mass, prgk, srgk, crgk, emr, emo):
    global running, chain_lengths_id, emo_a, composition_tuple
    progress['value'] = 0
    clear_values()
    a = str_to_class(a)()
    b = str_to_class(b)()
    rt = str_to_class(rt)()
    eor = float(eor)
    emr = float(emr)
    a.mass = float(a_mass)
    b.mass = float(b_mass)
    prgK = float(prgk)
    srgK = float(srgk)
    cgK = float(crgk)
    a.mol = round(a.mass / a.mw, 3) * int(samples)
    b.mol = (round(b.mass / b.mw, 3) * int(samples)) * eor
    running = True
    unreacted = b.mol - (eor * b.mol)
    bms = b.mol

    # Creates final product molar masses from final product name(s)
    final_product_masses = ({a.sn: round(a.mw, 1), b.sn: round(b.mw, 1)})
    if rt.name != PolyCondensation:
        final_product_masses.update(
            {f"{a.sn}({1})_{b.sn}({str(i)})": round(a.mw + (i * b.mw - i * rt.wl), 1) for i in range(1, 500)})
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

    # Creates a list with the ID's of the chain lengths
    if rt.name == PolyCondensation:
        cw = a.prgmw
        cw2 = b.prgmw
        chain_lengths_id = [((0, a.prgmw), a.rg), ((1, b.prgmw), b.rg)]
        for chain_length in range(2, 100, 2):
            cw = cw + b.mw - rt.wl
            chain_lengths_id.append(((chain_length - 1, round(cw, 3)), b.rg))
            cw = cw + a.mw - rt.wl
            chain_lengths_id.append(((chain_length, round(cw, 3)), a.rg))
            cw2 = cw2 + a.mw - rt.wl
            chain_lengths_id.append(((chain_length, round(cw2, 3)), a.rg))
            cw2 = cw2 + b.mw - rt.wl
            chain_lengths_id.append(((chain_length, round(cw2, 3)), b.rg))

    # Creates starting composition list
    composition = []
    if rt.name != PolyCondensation:
        try:
            for i in range(0, int(a.mol)):
                composition.extend(group for group in a.comp)
        except TypeError:
            for i in range(0, int(a.mol)):
                composition.append(a.mw)
    if rt.name == PolyCondensation:
        try:
            for i in range(0, int(a.mol)):
                composition.extend(group for group in a.comp)
        except TypeError:
            for i in range(0, int(a.mol)):
                composition.append(a.mw)
        try:
            for i in range(0, int(b.mol)):
                composition.extend(group for group in b.comp)
        except TypeError:
            for i in range(0, int(b.mol)):
                composition.append(b.mw)

    # Reacts away b.mol until gone.
    if rt.name != PolyCondensation:
        weights = []
        for group in composition:
            if group == a.prgmw:
                weights.append(int(prgK))
            elif group == a.srgmw:
                weights.append(int(srgK))
        while b.mol >= 0 and running == True:
            MC = random.choices(list(enumerate(composition)), weights=weights, k=1)[0]
            if MC[1] == a.prgmw or MC[1] == a.srgmw:
                composition[MC[0]] = round(MC[1] + b.mw - rt.wl, 4)
                weights[MC[0]] = cgK
            else:
                composition[MC[0]] = round(MC[1] + b.mw - rt.wl, 4)
            b.mol -= 1
            progress['value'] = round(100 - (b.mol / bms * 100), 1)
            window.update()
        try:
            composition = [composition[x:x + len(a.comp)] for x in range(0, len(composition), len(a.comp))]
            composition_tuple = [tuple(l) for l in composition]
        except TypeError:
            composition = [composition[x:x + 1] for x in range(0, len(composition), 1)]
            composition_tuple = [tuple(l) for l in composition]
    elif rt.name == PolyCondensation:
        # determines starting status of the reaction
        IDLIST = []
        amine_ct = 0
        acid_ct = 0
        alcohol_ct = 0
        for chain in composition:
            for chain_ID in range(0, len(chain_lengths_id)):
                if math.isclose(chain, chain_lengths_id[chain_ID][0][1], abs_tol=1):
                    ID = chain_lengths_id[chain_ID][1]
                    IDLIST.append(ID)
                    if ID == "Amine":
                        amine_ct += 1
                    elif ID == "Acid":
                        acid_ct += 1
                    elif ID == "Alcohol":
                        alcohol_ct += 1
                    break
        if emo == "Amine_Value":
            emo_a = round((amine_ct * 56100) / (sum(composition)), 2)
        elif emo == "Acid_Value":
            emo_a = round((acid_ct * 56100) / (sum(composition)), 2)
        elif emo == "OH_Value":
            emo_a = round((alcohol_ct * 56100) / (sum(composition)), 2)
        try:
            composition = [composition[x:x + len(a.comp)] for x in range(0, len(composition), len(a.comp))]
            composition_tuple = [tuple(l) for l in composition]
            IDLIST = [IDLIST[x:x + len(a.comp)] for x in range(0, len(IDLIST), len(a.comp))]
            IDLIST_tuple = [tuple(l) for l in IDLIST]
        except TypeError:
            composition = [composition[x:x + 1] for x in range(0, len(composition), 1)]
            composition_tuple = [tuple(l) for l in composition]
            IDLIST = [IDLIST[x:x + 1] for x in range(0, len(IDLIST), 1)]
            IDLIST_tuple = [tuple(l) for l in IDLIST]
        composition_tuple = [list(l) for l in composition_tuple]
        IDLIST_tuple = [list(l) for l in IDLIST_tuple]

        # runs the reaction
        while emo_a > emr and running == True:
            RC = random.choice(list(enumerate(IDLIST_tuple)))
            RCR_temp = random.choice(list(enumerate(RC[1])))
            RCR = RCR_temp[1]
            RCR_index = RCR_temp[0]
            RC2 = random.choice(list(enumerate(IDLIST_tuple)))
            RCR2_temp = random.choice(list(enumerate(RC2[1])))
            RCR2 = RCR2_temp[1]
            RCR2_index = RCR2_temp[0]
            while RCR == RCR2 and RC[0] == RC2[0]:
                RC = random.choice(list(enumerate(IDLIST_tuple)))
                RCR_temp = random.choice(list(enumerate(RC[1])))
                RCR = RCR_temp[1]
                RCR_index = RCR_temp[0]
                RC2 = random.choice(list(enumerate(IDLIST_tuple)))
                RCR2_temp = random.choice(list(enumerate(RC2[1])))
                RCR2 = RCR2_temp[1]
                RCR2_index = RCR2_temp[0]
            # randomly select another value from RCR2_index other than RCR2_value
            RCR2_other = random.choice(list(enumerate(RC2[1])))
            while RCR2_other[0] == RCR2_index:
                RCR2_other = random.choice(list(enumerate(RC2[1])))
            RCR2_other_index = RCR2_other[0]
            if RCR != RCR2 and RC[0] != RC2[0]:
                composition_tuple[RC[0]][RCR_index] += (sum(composition_tuple[RC2[0]]) - rt.wl)
                IDLIST_tuple[RC[0]][RCR_index] = IDLIST_tuple[RC2[0]][RCR2_other_index]
                del composition_tuple[RC2[0]]
                del IDLIST_tuple[RC2[0]]
            else:
                pass


            # determines current status of reaction
            composition_tuple_temp = list(itertools.chain(*composition_tuple))
            IDLIST = [None] * len(composition_tuple_temp)
            amine_ct = 0
            acid_ct = 0
            alcohol_ct = 0
            for chain in composition_tuple_temp:
                for chain_ID in range(0, len(chain_lengths_id)):
                    if math.isclose(chain, chain_lengths_id[chain_ID][0][1], abs_tol=1):
                        ID = chain_lengths_id[chain_ID][1]
                        IDLIST.append(ID)
                        if ID == "Amine":
                            amine_ct += 1
                        elif ID == "Acid":
                            acid_ct += 1
                        elif ID == "Alcohol":
                            alcohol_ct += 1
                        break
            if emo == "Amine_Value":
                emo_a = round((amine_ct * 56100) / (sum(composition_tuple_temp)), 2)
            elif emo == "Acid_Value":
                emo_a = round((acid_ct * 56100) / (sum(composition_tuple_temp)), 2)
            elif emo == "OH_Value":
                emo_a = round((alcohol_ct * 56100) / (sum(composition_tuple_temp)), 2)
            progress['value'] = round((emr/emo_a)*100, 1)
            window.update()
        composition_tuple = [tuple(l) for l in composition_tuple]

    # Tabulates final composition and converts to dataframe
    rxn_summary = collections.Counter(composition_tuple)
    RS = []
    for key in rxn_summary:
        MS = sum(key)
        for item in final_product_masses:
            if math.isclose(MS, final_product_masses[item], abs_tol=1):
                RS.append((item, rxn_summary[key], key))

    # Convert RS to dataframe
    rxn_summary_df = pandas.DataFrame(RS, columns=['Product', 'Count', 'Mass Distribution'])
    rxn_summary_df.set_index('Product', inplace=True)
    rxn_summary_df.loc[f"{b.sn}"] = [unreacted, b.mw]

    # print each value in each row from Mass Distribution
    if rt.name == PolyCondensation:
        for i in range(len(rxn_summary_df)):
            amine_ct = 0
            acid_ct = 0
            alcohol_ct = 0
            try:
                for j in range(len(rxn_summary_df.iloc[i]['Mass Distribution'])):
                    for chain_length in range(0, len(chain_lengths_id)):
                        if math.isclose(rxn_summary_df.iloc[i]['Mass Distribution'][j],
                                        chain_lengths_id[chain_length][0][1], abs_tol=1):
                            chain_ID = chain_lengths_id[chain_length][1]
                            if chain_ID == "Amine":
                                amine_ct += 1
                            if chain_ID == "Acid":
                                acid_ct += 1
                            if chain_ID == "Alcohol":
                                alcohol_ct += 1
                            break
                amine_value = round((amine_ct * 56100) / sum((rxn_summary_df.iloc[i]['Mass Distribution'])), 2)
                acid_value = round((acid_ct * 56100) / sum((rxn_summary_df.iloc[i]['Mass Distribution'])), 2)
                alcohol_value = round((alcohol_ct * 56100) / sum((rxn_summary_df.iloc[i]['Mass Distribution'])), 2)
                rxn_summary_df.loc[f"{rxn_summary_df.index[i]}", "Amine Value"] = amine_value
                rxn_summary_df.loc[f"{rxn_summary_df.index[i]}", "Acid Value"] = acid_value
                rxn_summary_df.loc[f"{rxn_summary_df.index[i]}", "OH Value"] = alcohol_value
            except TypeError:
                chain_ID = "Amine"
                amine_ct += 1

    global expanded_results
    expanded_results = rxn_summary_df

    # Add columns to dataframe
    rxn_summary_df['Molar Mass'] = rxn_summary_df.index.map(final_product_masses.get)
    rxn_summary_df.sort_values(by=['Molar Mass'], ascending=True, inplace=True)
    rxn_summary_df['Mass'] = rxn_summary_df['Molar Mass'] * rxn_summary_df['Count']
    rxn_summary_df['Mol %'] = round(rxn_summary_df['Count'] / rxn_summary_df['Count'].sum() * 100, 4)
    rxn_summary_df['Wt %'] = round(rxn_summary_df['Mass'] / rxn_summary_df['Mass'].sum() * 100, 4)
    try:
        rxn_summary_df['OH Value'] = round(rxn_summary_df['OH Value'] * rxn_summary_df['Wt %'] / 100, 2)
        rxn_summary_df['Amine Value'] = round(rxn_summary_df['Amine Value'] * rxn_summary_df['Wt %'] / 100, 2)
        rxn_summary_df['Acid Value'] = round(rxn_summary_df['Acid Value'] * rxn_summary_df['Wt %'] / 100, 2)
    except KeyError:
        pass

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

    # sum rxn_summary_df by product but keep Molar mass the same
    rxn_summary_df = rxn_summary_df.groupby(['Product', 'Molar Mass']).sum()
    rxn_summary_df.sort_values(by=['Molar Mass'], ascending=True, inplace=True)
    rxn_summary_df_compact = rxn_summary_df.groupby(['Product', 'Molar Mass']).sum()
    rxn_summary_df_compact.sort_values(by=['Molar Mass'], ascending=True, inplace=True)

    if rt.name == PolyCondensation:
        update_Acid_Value(round(rxn_summary_df['Acid Value'].sum(), 2))
        update_Amine_Value(round(rxn_summary_df['Amine Value'].sum(), 2))
        update_OH_Value(round(rxn_summary_df['OH Value'].sum(), 2))

    show_results(rxn_summary_df_compact)

#-------------------------------------------Aux Functions---------------------------------#

def show_results(rxn_summary_df_compact):
    global results
    global frame
    try:
        results.destroy()
        frame.destroy()
    except NameError:
        pass
    frame = tkinter.Frame(window)
    x = (window.winfo_screenwidth() - frame.winfo_reqwidth()) / 2
    y = (window.winfo_screenheight() - frame.winfo_reqheight()) / 2
    x = x + 100
    frame.place(x=x, y=y, anchor='center')
    results = Table(frame, dataframe=rxn_summary_df_compact, showtoolbar=True, showstatusbar=True, showindex=True,
                    width=x,
                    height=y, align='center')
    results.show()

def show_results_expanded():
    global results
    global frame
    try:
        results.destroy()
        frame.destroy()
    except NameError:
        pass
    frame = tkinter.Frame(window)
    x = (window.winfo_screenwidth() - frame.winfo_reqwidth()) / 2
    y = (window.winfo_screenheight() - frame.winfo_reqheight()) / 2
    x = x + 100
    frame.place(x=x, y=y, anchor='center')
    results = Table(frame, dataframe=expanded_results, showtoolbar=True, showstatusbar=True, showindex=True, width=x,
                    height=y, align='center')
    results.show()

def update_moles_A(self):
    a = str_to_class(speciesA.get())()
    molesA = float(Mass_of_A.get()) / float(a.mw)
    Moles_of_A.delete(0, 'end')
    Moles_of_A.insert(0, round(molesA, 4))

def update_moles_B(self):
    b = str_to_class(speciesB.get())()
    molesB = float(Mass_of_B.get()) / float(b.mw)
    Moles_of_B.delete(0, 'end')
    Moles_of_B.insert(0, round(molesB, 4))

def update_percent_EHC(Value):
    Percent_EHC.delete(0, tkinter.END)
    Percent_EHC.insert(0, Value)

def update_WPE(Value):
    Theoretical_WPE.delete(0, tkinter.END)
    Theoretical_WPE.insert(0, Value)

def update_Acid_Value(Value):
    Acid_Value.delete(0, tkinter.END)
    Acid_Value.insert(0, Value)

def update_Amine_Value(Value):
    Amine_Value.delete(0, tkinter.END)
    Amine_Value.insert(0, Value)

def update_OH_Value(Value):
    OH_Value.delete(0, tkinter.END)
    OH_Value.insert(0, Value)

def clear_values():
    Percent_EHC.delete(0, tkinter.END)
    Theoretical_WPE.delete(0, tkinter.END)
    Acid_Value.delete(0, tkinter.END)
    Amine_Value.delete(0, tkinter.END)
    OH_Value.delete(0, tkinter.END)

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

def stop():
    global running
    if running:
        running = False
    else:
        pass

def sim_values():
    try:
        simulate(a=speciesA.get(), b=speciesB.get(), rt=reaction_type.get(), samples=Samples.get(), eor=EOR.get(),
                 a_mass=Mass_of_A.get(), b_mass=Mass_of_B.get(), prgk=PRGk.get(), srgk=SRGk.get(), crgk=CGRk.get(),
                 emr=End_Metric_Entry.get(), emo=End_Metric_Selection.get())
    except AttributeError:
        messagebox.showerror("Field Error", "Please fill out all fields!")
        pass

# ---------------------------------------------------User-Interface----------------------------------------------#
window = tkinter.Tk()
window.iconbitmap("testtube.ico")
window.title("Monte Karlo")
window.geometry("{0}x{1}+0+0".format(window.winfo_screenwidth(), window.winfo_screenheight()))
window.configure(background="#00BFFF")
Mass_of_A = tkinter.Entry(window, width=20)
Mass_of_A.insert(0, "100")
Mass_of_A.grid(row=2, column=1)
Moles_of_A = tkinter.Entry(window)
Moles_of_A.grid(row=1, column=3)
Mass_of_B = tkinter.Entry(window, width=20)
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
Acid_Value = tkinter.Entry(window)
Acid_Value.grid(row=17, column=1)
Amine_Value = tkinter.Entry(window)
Amine_Value.grid(row=18, column=1)
OH_Value = tkinter.Entry(window)
OH_Value.grid(row=19, column=1)
End_Metric_Selection = tkinter.StringVar()
End_Metric_Selection.set("Amine_Value")
End_Metric_Options = tkinter.OptionMenu(window, End_Metric_Selection, *End_Metrics)
End_Metric_Options.grid(row=21, column=1)
End_Metric_Entry = tkinter.Entry(window)
End_Metric_Entry.insert(0, "250")
End_Metric_Entry.grid(row=22, column=1)
PRGk = tkinter.Entry(window)
PRGk.insert(0, 1)
PRGk.grid(row=8, column=1)
SRGk = tkinter.Entry(window)
SRGk.insert(0, 0)
SRGk.grid(row=9, column=1)
CGRk = tkinter.Entry(window)
CGRk.insert(0, 0)
CGRk.grid(row=10, column=1)
results = tkinter.Text(window, height=20, width=50)

# add button to simulate
button = tkinter.Button(window, text="Simulate", command=sim_values, width=15, bg="Green")
button.grid(row=11, column=1)

# add a button to stop the simulation
stop_button = tkinter.Button(window, text="Stop", command=stop, width=15, bg="Red")
stop_button.grid(row=12, column=1)

# add button to expand dataframe
expand = tkinter.Button(window, text="Expand Data", command=show_results_expanded, width=15, bg="green")
expand.grid(row=23, column=1)

# Update moles when user changes the value of Mass_of_A or Mass_of_B
Mass_of_A.bind("<KeyRelease>", update_moles_A)
Mass_of_B.bind("<KeyRelease>", update_moles_B)

# add a determinate progress bar to window using sim_status
progress = ttk.Progressbar(window, orient="horizontal", length=300, mode="determinate")
progress.grid(row=1, column=5)

# ---------------------------------------------Labels for UI---------------------------------#
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
tkinter.Label(window, text="Acid Value: ", bg=bg_color).grid(row=17, column=0)
tkinter.Label(window, text="Amine Value: ", bg=bg_color).grid(row=18, column=0)
tkinter.Label(window, text="OH Value: ", bg=bg_color).grid(row=19, column=0)
tkinter.Label(window, text="End Metric: ", bg=bg_color).grid(row=21, column=0)
tkinter.Label(window, text="Primary K: ", bg=bg_color).grid(row=8, column=0)
tkinter.Label(window, text="Secondary k: ", bg=bg_color).grid(row=9, column=0)
tkinter.Label(window, text="Child k: ", bg=bg_color).grid(row=10, column=0)

window.mainloop()
