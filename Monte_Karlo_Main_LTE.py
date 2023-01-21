import collections
import random
import sys
import tkinter
from tkinter import ttk, messagebox
import pandas
from ttkwidgets.autocomplete import AutocompleteEntry, AutocompleteCombobox
from Database import *
from Reactions import reactive_groups, NH2,NH,COOH,COC,OH
from Reactions import *
import itertools
from pandastable import Table, TableModel, config
import statsmodels
import math
from Reactants import *
from Reactants import R1Data, R2Data, R3Data, R4Data, R5Data, R6Data, R7Data, R8Data, R9Data, R10Data, R11Data, R12Data, \
    R13Data, R14Data

# Set pandas dataframe display
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)
pandas.set_option('display.width', 100)

# Runs the simulation
global running, emo_a, results, frame, expanded_results

global groupA, groupB
def simulate(starting_materials):
    running = True
    sim.progress['value'] = 0

    composition = [[[[group[0], group[1] * compound[3][0]] for group in compound[0]], compound[2], compound[3], compound[1]] for
                   compound in starting_materials]


    def check_react(groups):
        global groupA, groupB
        groupA = groups[0][2]
        groupB = groups[1][2]
        if groupB in getattr(rg, groupA):
            new_group(groupA, groupB)
            return True
        else:
            return False

    def new_group(groupA, groupB):
        NG = getattr(eval(groupA + '()'), groupB)
        WL = getattr(eval(groupA + '()'), groupB + '_wl')
        return {'NG': NG, 'WL': WL}

    def update_comp(composition, groups):
        NC = composition[groups[0][0]]
        compoundA = composition[groups[0][0]]
        compoundB = composition[groups[1][0]]
        compoundAloc = groups[0][1]
        compoundBloc = groups[1][1]
        NW = compoundA[3][0] + compoundB[3][0] - new_group(groupA, groupB)['WL']
        NC = [[[group[0], group[1] / NC[2][0]] if group[1] != 0 else group for group in NC[0]], NC[1], [1], [NW]]
        NC[0][groups[0][1]][0] = new_group(groupA, groupB)['NG']
        NC[0][groups[0][1]][1] = (compoundA[0][compoundAloc][1] / compoundA[2][0]) * (compoundB[0][compoundBloc][1] / compoundB[2][0])
        NC[0][groups[0][1]][1] = 0
        NCA = [[[group[0], group[1] / compoundA[2][0]] if group[1] != 0 else group for group in compoundA[0]], compoundA[1], [compoundA[2][0] - 1] if compoundA[2][0] != 0 else [0], compoundA[3]]
        NCA = [[[group[0], group[1] * NCA[2][0]] if group[1] != 0 else group for group in NCA[0]], NCA[1], [NCA[2][0]] if NCA[2][0] != 0 else [0], NCA[3]]
        NCB = [[[group[0], group[1] / compoundB[2][0]] if group[1] != 0 else group for group in compoundB[0]], compoundB[1], [compoundB[2][0] - 1] if compoundB[2][0] != 0 else [0], compoundB[3]]
        NCB = [[[group[0], group[1] * NCB[2][0]] if group[1] != 0 else group for group in NCB[0]], NCB[1], [NCB[2][0]] if NCB[2][0] != 0 else [0], NCB[3]]
        composition[groups[0][0]] = NCA
        composition[groups[1][0]] = NCB
        composition.append(NC)
        print(composition)
        NC[45]



    while running:
        weights = []
        chemical = []
        for i, chemicals in enumerate(composition):
            index = 0
            for group in chemicals[0]:
                chemical.append([i, index, group[0]])
                weights.append(group[1])
                index += 1
        groups = random.choices(chemical, weights, k=2)
        while groups[0][0] == groups[1][0] or check_react(groups) is False:
            groups = random.choices(chemical, weights, k=2)
        update_comp(composition, groups)

        print("move forward")

    # global running, emo_a, results, expanded_results
    # eor = float(eor)
    # emr = float(emr)

    # 
    # 
    # 
    # # elif rt.name == PolyCondensation:
    # #     final_product_masses.update(
    # #         {f"{a.sn}({i})_{b.sn}({str(i)})": round(i * a.mw + i * b.mw - (i + i - 1) * rt.wl, 2) for i in
    # #          range(1, 1001)})
    # #     final_product_masses.update(
    # #         {f"{a.sn}({i - 1})_{b.sn}({str(i)})": round((i - 1) * a.mw + i * b.mw - (i + i - 2) * rt.wl, 1) for i in
    # #          range(2, 1001)})
    # #     final_product_masses.update(
    # #         {f"{a.sn}({i})_{b.sn}({str(i - 1)})": round(i * a.mw + (i - 1) * b.mw - (i + i - 2) * rt.wl, 1) for i in
    # #          range(2, 1001)})
    # 
    # # Creates a list with the ID's of the chain lengths
    # if rt.name == PolyCondensation:
    #     cw = a.prgmw
    #     cw2 = b.prgmw
    #     chain_lengths_id = [((0, a.prgmw), a.rg), ((1, b.prgmw), b.rg)]
    #     for chain_length in range(2, 100, 2):
    #         cw = cw + b.mw - rt.wl
    #         chain_lengths_id.append(((chain_length - 1, round(cw, 3)), b.rg))
    #         cw = cw + a.mw - rt.wl
    #         chain_lengths_id.append(((chain_length, round(cw, 3)), a.rg))
    #         cw2 = cw2 + a.mw - rt.wl
    #         chain_lengths_id.append(((chain_length, round(cw2, 3)), a.rg))
    #         cw2 = cw2 + b.mw - rt.wl
    #         chain_lengths_id.append(((chain_length, round(cw2, 3)), b.rg))
    # 
    # # Creates starting composition list
    # composition = []
    # try:
    #     for i in range(0, int(a.mol)):
    #         composition.extend(group for group in a.comp)
    # except TypeError:
    #     for i in range(0, int(a.mol)):
    #         composition.append(a.mw)
    # try:
    #     for i in range(0, int(b.mol)):
    #         composition.extend(group for group in b.comp)
    # except TypeError:
    #     for i in range(0, int(b.mol)):
    #         composition.append(b.mw)
    # 
    # # Reacts away b.mol until gone.
    # if rt.name != PolyCondensation:
    #     weights = []
    #     for B in composition:
    #         if B == a.prgmw:
    #             weights.append(int(prgK))
    #         elif B == a.srgmw:
    #             weights.append(int(srgK))
    #         elif B == b.prgmw:
    #             weights.append(1)
    #         else:
    #             weights.append(int(cgK))
    #     indicesA = [i for i, x in enumerate(composition) if x == a.prgmw or x == a.srgmw or x != b.prgmw]
    #     indicesB = [i for i, x in enumerate(composition) if x == b.prgmw]
    #     while len(indicesA) > 1 and len(indicesB) > 1 and running is True:
    #         indicesA = [i for i, x in enumerate(composition) if x == a.prgmw or x == a.srgmw or x != b.prgmw]
    #         weightsA = [weights[i] for i in indicesA]
    #         indicesB = [i for i, x in enumerate(composition) if x == b.prgmw]
    #         weightsB = [weights[i] for i in indicesB]
    #         MCA = random.choices(list(enumerate(indicesA)), weights=weightsA, k=1)[0]
    #         MCB = random.choices(list(enumerate(indicesB)), weights=weightsB, k=1)[0]
    #         composition[MCA[1]] = round(composition[MCA[1]] + composition[MCB[1]] - rt.wl, 3)
    #         composition.pop(MCB[1])
    #         weights[MCA[0]] = cgK
    #         b.mol -= 1
    #         sim.progress['value'] = round(100 - (b.mol / bms * 100), 1)
    #         window.update()
    #     try:
    #         composition = [composition[x:x + len(a.comp)] for x in range(0, len(composition), len(a.comp))]
    #         composition_tuple = [tuple(item) for item in composition]
    #     except TypeError:
    #         composition = [composition[x:x + 1] for x in range(0, len(composition), 1)]
    #         composition_tuple = [tuple(item) for item in composition]
    # 
    # elif rt.name == PolyCondensation:
    #     # determines starting status of the reaction
    #     IDLIST = []
    #     amine_ct = 0
    #     acid_ct = 0
    #     alcohol_ct = 0
    #     for chain in composition:
    #         for chain_ID in range(0, len(chain_lengths_id)):
    #             if math.isclose(chain, chain_lengths_id[chain_ID][0][1], abs_tol=1):
    #                 ID = chain_lengths_id[chain_ID][1]
    #                 IDLIST.append(ID)
    #                 if ID == "Amine":
    #                     amine_ct += 1
    #                 elif ID == "Acid":
    #                     acid_ct += 1
    #                 elif ID == "Alcohol":
    #                     alcohol_ct += 1
    #                 break
    #     if emo == "Amine_Value":
    #         emo_a = round((amine_ct * 56100) / (sum(composition)), 2)
    #     elif emo == "Acid_Value":
    #         emo_a = round((acid_ct * 56100) / (sum(composition)), 2)
    #     elif emo == "OH_Value":
    #         emo_a = round((alcohol_ct * 56100) / (sum(composition)), 2)
    #     try:
    #         composition = [composition[x:x + len(a.comp)] for x in range(0, len(composition), len(a.comp))]
    #         composition_tuple = [tuple(item) for item in composition]
    #         IDLIST = [IDLIST[x:x + len(a.comp)] for x in range(0, len(IDLIST), len(a.comp))]
    #         IDLIST_tuple = [tuple(item) for item in IDLIST]
    #     except TypeError:
    #         composition = [composition[x:x + 1] for x in range(0, len(composition), 1)]
    #         composition_tuple = [tuple(item) for item in composition]
    #         IDLIST = [IDLIST[x:x + 1] for x in range(0, len(IDLIST), 1)]
    #         IDLIST_tuple = [tuple(item) for item in IDLIST]
    #     composition_tuple = [list(item) for item in composition_tuple]
    #     IDLIST_tuple = [list(item) for item in IDLIST_tuple]
    # 
    #     # runs the reaction
    #     while emo_a > emr and running == True:
    #         RC = random.choice(list(enumerate(IDLIST_tuple)))
    #         RCR_temp = random.choice(list(enumerate(RC[1])))
    #         RCR = RCR_temp[1]
    #         RCR_index = RCR_temp[0]
    #         RC2 = random.choice(list(enumerate(IDLIST_tuple)))
    #         RCR2_temp = random.choice(list(enumerate(RC2[1])))
    #         RCR2 = RCR2_temp[1]
    #         RCR2_index = RCR2_temp[0]
    #         while RCR == RCR2 and RC[0] == RC2[0]:
    #             RC = random.choice(list(enumerate(IDLIST_tuple)))
    #             RCR_temp = random.choice(list(enumerate(RC[1])))
    #             RCR = RCR_temp[1]
    #             RCR_index = RCR_temp[0]
    #             RC2 = random.choice(list(enumerate(IDLIST_tuple)))
    #             RCR2_temp = random.choice(list(enumerate(RC2[1])))
    #             RCR2 = RCR2_temp[1]
    #             RCR2_index = RCR2_temp[0]
    #         # randomly select another value from RCR2_index other than RCR2_value
    #         RCR2_other = random.choice(list(enumerate(RC2[1])))
    #         while RCR2_other[0] == RCR2_index:
    #             RCR2_other = random.choice(list(enumerate(RC2[1])))
    #         RCR2_other_index = RCR2_other[0]
    #         if RCR != RCR2 and RC[0] != RC2[0]:
    #             composition_tuple[RC[0]][RCR_index] += (sum(composition_tuple[RC2[0]]) - rt.wl)
    #             IDLIST_tuple[RC[0]][RCR_index] = IDLIST_tuple[RC2[0]][RCR2_other_index]
    #             del composition_tuple[RC2[0]]
    #             del IDLIST_tuple[RC2[0]]
    #         else:
    #             pass
    # 
    #         # determines current status of reaction
    #         composition_tuple_temp = list(itertools.chain(*composition_tuple))
    #         IDLIST = [None] * len(composition_tuple_temp)
    #         amine_ct = 0
    #         acid_ct = 0
    #         alcohol_ct = 0
    #         for chain in composition_tuple_temp:
    #             for chain_ID in range(0, len(chain_lengths_id)):
    #                 if math.isclose(chain, chain_lengths_id[chain_ID][0][1], abs_tol=1):
    #                     ID = chain_lengths_id[chain_ID][1]
    #                     IDLIST.append(ID)
    #                     if ID == "Amine":
    #                         amine_ct += 1
    #                     elif ID == "Acid":
    #                         acid_ct += 1
    #                     elif ID == "Alcohol":
    #                         alcohol_ct += 1
    #                     break
    #         if emo == "Amine_Value":
    #             emo_a = round((amine_ct * 56100) / (sum(composition_tuple_temp)), 2)
    #         elif emo == "Acid_Value":
    #             emo_a = round((acid_ct * 56100) / (sum(composition_tuple_temp)), 2)
    #         elif emo == "OH_Value":
    #             emo_a = round((alcohol_ct * 56100) / (sum(composition_tuple_temp)), 2)
    #         sim.progress['value'] = round((emo_a / emr) * 100, 2)
    #         window.update()
    # 
    #     composition_tuple = [tuple(item) for item in composition_tuple]
    # 
    # # Tabulates final composition and converts to dataframe
    # rxn_summary = collections.Counter(composition_tuple)
    # RS = []
    # for key in rxn_summary:
    #     MS = sum(key)
    #     for item in final_product_masses:
    #         if math.isclose(MS, final_product_masses[item], abs_tol=1):
    #             RS.append((item, rxn_summary[key], key))
    # 
    # # Convert RS to dataframe
    # rxn_summary_df = pandas.DataFrame(RS, columns=['Product', 'Count', 'Mass Distribution'])
    # rxn_summary_df.set_index('Product', inplace=True)
    # # rxn_summary_df.loc[f"{b.sn}"] = [unreacted, b.mw]
    # 
    # # print each value in each row from Mass Distribution
    # if rt.name == PolyCondensation:
    #     for i in range(len(rxn_summary_df)):
    #         amine_ct = 0
    #         acid_ct = 0
    #         alcohol_ct = 0
    #         try:
    #             for j in range(len(rxn_summary_df.iloc[i]['Mass Distribution'])):
    #                 for chain_length in range(0, len(chain_lengths_id)):
    #                     if math.isclose(rxn_summary_df.iloc[i]['Mass Distribution'][j],
    #                                     chain_lengths_id[chain_length][0][1], abs_tol=1):
    #                         chain_ID = chain_lengths_id[chain_length][1]
    #                         if chain_ID == "Amine":
    #                             amine_ct += 1
    #                         if chain_ID == "Acid":
    #                             acid_ct += 1
    #                         if chain_ID == "Alcohol":
    #                             alcohol_ct += 1
    #                         break
    #             amine_value = round((amine_ct * 56100) / sum((rxn_summary_df.iloc[i]['Mass Distribution'])), 2)
    #             acid_value = round((acid_ct * 56100) / sum((rxn_summary_df.iloc[i]['Mass Distribution'])), 2)
    #             alcohol_value = round((alcohol_ct * 56100) / sum((rxn_summary_df.iloc[i]['Mass Distribution'])), 2)
    #             rxn_summary_df.loc[f"{rxn_summary_df.index[i]}", "Amine Value"] = amine_value
    #             rxn_summary_df.loc[f"{rxn_summary_df.index[i]}", "Acid Value"] = acid_value
    #             rxn_summary_df.loc[f"{rxn_summary_df.index[i]}", "OH Value"] = alcohol_value
    #         except TypeError:
    #             chain_ID = "Amine"
    #             amine_ct += 1
    # 
    # global expanded_results
    # expanded_results = rxn_summary_df
    # 
    # # Add columns to dataframe
    # rxn_summary_df['Molar Mass'] = rxn_summary_df.index.map(final_product_masses.get)
    # rxn_summary_df.sort_values(by=['Molar Mass'], ascending=True, inplace=True)
    # rxn_summary_df['Mass'] = rxn_summary_df['Molar Mass'] * rxn_summary_df['Count']
    # rxn_summary_df['Mol %'] = round(rxn_summary_df['Count'] / rxn_summary_df['Count'].sum() * 100, 4)
    # rxn_summary_df['Wt %'] = round(rxn_summary_df['Mass'] / rxn_summary_df['Mass'].sum() * 100, 4)
    # try:
    #     rxn_summary_df['OH Value'] = round(rxn_summary_df['OH Value'] * rxn_summary_df['Wt %'] / 100, 2)
    #     rxn_summary_df['Amine Value'] = round(rxn_summary_df['Amine Value'] * rxn_summary_df['Wt %'] / 100, 2)
    #     rxn_summary_df['Acid Value'] = round(rxn_summary_df['Acid Value'] * rxn_summary_df['Wt %'] / 100, 2)
    # except KeyError:
    #     pass
    # 
    # # Add ehc to dataframe if rt == Etherification
    # if rt.name == Etherification:
    #     ehc = []
    #     for i in rxn_summary_df["Mass Distribution"]:
    #         try:
    #             EHCCount = 0
    #             EHCCount += sum(chain_weight > max(a.comp) for chain_weight in i)
    #             ehc.append(((EHCCount * 35.453) / sum(i)) * 100)
    #         except TypeError:
    #             try:
    #                 ehc.append(35.453 / i * 100)
    #             except TypeError:
    #                 if sum(i) == a.mw:
    #                     ehc.append(0)
    #                 else:
    #                     ehc.append(35.453 / sum(i) * 100)
    #     rxn_summary_df['ehc'] = ehc
    #     rxn_summary_df['% ehc'] = (rxn_summary_df['ehc'] * rxn_summary_df['Wt %']) / 100
    #     EHCp = round(rxn_summary_df['% ehc'].sum(), 4)
    #     RM.update_EHC(round(EHCp, 2))
    #     WPE = (3545.3 / EHCp) - 36.4
    #     RM.update_WPE(round(WPE, 2))
    # 
    # # sum rxn_summary_df by product but keep Molar mass the same
    # rxn_summary_df = rxn_summary_df.groupby(['Product', 'Molar Mass']).sum(numeric_only=True)
    # rxn_summary_df.sort_values(by=['Molar Mass'], ascending=True, inplace=True)
    # rxn_summary_df_compact = rxn_summary_df.groupby(['Product', 'Molar Mass']).sum()
    # rxn_summary_df_compact.sort_values(by=['Molar Mass'], ascending=True, inplace=True)
    # 
    # if rt.name == PolyCondensation:
    #     RM.updateAV(round(rxn_summary_df['Acid Value'].sum(), 2))
    #     RM.updateTAV(round(rxn_summary_df['Amine Value'].sum(), 2))
    #     RM.updateOHV(round(rxn_summary_df['OH Value'].sum(), 2))
    # 
    # show_results(rxn_summary_df_compact)


# -------------------------------------------Aux Functions---------------------------------#

def show_results(rxn_summary_df_compact):
    global results, frame
    try:
        results.destroy()
        frame.destroy()
    except NameError:
        pass
    frame = tkinter.Frame(window)
    x = ((window.winfo_screenwidth() - frame.winfo_reqwidth()) / 2) + 100
    y = (window.winfo_screenheight() - frame.winfo_reqheight()) / 2
    frame.place(x=x, y=y, anchor='center')
    results = Table(frame, dataframe=rxn_summary_df_compact, showtoolbar=True, showstatusbar=True, showindex=True,
                    width=x, height=y, align='center')
    results.show()


def show_results_expanded():
    global results, frame
    try:
        results.destroy()
        frame.destroy()
    except NameError:
        pass
    frame = tkinter.Frame(window)
    x = ((window.winfo_screenwidth() - frame.winfo_reqwidth()) / 2) + 100
    y = (window.winfo_screenheight() - frame.winfo_reqheight()) / 2
    frame.place(x=x, y=y, anchor='center')
    results = Table(frame, dataframe=expanded_results, showtoolbar=True, showstatusbar=True, showindex=True, width=x,
                    height=y, align='center')
    results.show()


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


def stop():
    global running
    if running:
        running = False
    else:
        pass


def sim_values():
    cell = 15
    index = 0
    starting_materials = []
    for i in range(RET.tableheight - 1):
        if RET.entries[cell].get() != "" and RET.entries[cell + 1].get() != "":
            str_to_class(RDE[index]).assign(name=str_to_class(Entry_Reactants[index].get())(),
                                            mass=Entry_masses[index].get(),
                                            moles=round(float(RET.entries[cell + 2].get()), 4),
                                            prgID=RET.entries[cell + 3].get(), prgk=RET.entries[cell + 4].get(),
                                            cprgID=RET.entries[cell + 5].get(), cprgk=RET.entries[cell + 6].get(),
                                            srgID=RET.entries[cell + 7].get(), srgk=RET.entries[cell + 8].get(),
                                            csrgID=RET.entries[cell + 9].get(), csrgk=RET.entries[cell + 10].get(),
                                            trgID=RET.entries[cell + 11].get(), trgk=RET.entries[cell + 12].get(),
                                            ctrgID=RET.entries[cell + 13].get(), ctrgk=RET.entries[cell + 14].get(),
                                            ct=RXN_Samples.get())
            cell = cell + RET.tablewidth
            starting_materials.append(str_to_class(RDE[index]).comp)
            index = index + 1
        else:
            break
    simulate(starting_materials)
    # try:
    #     simulate(a=A1reactant.get(), b=B1reactant.get(), rt=RXN_Type.get(), samples=RXN_Samples.get(), eor=RXN_EOR.get(),
    #              a_mass=massA1.get(), b_mass=massB1.get(), prgk=RET.entries[24].get(), srgk=RET.entries[34].get(), crgk=RET.entries[29].get(),
    #              emr=RXN_EM_Value.get(), emo=RXN_EM.get())
    # except AttributeError as e:
    #     messagebox.showerror("Exception raised", str(e))
    #     pass


# ---------------------------------------------------User-Interface----------------------------------------------#
window = tkinter.Tk()
window.iconbitmap("testtube.ico")
window.title("Monte Karlo")
window.geometry("{0}x{1}+0+0".format(window.winfo_screenwidth(), window.winfo_screenheight()))
window.configure(background="#000000")

Entry_Reactants = ['R1Reactant', 'R2Reactant', 'R3Reactant', 'R4Reactant', 'R5Reactant', 'R6Reactant', 'R7Reactant',
                   'R8Reactant', 'R9Reactant', 'R10Reactant', 'R11Reactant', 'R12Reactant', 'R13Reactant',
                   'R14Reactant']
Entry_masses = ['R1mass', 'R2mass', 'R3mass', 'R4mass', 'R5mass', 'R6mass', 'R7mass', 'R8mass', 'R9mass', 'R10mass',
                'R11mass', 'R12mass', 'R13mass', 'R14mass']
RDE = ['R1Data', 'R2Data', 'R3Data', 'R4Data', 'R5Data', 'R6Data', 'R7Data', 'R8Data', 'R9Data', 'R10Data', 'R11Data',
       'R12Data', 'R13Data', 'R14Data']

global massA1, massB1, A1reactant, B1reactant, starting_cell
starting_cell = 15


def check_entry(entry, index, cell):
    RET.entries[entry].get()
    if RET.entries[entry].get() not in Reactants and RET.entries[entry].get() != "":
        RET.entries[entry].delete(0, 'end')
        messagebox.showerror("Error", "Please enter a valid reactant")
    else:
        RET.update_table(index, cell)
        RET.update_rates(index, cell)


class RxnEntryTable(tkinter.Frame):
    def __init__(self, master=window):
        tkinter.Frame.__init__(self, master)
        self.tablewidth = 15
        self.tableheight = 15
        self.entries = None
        self.grid(row=5, column=0, padx=5, pady=5)
        self.create_table()

    def create_table(self):
        self.entries = {}
        counter = 0
        for row in range(self.tableheight):
            for column in range(self.tablewidth):
                self.entries[counter] = tkinter.Entry(self)
                self.entries[counter].grid(row=row, column=column)
                # self.entries[counter].insert(0, str(counter))
                self.entries[counter].config(justify="center", width=10)
                counter += 1
        self.entries[0].config(width=27)
        self.tabel_labels()

    def tabel_labels(self):
        offset = 0
        self.entries[offset + 0].insert(0, "Reactant")
        self.entries[offset + 0].config(state="readonly", font=("Helvetica", 8, "bold"))
        self.entries[offset + 1].insert(0, "Mass (g)")
        self.entries[offset + 1].config(state="readonly", font=("Helvetica", 8, "bold"))
        self.entries[offset + 2].insert(0, "Moles")
        self.entries[offset + 2].config(state="readonly", font=("Helvetica", 8, "bold"))
        self.entries[offset + 3].insert(0, "1° - ID")
        self.entries[offset + 3].config(state="readonly", font=("Helvetica", 8, "bold"))
        self.entries[offset + 4].insert(0, "1° - K")
        self.entries[offset + 4].config(state="readonly", font=("Helvetica", 8, "bold"))
        self.entries[offset + 5].insert(0, "C1° - ID")
        self.entries[offset + 5].config(state="readonly", font=("Helvetica", 8, "bold"))
        self.entries[offset + 6].insert(0, "1° - Child K")
        self.entries[offset + 6].config(state="readonly", font=("Helvetica", 8, "bold"))
        self.entries[offset + 7].insert(0, "2° - ID")
        self.entries[offset + 7].config(state="readonly", font=("Helvetica", 8, "bold"))
        self.entries[offset + 8].insert(0, "2° - K")
        self.entries[offset + 8].config(state="readonly", font=("Helvetica", 8, "bold"))
        self.entries[offset + 9].insert(0, "C2° - ID")
        self.entries[offset + 9].config(state="readonly", font=("Helvetica", 8, "bold"))
        self.entries[offset + 10].insert(0, "2° - Child K")
        self.entries[offset + 10].config(state="readonly", font=("Helvetica", 8, "bold"))
        self.entries[offset + 11].insert(0, "3° - ID")
        self.entries[offset + 11].config(state="readonly", font=("Helvetica", 8, "bold"))
        self.entries[offset + 12].insert(0, "3° - K")
        self.entries[offset + 12].config(state="readonly", font=("Helvetica", 8, "bold"))
        self.entries[offset + 13].insert(0, "C3° - ID")
        self.entries[offset + 13].config(state="readonly", font=("Helvetica", 8, "bold"))
        self.entries[offset + 14].insert(0, "3° - Child K")
        self.entries[offset + 14].config(state="readonly", font=("Helvetica", 8, "bold"))
        self.user_entry()

    def user_entry(self):
        cell = starting_cell
        row = 1
        index = 0
        for species in range(self.tableheight - 1):
            Entry_Reactants[index] = tkinter.StringVar()
            self.entries[cell] = AutocompleteCombobox(self, completevalues=Reactants, width=24,
                                                      textvariable=Entry_Reactants[index])
            self.entries[cell].grid(row=row, column=0)
            self.entries[cell].config(justify="center")
            cell = cell + self.tablewidth
            row = row + 1
            index = index + 1
        cell = starting_cell + 1
        row = 1
        index = 0
        for species in range(self.tableheight - 1):
            Entry_masses[index] = self.entries[cell]
            cell = cell + self.tablewidth
            row = row + 1
            index = index + 1

    def update_table(self, index, cell):
        if self.entries[cell].get() == "Clear":
            self.entries[cell].delete(0, tkinter.END)
            self.entries[cell + 1].delete(0, tkinter.END)
            self.entries[cell + 2].delete(0, tkinter.END)
            self.entries[cell + 3].config(state="normal")
            self.entries[cell + 3].delete(0, tkinter.END)
            self.entries[cell + 4].delete(0, tkinter.END)
            self.entries[cell + 5].config(state="normal")
            self.entries[cell + 5].delete(0, tkinter.END)
            self.entries[cell + 6].delete(0, tkinter.END)
            self.entries[cell + 7].config(state="normal")
            self.entries[cell + 7].delete(0, tkinter.END)
            self.entries[cell + 8].delete(0, tkinter.END)
            self.entries[cell + 9].config(state="normal")
            self.entries[cell + 9].delete(0, tkinter.END)
            self.entries[cell + 10].delete(0, tkinter.END)
            self.entries[cell + 11].config(state="normal")
            self.entries[cell + 11].delete(0, tkinter.END)
            self.entries[cell + 12].delete(0, tkinter.END)
            self.entries[cell + 13].config(state="normal")
            self.entries[cell + 13].delete(0, tkinter.END)
            self.entries[cell + 14].delete(0, tkinter.END)
        else:
            if self.entries[cell].get() != "" and self.entries[cell + 1].get() != "":
                a = str_to_class(Entry_Reactants[index].get())()
                molesA = float(Entry_masses[index].get()) / float(a.mw)
                self.entries[cell + 2].delete(0, tkinter.END)
                self.entries[cell + 2].insert(0, str(round(molesA, 4)))

    def update_rates(self, index, cell):
        if self.entries[cell].get() != "Clear" and self.entries[cell].get() != "":
            a = str_to_class(Entry_Reactants[index].get())()
            self.entries[cell + 3].config(state="normal")
            self.entries[cell + 3].delete(0, tkinter.END)
            self.entries[cell + 3].insert(0, str(a.prgID))
            self.entries[cell + 3].config(state="readonly")
            self.entries[cell + 4].delete(0, tkinter.END)
            self.entries[cell + 4].insert(0, str(a.prgk))
            self.entries[cell + 5].config(state="normal")
            self.entries[cell + 5].delete(0, tkinter.END)
            self.entries[cell + 5].insert(0, str(a.cprgID))
            self.entries[cell + 5].config(state="readonly")
            self.entries[cell + 6].delete(0, tkinter.END)
            self.entries[cell + 6].insert(0, str(a.cprgk))
            self.entries[cell + 7].config(state="normal")
            self.entries[cell + 7].delete(0, tkinter.END)
            self.entries[cell + 7].insert(0, str(a.srgID))
            self.entries[cell + 7].config(state="readonly")
            self.entries[cell + 8].delete(0, tkinter.END)
            self.entries[cell + 8].insert(0, str(a.srgk))
            self.entries[cell + 9].config(state="normal")
            self.entries[cell + 9].delete(0, tkinter.END)
            self.entries[cell + 9].insert(0, str(a.csrgID))
            self.entries[cell + 9].config(state="readonly")
            self.entries[cell + 10].delete(0, tkinter.END)
            self.entries[cell + 10].insert(0, str(a.csrgk))
            self.entries[cell + 11].config(state="normal")
            self.entries[cell + 11].delete(0, tkinter.END)
            self.entries[cell + 11].insert(0, str(a.trgID))
            self.entries[cell + 11].config(state="readonly")
            self.entries[cell + 12].delete(0, tkinter.END)
            self.entries[cell + 12].insert(0, str(a.trgk))
            self.entries[cell + 13].config(state="normal")
            self.entries[cell + 13].delete(0, tkinter.END)
            self.entries[cell + 13].insert(0, str(a.ctrgID))
            self.entries[cell + 13].config(state="readonly")
            self.entries[cell + 14].delete(0, tkinter.END)
            self.entries[cell + 14].insert(0, str(a.ctrgk))
        else:
            pass


global RXN_Type, RXN_Samples, RXN_EOR


class RxnDetails(tkinter.Frame):
    def __init__(self, master=window):
        tkinter.Frame.__init__(self, master)
        self.tableheight = None
        self.tablewidth = None
        self.entries = None
        self.grid(row=0, column=0, padx=5, pady=(20, 5))
        self.create_table()

    def create_table(self):
        self.entries = {}
        self.tableheight = 3
        self.tablewidth = 2
        counter = 0
        for column in range(self.tablewidth):
            for row in range(self.tableheight):
                self.entries[counter] = tkinter.Entry(self)
                self.entries[counter].grid(row=row, column=column)
                # self.entries[counter].insert(0, str(counter))
                self.entries[counter].config(justify="center", width=18)
                counter += 1
        self.table_labels()

    def table_labels(self):
        self.entries[0].delete(0, tkinter.END)
        self.entries[0].insert(0, "Reaction Type =")
        self.entries[1].delete(0, tkinter.END)
        self.entries[1].insert(0, "# of Samples =")
        self.entries[2].delete(0, tkinter.END)
        self.entries[2].insert(0, "Extent of Reaction =")
        self.entries[5].delete(0, tkinter.END)
        self.entries[5].insert(0, "1")
        self.user_entry()

    def user_entry(self):
        global RXN_Type, RXN_Samples, RXN_EOR
        RXN_Type = tkinter.StringVar()
        RXN_Type_Entry = AutocompleteCombobox(self, completevalues=Reactions, width=15, textvariable=RXN_Type)
        RXN_Type_Entry.grid(row=0, column=1)
        RXN_Type_Entry.config(justify="center")
        RXN_Samples = tkinter.StringVar()
        RXN_Samples_Entry = AutocompleteCombobox(self, completevalues=Num_Samples, width=15, textvariable=RXN_Samples)
        RXN_Samples_Entry.insert(0, "1000")
        RXN_Samples_Entry.grid(row=1, column=1)
        RXN_Samples_Entry.config(justify="center")
        RXN_EOR = self.entries[5]


global RXN_EM, RXN_EM_Value


class RxnMetrics(tkinter.Frame):
    def __init__(self, master=window):
        tkinter.Frame.__init__(self, master)
        self.tablewidth = None
        self.tableheight = None
        self.entries = None
        self.grid(row=3, column=0, padx=5, pady=5)
        self.create_table()

    def create_table(self):
        self.entries = {}
        self.tableheight = 6
        self.tablewidth = 2
        counter = 0
        for column in range(self.tablewidth):
            for row in range(self.tableheight):
                self.entries[counter] = tkinter.Entry(self)
                self.entries[counter].grid(row=row, column=column)
                # self.entries[counter].insert(0, str(counter))
                self.entries[counter].config(justify="center", width=18)
                counter += 1
        self.table_labels()

    def table_labels(self):
        self.entries[0].delete(0, tkinter.END)
        self.entries[0].insert(0, "EHC, % =")
        self.entries[1].delete(0, tkinter.END)
        self.entries[1].insert(0, "Theory WPE =")
        self.entries[2].delete(0, tkinter.END)
        self.entries[2].insert(0, "Acid Value =")
        self.entries[3].delete(0, tkinter.END)
        self.entries[3].insert(0, "Amine Value =")
        self.entries[4].delete(0, tkinter.END)
        self.entries[4].insert(0, "OH Value =")
        self.user_entry()

    def user_entry(self):
        global RXN_EM, RXN_EM_Value
        RXN_EM = tkinter.StringVar()
        RXN_EM_Entry = AutocompleteCombobox(self, completevalues=End_Metrics, width=15, textvariable=RXN_EM)
        RXN_EM_Entry.grid(row=5, column=0)
        RXN_EM_Entry.config(justify="center")
        RXN_EM_Entry.insert(0, "End Metric")
        RXN_EM_Value = self.entries[11]

    def update_EHC(self, EHCpercent):
        self.entries[6].delete(0, tkinter.END)
        self.entries[6].insert(0, EHCpercent)

    def update_WPE(self, WPE):
        self.entries[7].delete(0, tkinter.END)
        self.entries[7].insert(0, WPE)

    def updateAV(self, AV):
        self.entries[8].delete(0, tkinter.END)
        self.entries[8].insert(0, AV)

    def updateTAV(self, TAV):
        self.entries[9].delete(0, tkinter.END)
        self.entries[9].insert(0, TAV)

    def updateOHV(self, OHV):
        self.entries[10].delete(0, tkinter.END)
        self.entries[10].insert(0, OHV)


class Buttons(tkinter.Frame):
    def __init__(self, master=window):
        tkinter.Frame.__init__(self, master)
        self.tablewidth = None
        self.tableheight = None
        self.entries = None
        self.grid(row=1, column=0, padx=5, pady=5)
        self.create_table()

    def create_table(self):
        self.entries = {}
        self.tableheight = 3
        self.tablewidth = 1
        counter = 0
        for column in range(self.tablewidth):
            for row in range(self.tableheight):
                self.entries[counter] = tkinter.Entry(self)
                self.entries[counter].grid(row=row, column=column)
                self.entries[counter].insert(0, str(counter))
                self.entries[counter].config(justify="center", width=18)
                counter += 1
        self.add_buttons()

    def add_buttons(self):
        Simulate = tkinter.Button(self, text="Simulate", command=sim_values, width=15, bg="Green")
        Simulate.grid(row=0, column=0)
        stop_button = tkinter.Button(self, text="Stop", command=stop, width=15, bg="Red")
        stop_button.grid(row=1, column=0)
        expand = tkinter.Button(self, text="Expand Data", command=show_results_expanded, width=15, bg="blue")
        expand.grid(row=2, column=0)


class SimStatus(tkinter.Frame):
    def __init__(self, master=window):
        tkinter.Frame.__init__(self, master)
        self.tablewidth = None
        self.tableheight = None
        self.progress = None
        self.entries = None
        self.grid(row=0, column=2)
        self.create_table()

    def create_table(self):
        self.entries = {}
        self.tableheight = 1
        self.tablewidth = 2
        counter = 0
        for column in range(self.tablewidth):
            for row in range(self.tableheight):
                self.entries[counter] = tkinter.Entry(self)
                self.entries[counter].grid(row=row, column=column)
                self.entries[counter].insert(0, str(counter))
                self.entries[counter].config(justify="center", width=18)
                counter += 1
        self.tabel_labels()

    def tabel_labels(self):
        self.entries[0].delete(0, tkinter.END)
        self.entries[0].insert(0, "Simulation Status")
        self.entries[0].config(state="readonly")
        self.add_buttons()

    def add_buttons(self):
        self.progress = ttk.Progressbar(self, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=0, column=1)


RET = RxnEntryTable()
RD = RxnDetails()
RM = RxnMetrics()
Buttons = Buttons()
sim = SimStatus()

# run update_table if user changes value in RET

RET.entries[15].bind('<FocusOut>', lambda *args, entry=15, index=0, cell=15: check_entry(entry, index, cell))
RET.entries[30].bind('<FocusOut>', lambda *args, entry=30, index=1, cell=30: check_entry(entry, index, cell))
RET.entries[45].bind('<FocusOut>', lambda *args, entry=45, index=2, cell=45: check_entry(entry, index, cell))
RET.entries[60].bind('<FocusOut>', lambda *args, entry=60, index=3, cell=60: check_entry(entry, index, cell))
RET.entries[75].bind('<FocusOut>', lambda *args, entry=75, index=4, cell=75: check_entry(entry, index, cell))
RET.entries[90].bind('<FocusOut>', lambda *args, entry=90, index=5, cell=90: check_entry(entry, index, cell))
RET.entries[105].bind('<FocusOut>', lambda *args, entry=105, index=6, cell=105: check_entry(entry, index, cell))
RET.entries[120].bind('<FocusOut>', lambda *args, entry=120, index=7, cell=120: check_entry(entry, index, cell))
RET.entries[135].bind('<FocusOut>', lambda *args, entry=135, index=8, cell=135: check_entry(entry, index, cell))
RET.entries[150].bind('<FocusOut>', lambda *args, entry=150, index=9, cell=150: check_entry(entry, index, cell))
RET.entries[165].bind('<FocusOut>', lambda *args, entry=165, index=10, cell=165: check_entry(entry, index, cell))
RET.entries[180].bind('<FocusOut>', lambda *args, entry=180, index=11, cell=180: check_entry(entry, index, cell))
RET.entries[195].bind('<FocusOut>', lambda *args, entry=195, index=12, cell=195: check_entry(entry, index, cell))
RET.entries[210].bind('<FocusOut>', lambda *args, entry=210, index=13, cell=210: check_entry(entry, index, cell))

Entry_masses[0].bind("<KeyRelease>", lambda *args, index=0, cell=15: RET.update_table(index, cell))
Entry_masses[1].bind("<KeyRelease>", lambda *args, index=1, cell=30: RET.update_table(index, cell))
Entry_masses[2].bind("<KeyRelease>", lambda *args, index=2, cell=45: RET.update_table(index, cell))
Entry_masses[3].bind("<KeyRelease>", lambda *args, index=3, cell=60: RET.update_table(index, cell))
Entry_masses[4].bind("<KeyRelease>", lambda *args, index=4, cell=75: RET.update_table(index, cell))
Entry_masses[5].bind("<KeyRelease>", lambda *args, index=5, cell=90: RET.update_table(index, cell))
Entry_masses[6].bind("<KeyRelease>", lambda *args, index=6, cell=105: RET.update_table(index, cell))
Entry_masses[7].bind("<KeyRelease>", lambda *args, index=7, cell=120: RET.update_table(index, cell))
Entry_masses[8].bind("<KeyRelease>", lambda *args, index=8, cell=135: RET.update_table(index, cell))
Entry_masses[9].bind("<KeyRelease>", lambda *args, index=9, cell=150: RET.update_table(index, cell))
Entry_masses[10].bind("<KeyRelease>", lambda *args, index=10, cell=165: RET.update_table(index, cell))
Entry_masses[11].bind("<KeyRelease>", lambda *args, index=11, cell=180: RET.update_table(index, cell))
Entry_masses[12].bind("<KeyRelease>", lambda *args, index=12, cell=195: RET.update_table(index, cell))
Entry_masses[13].bind("<KeyRelease>", lambda *args, index=13, cell=210: RET.update_table(index, cell))

R1Data = R1Data()
R2Data = R2Data()
R3Data = R3Data()
R4Data = R4Data()
R5Data = R5Data()
R6Data = R6Data()
R7Data = R7Data()
R8Data = R8Data()
R9Data = R9Data()
R10Data = R10Data()
R11Data = R11Data()
R12Data = R12Data()
R13Data = R13Data()
R14Data = R14Data()
rg = reactive_groups()

window.mainloop()
