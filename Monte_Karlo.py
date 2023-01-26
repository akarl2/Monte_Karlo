from collections import Counter
import random
import sys
import tkinter
from tkinter import ttk, messagebox
import pandas
from ttkwidgets.autocomplete import AutocompleteCombobox
from Database import *
from Reactions import reactive_groups,NH2,NH,COOH,COC,POH,SOH,C3OHCl
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
global running, emo_a, results, frame, expanded_results, groupA, groupB
def simulate(starting_materials):
    global running
    running = True
    sim.progress['value'] = 0
    end_metric_selection = str(RXN_EM.get())
    end_metric_value = float(RXN_EM_Value.get())
    composition = []
    for compound in starting_materials:
        for i in range(compound[3][0]):
            inner_result = []
            for group in compound[0]:
                inner_result.append([group[0], group[1]])
            composition.append([inner_result, compound[2], compound[1]])

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
        new_name = {}
        for group, count in compoundA[1] + compoundB[1]:
            if group in new_name:
                new_name[group] += count
            else:
                new_name[group] = count
        new_name = [[group, count] for group, count in new_name.items()]
        new_name.sort(key=lambda x: x[0])
        NW = compoundA[2][0] + compoundB[2][0] - new_group(groupA, groupB)['WL']
        NC = [[[group[0], group[1]] for group in NC[0]], new_name, [round(NW,2)]]
        NC[0][groups[0][1]][0] = new_group(groupA, groupB)['NG']
        NC[0][groups[0][1]][1] = 0.06
        old_groups = compoundB[0]
        if len(old_groups) == 1:
            pass
        else:
            del(old_groups[compoundBloc])

            for sublist in old_groups:
                NC[0].append(sublist)
        NC[0].sort(key=lambda x: x[0])
        composition[groups[0][0]] = NC
        del(composition[groups[1][0]])
        RXN_Status(composition)

    def RXN_Status(composition):
        global running
        comp_summary = Counter([(tuple(tuple(i) for i in sublist[0]), tuple(tuple(i) for i in sublist[1]), sublist[2][0]) for sublist in composition])
        sum_comp = 0
        for key in comp_summary:
            sum_comp = sum_comp + (comp_summary[key] * key[2])
        amine_ct = 0
        acid_ct = 0
        alcohol_ct = 0
        epoxide_ct = 0
        EHC_ct = 0
        for key in comp_summary:
            for group in key[0]:
                if group[0] == 'NH2' or group[0] == 'NH':
                    amine_ct += comp_summary[key]
                elif group[0] == 'COOH':
                    acid_ct += comp_summary[key]
                elif group[0] == 'OH':
                    alcohol_ct += comp_summary[key]
                elif group[0] == 'COC':
                    epoxide_ct += comp_summary[key]
                elif group[0] == 'C3OHCl':
                    EHC_ct += comp_summary[key]
        TAV = round((amine_ct * 56100) / (sum_comp), 2)
        AV = round((acid_ct * 56100) / (sum_comp), 2)
        OH = round((alcohol_ct * 56100) / (sum_comp), 2)
        COC = round((epoxide_ct * 56100) / (sum_comp), 2)
        EHC = round((EHC_ct * 35.453) / (sum_comp) * 100, 2)
        print(EHC)

        if end_metric_selection == 'Amine Value':
            sim.progress['value'] = round(((end_metric_value / TAV ) * 100), 2)
            if TAV <= end_metric_value:
                running = False
                RXN_Results(composition, TAV, AV, OH, EHC)
                sim.progress['value'] = 100
            window.update()
        elif end_metric_selection == 'Acid Value':
            sim.progress['value'] = round(((end_metric_value / AV) * 100), 2)
            if AV <= end_metric_value:
                running = False
                RXN_Results(composition, TAV, AV, OH, EHC)
                sim.progress['value'] = 100
            window.update()
        elif end_metric_selection == 'OH Value':
            sim.progress['value'] = round(((end_metric_value / OH) * 100), 2)
            if OH <= end_metric_value:
                running = False
                RXN_Results(composition, TAV, AV, OH, EHC)
                sim.progress['value'] = 100
            window.update()
        elif end_metric_selection == 'Epoxide Value':
            sim.progress['value'] = round(((end_metric_value / COC) * 100), 2)
            if COC <= end_metric_value:
                running = False
                RXN_Results(composition, TAV, AV, OH, EHC)
                sim.progress['value'] = 100
            window.update()
        elif end_metric_selection == '% EHC':
            sim.progress['value'] = round(((EHC / end_metric_value) * 100), 2)
            if EHC >= end_metric_value:
                running = False
                RXN_Results(composition, TAV, AV, OH, EHC)
                sim.progress['value'] = 100
            window.update()

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

def RXN_Results(composition, TAV, AV, OH, EHC):
    RM.entries[6].delete(0, tkinter.END)
    RM.entries[6].insert(0, EHC)
    RM.entries[7].delete(0, tkinter.END)
    RM.entries[7].insert(0, (3545.3 / EHC) - 36.4)
    RM.entries[8].delete(0, tkinter.END)
    RM.entries[8].insert(0, AV)
    RM.entries[9].delete(0, tkinter.END)
    RM.entries[9].insert(0, TAV)
    RM.entries[10].delete(0, tkinter.END)
    RM.entries[10].insert(0, OH)
    comp_summary = Counter([(tuple(tuple(i) for i in sublist[0]), tuple(tuple(i) for i in sublist[1]), sublist[2][0]) for sublist in composition])
    RS = []
    for key in comp_summary:
        RS.append((key[0], str(key[1]), key[2], comp_summary[key]))
    rxn_summary_df = pandas.DataFrame(RS, columns=['Groups', 'Name', 'MW', 'Count'])
    rxn_summary_df.set_index('Name', inplace=True)
    rxn_summary_df['Mass'] = rxn_summary_df['MW'] * rxn_summary_df['Count']
    rxn_summary_df['Mol %'] = round(rxn_summary_df['Count'] / rxn_summary_df['Count'].sum() * 100, 4)
    rxn_summary_df['Wt %'] = round(rxn_summary_df['Mass'] / rxn_summary_df['Mass'].sum() * 100, 4)
    rxn_summary_df.sort_values(by=['MW'], ascending=True, inplace=True)
    rxn_summary_df['Groups'] = rxn_summary_df['Groups'].apply(lambda x: str(x)[1:-1])
    rxn_summary_df_compact = rxn_summary_df

    show_results(rxn_summary_df_compact)


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


# -------------------------------------------Aux Functions---------------------------------#

def show_results(rxn_summary_df_compact):
    global results, frame
    try:
        results.destroy()
        frame.destroy()
    except NameError:
        pass
    frame = tkinter.Frame(tab2)
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
    try:
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
    except AttributeError as e:
        messagebox.showerror("Exception raised", str(e))
        pass
    simulate(starting_materials)

# ---------------------------------------------------User-Interface----------------------------------------------#
window = tkinter.Tk()
style = ttk.Style()
style.configure('TNotebook.Tab', background="red")
window.iconbitmap("testtube.ico")
window.title("Monte Karlo")
window.geometry("{0}x{1}+0+0".format(window.winfo_screenwidth(), window.winfo_screenheight()))
window.configure(background="#000000")

tab_control = ttk.Notebook(window)
tab1 = ttk.Frame(tab_control, style='TNotebook.Tab')
tab2 = ttk.Frame(tab_control, style='TNotebook.Tab')
tab_control.add(tab1, text='Reactor')
tab_control.add(tab2, text='Reaction Results')
tkinter.Grid.rowconfigure(window, 0, weight=1)
tkinter.Grid.columnconfigure(window, 0, weight=1)
tab_control.grid(row=0, column=0, sticky=tkinter.E + tkinter.W + tkinter.N + tkinter.S)


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
    def __init__(self, master=tab1):
        tkinter.Frame.__init__(self, master)
        self.tablewidth = 15
        self.tableheight = 15
        self.entries = None
        self.grid(row=5, column=1, padx=10, pady=10)
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
            self.entries[cell] = AutocompleteCombobox(self, completevalues=Reactants, width=24, textvariable=Entry_Reactants[index])
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
    def __init__(self, master=tab1):
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
        self.entries[1].config(state="readonly")
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
    def __init__(self, master=tab1):
        tkinter.Frame.__init__(self, master)
        self.tablewidth = None
        self.tableheight = None
        self.entries = None
        self.grid(row=2, column=1, padx=5, pady=5)
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
        self.entries[0].config(state="readonly")
        self.entries[1].delete(0, tkinter.END)
        self.entries[1].insert(0, "Theory WPE =")
        self.entries[1].config(state="readonly")
        self.entries[2].delete(0, tkinter.END)
        self.entries[2].insert(0, "Acid Value =")
        self.entries[2].config(state="readonly")
        self.entries[3].delete(0, tkinter.END)
        self.entries[3].insert(0, "Amine Value =")
        self.entries[3].config(state="readonly")
        self.entries[4].delete(0, tkinter.END)
        self.entries[4].insert(0, "OH Value =")
        self.entries[4].config(state="readonly")
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
    def __init__(self, master=tab1):
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
    def __init__(self, master=tab1):
        tkinter.Frame.__init__(self, master)
        self.tablewidth = None
        self.tableheight = None
        self.progress = None
        self.entries = None
        self.grid(row=0, column=1)
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
