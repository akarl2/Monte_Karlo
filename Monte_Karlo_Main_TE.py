import tkinter

from Module_Imports import *

# Set pandas dataframe display
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)
pandas.set_option('display.width', 100)

global running, emo_a, results, frame, expanded_results
running = False

# Runs the simulation
def simulate(a, b, rt, samples, eor, a_mass, b_mass, prgk, srgk, crgk, emr, emo):
    global running, emo_a, results, expanded_results
    sim.progress['value'] = 0
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
            elif group == b.prgmw:
                weights.append(1)
            else:
                weights.append(int(cgK))
        indicesA = [i for i, x in enumerate(composition) if x == a.prgmw or x == a.srgmw or x != b.prgmw]
        indicesB = [i for i, x in enumerate(composition) if x == b.prgmw]
        while len(indicesA) > 1 and len(indicesB) > 1 and running == True:
            indicesA = [i for i, x in enumerate(composition) if x == a.prgmw or x == a.srgmw or x != b.prgmw]
            weightsA = [weights[i] for i in indicesA]
            indicesB = [i for i, x in enumerate(composition) if x == b.prgmw]
            weightsB = [weights[i] for i in indicesB]
            MCA = random.choices(list(enumerate(indicesA)), weights=weightsA, k=1)[0]
            MCB = random.choices(list(enumerate(indicesB)), weights=weightsB, k=1)[0]
            composition[MCA[1]] = round(composition[MCA[1]] + composition[MCB[1]] - rt.wl, 3)
            composition.pop(MCB[1])
            weights[MCA[0]] = cgK
            b.mol -= 1
            sim.progress['value'] = round(100 - (b.mol / bms * 100), 1)
            window.update()
        try:
            composition = [composition[x:x + len(a.comp)] for x in range(0, len(composition), len(a.comp))]
            composition_tuple = [tuple(item) for item in composition]
        except TypeError:
            composition = [composition[x:x + 1] for x in range(0, len(composition), 1)]
            composition_tuple = [tuple(item) for item in composition]

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
            composition_tuple = [tuple(item) for item in composition]
            IDLIST = [IDLIST[x:x + len(a.comp)] for x in range(0, len(IDLIST), len(a.comp))]
            IDLIST_tuple = [tuple(item) for item in IDLIST]
        except TypeError:
            composition = [composition[x:x + 1] for x in range(0, len(composition), 1)]
            composition_tuple = [tuple(item) for item in composition]
            IDLIST = [IDLIST[x:x + 1] for x in range(0, len(IDLIST), 1)]
            IDLIST_tuple = [tuple(item) for item in IDLIST]
        composition_tuple = [list(item) for item in composition_tuple]
        IDLIST_tuple = [list(item) for item in IDLIST_tuple]

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
            sim.progress = round((emo_a / emr) * 100, 2)
            window.update()

        composition_tuple = [tuple(item) for item in composition_tuple]

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
    # rxn_summary_df.loc[f"{b.sn}"] = [unreacted, b.mw]

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
        RM.update_EHC(round(EHCp, 2))
        WPE = (3545.3 / EHCp) - 36.4
        RM.update_WPE(round(WPE, 2))

    # sum rxn_summary_df by product but keep Molar mass the same
    rxn_summary_df = rxn_summary_df.groupby(['Product', 'Molar Mass']).sum(numeric_only=True)
    rxn_summary_df.sort_values(by=['Molar Mass'], ascending=True, inplace=True)
    rxn_summary_df_compact = rxn_summary_df.groupby(['Product', 'Molar Mass']).sum()
    rxn_summary_df_compact.sort_values(by=['Molar Mass'], ascending=True, inplace=True)

    if rt.name == PolyCondensation:
        RM.updateAV(round(rxn_summary_df['Acid Value'].sum(), 2))
        RM.updateTAV(round(rxn_summary_df['Amine Value'].sum(), 2))
        RM.updateOHV(round(rxn_summary_df['OH Value'].sum(), 2))

    show_results(rxn_summary_df_compact)

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
    try:
        simulate(a=A1reactant.get(), b=B1reactant.get(), rt=RXN_Type.get(), samples=RXN_Samples.get(), eor=RXN_EOR.get(),
                 a_mass=massA1.get(), b_mass=massB1.get(), prgk=RET.entries[24].get(), srgk=RET.entries[34].get(), crgk=RET.entries[29].get(),
                 emr=RXN_EM_Value.get(), emo=RXN_EM.get())
    except AttributeError as e:
        messagebox.showerror("Exception raised", str(e))
        pass

# ---------------------------------------------------User-Interface----------------------------------------------#
window = tkinter.Tk()
window.iconbitmap("testtube.ico")
window.title("Monte Karlo")
window.geometry("{0}x{1}+0+0".format(window.winfo_screenwidth(), window.winfo_screenheight()))
window.configure(background="#00BFFF")

global massA1, massB1, A1reactant, B1reactant
class RxnEntryTable(tkinter.Frame):
    def __init__(self, master=window):
        tkinter.Frame.__init__(self, master)
        self.tablewidth = 8
        self.tableheight = 5
        self.entries = None
        self.grid(row=10, column=4)
        self.create_table()

    def create_table(self):
        self.entries = {}
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
        self.entries[3].delete(0, tkinter.END)
        self.entries[3].insert(0, "Mass (g)")
        self.entries[3].config(state="readonly", font=("Helvetica", 8, "bold"))
        self.entries[8].delete(0, tkinter.END)
        self.entries[8].insert(0, "Moles")
        self.entries[8].config(state="readonly", font=("Helvetica", 8, "bold"))
        self.entries[13].delete(0, tkinter.END)
        self.entries[13].insert(0, "Reactant")
        self.entries[13].config(state="readonly", font=("Helvetica", 8, "bold"))
        self.entries[15].delete(0, tkinter.END)
        self.entries[15].insert(0, "Mass (g) = ")
        self.entries[15].config(state="readonly", font=("Helvetica", 8, "bold"))
        self.entries[16].delete(0, tkinter.END)
        self.entries[16].insert(0, "Moles = ")
        self.entries[16].config(state="readonly", font=("Helvetica", 8, "bold"))
        self.entries[17].delete(0, tkinter.END)
        self.entries[17].insert(0, "Reactant = ")
        self.entries[17].config(state="readonly", font=("Helvetica", 8, "bold"))
        self.entries[19].delete(0, tkinter.END)
        self.entries[19].insert(0, "1° K")
        self.entries[19].config(state="readonly", font=("Helvetica", 8, "bold"))
        self.entries[23].delete(0, tkinter.END)
        self.entries[23].insert(0, "1° K")
        self.entries[23].config(state="readonly", font=("Helvetica", 8, "bold"))
        self.entries[28].delete(0, tkinter.END)
        self.entries[28].insert(0, "Child K - 1°")
        self.entries[28].config(state="readonly", font=("Helvetica", 8, "bold"))
        self.entries[33].delete(0, tkinter.END)
        self.entries[33].insert(0, "2° K")
        self.entries[33].config(state="readonly", font=("Helvetica", 8, "bold"))
        self.entries[38].delete(0, tkinter.END)
        self.entries[38].insert(0, "Child K - 2°")
        self.entries[38].config(state="readonly", font=("Helvetica", 8, "bold"))
        self.entries[0].delete(0, tkinter.END)
        self.entries[1].delete(0, tkinter.END)
        self.entries[2].delete(0, tkinter.END)
        self.entries[5].delete(0, tkinter.END)
        self.entries[6].delete(0, tkinter.END)
        self.entries[7].delete(0, tkinter.END)
        self.entries[10].delete(0, tkinter.END)
        self.entries[11].delete(0, tkinter.END)
        self.entries[12].delete(0, tkinter.END)
        self.entries[18].delete(0, tkinter.END)
        self.entries[4].delete(0, tkinter.END)
        self.entries[4].insert(0, "100")
        self.entries[20].delete(0, tkinter.END)
        self.entries[20].insert(0, "100")
        self.user_entry()

    def user_entry(self):
        global A1reactant, B1reactant, massA1, massB1
        A1reactant = tkinter.StringVar()
        entryA1 = AutocompleteCombobox(self, completevalues=Reactants, width=15, textvariable=A1reactant)
        entryA1.grid(row=2, column=4)
        entryA1.config(justify="center")
        B1reactant = tkinter.StringVar()
        entryB1 = AutocompleteCombobox(self, completevalues=Reactants, width=15, textvariable=B1reactant)
        entryB1.grid(row=4, column=2)
        entryB1.config(justify="center")
        massA1 = self.entries[20]
        massB1 = self.entries[4]

    def update_table(self):
        try:
            a = str_to_class(A1reactant.get())()
            molesA = float(massA1.get()) / float(a.mw)
            self.entries[21].delete(0, tkinter.END)
            self.entries[21].insert(0, round(molesA, 4))
            self.entries[24].delete(0, tkinter.END)
            self.entries[24].insert(0, a.prgk)
            self.entries[34].delete(0, tkinter.END)
            self.entries[34].insert(0, a.srgk)
            self.entries[29].delete(0, tkinter.END)
            self.entries[29].insert(0, a.crgk)
            self.entries[39].delete(0, tkinter.END)
            self.entries[39].insert(0, a.crgk)
        except:
            pass
        try:
            b = str_to_class(B1reactant.get())()
            molesB = float(massB1.get()) / float(b.mw)
            self.entries[9].delete(0, tkinter.END)
            self.entries[9].insert(0, round(molesB, 4))
        except:
            pass

global RXN_Type, RXN_Samples, RXN_EOR
class RxnDetails(tkinter.Frame):
    def __init__(self, master=window):
        tkinter.Frame.__init__(self, master)
        self.tableheight = None
        self.tablewidth = None
        self.entries = None
        self.grid(row=0, column=4)
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
                self.entries[counter].insert(0, str(counter))
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
        self.grid(row=0, column=10)
        self.create_table()

    def create_table(self):
        self.entries = {}
        self.tableheight = 7
        self.tablewidth = 2
        counter = 0
        for column in range(self.tablewidth):
            for row in range(self.tableheight):
                self.entries[counter] = tkinter.Entry(self)
                self.entries[counter].grid(row=row, column=column)
                self.entries[counter].insert(0, str(counter))
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
        self.entries[5].delete(0, tkinter.END)
        self.entries[5].insert(0, "End Metric =")
        self.user_entry()

    def user_entry(self):
        global RXN_EM, RXN_EM_Value
        RXN_EM = tkinter.StringVar()
        RXN_EM_Entry = AutocompleteCombobox(self, completevalues=End_Metrics, width=15, textvariable=RXN_EM)
        RXN_EM_Entry.grid(row=5, column=1)
        RXN_EM_Entry.config(justify="center")
        RXN_EM_Value = self.entries[13]

    def update_EHC(self, EHCpercent):
        self.entries[7].delete(0, tkinter.END)
        self.entries[7].insert(0, EHCpercent)

    def update_WPE(self, WPE):
        self.entries[8].delete(0, tkinter.END)
        self.entries[8].insert(0, WPE)

    def updateAV(self, AV):
        self.entries[9].delete(0, tkinter.END)
        self.entries[9].insert(0, AV)

    def updateTAV(self, TAV):
        self.entries[10].delete(0, tkinter.END)
        self.entries[10].insert(0, TAV)

    def updateOHV(self, OHV):
        self.entries[11].delete(0, tkinter.END)
        self.entries[11].insert(0, OHV)

class Buttons(tkinter.Frame):
    def __init__(self, master=window):
        tkinter.Frame.__init__(self, master)
        self.tablewidth = None
        self.tableheight = None
        self.entries = None
        self.grid(row=0, column=25)
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

#update table with events
A1reactant.trace("w", lambda name, index, mode, sv=A1reactant: RET.update_table())
B1reactant.trace("w", lambda name, index, mode, sv=B1reactant: RET.update_table())
massA1.bind("<KeyRelease>", lambda event: RET.update_table())
massB1.bind("<KeyRelease>", lambda event: RET.update_table())


window.mainloop()