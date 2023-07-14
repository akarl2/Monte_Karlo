import collections
import queue
import multiprocessing
import os
import random
import sys
import tkinter
import time
from tkinter import *
import mplcursors

from matplotlib.widgets import Cursor
from mpl_toolkits import *
from matplotlib.backends.backend_tkagg import *
import matplotlib.backends.backend_tkagg as tkagg

import numpy as np
from openpyxl import Workbook
import customtkinter
from tkinter import ttk, messagebox, simpledialog, filedialog
import pandas
import concurrent.futures

import scipy
from ttkwidgets.autocomplete import AutocompleteCombobox
from Database import *
from Reactions import reactive_groups, NH2, NH, COOH, COC, POH, SOH
from Reactions import *
import itertools
from pandastable import Table, TableModel, config
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import statsmodels
import math
from Reactants import *
from Reactants import R1Data, R2Data, R3Data, R4Data, R5Data, R6Data, R7Data, R8Data, R9Data, R10Data, R11Data, R12Data, \
    R13Data, R14Data, R15Data, R16Data, R17Data, R18Data, R19Data

# Runs the simulation
global running, emo_a, results_table, frame_results, expanded_results, groupA, groupB, test_count, test_interval, \
    total_ct, total_ct_sec, sn_dist, TAV, AV, OH, COC, EHC, in_situ_values, in_situ_values_sec, Xn_list, Xn_list_sec, byproducts, \
    frame_byproducts, low_group, RXN_EM_2, RXN_EM_Entry_2, RXN_EM_2_SR, reactants_list, RXN_EM_2_SR, RXN_EM_Entry_2_SR, results_table_2, \
    frame_results_2, byproducts_table_2, frame_byproducts_2, RXN_EM_2_Active, RXN_EM_2_Check, RXN_EM_Value_2, in_primary, quick_add, quick_add_comp, \
    RXN_EM_Value, RXN_EM_Entry, rxn_summary_df, rxn_summary_df_2, Xn, Xn_2, end_metric_value, end_metric_value_sec, RXN_EM_2_Active_status, end_metric_selection, \
    end_metric_selection_sec, starting_mass_sec, starting_mass, sn_dict, samples_value, total_samples, canceled_by_user


def simulate(starting_materials, starting_materials_sec, end_metric_value, end_metric_selection, end_metric_value_sec, end_metric_selection_sec, sn_dict, RXN_EM_2_Active_status, total_ct, total_ct_sec, workers, process_queue, PID_list, progress_queue_sec):
    PID_list.append(os.getpid())
    currentPID = os.getpid()
    time.sleep(1)
    rg = reactive_groups()
    global test_count, test_interval, sn_dist, in_situ_values, Xn_list, byproducts, running, in_primary, in_situ_values_sec, Xn_list_sec, quick_add, comp_primary, comp_secondary
    in_situ_values = [[], [], [], [], [], [], [], [], []]
    in_situ_values_sec = [[], [], [], [], [], [], [], [], []]
    Xn_list, Xn_list_sec, byproducts, composition, composition_sec = [], [], [], [], []
    running, in_primary = True, True
    test_count = 0
    test_interval = 40
    try:
        end_metric_value_upper = end_metric_value + 15
        end_metric_value_lower = end_metric_value - 15
        if RXN_EM_2_Active_status:
            end_metric_value_upper_sec = end_metric_value_sec + 15
            end_metric_value_lower_sec = end_metric_value_sec - 15
    except ValueError:
        messagebox.showerror("Error", "Please enter a value for the end metric(s).")
        return

    for compound in starting_materials:
        for i in range(compound[3][0]):
            inner_result = []
            for group in compound[0]:
                inner_result.append([group[0], group[1]])
            composition.append([inner_result, compound[2], compound[1]])

    for compound in starting_materials_sec:
        for i in range(compound[3][0]):
            inner_result = []
            for group in compound[0]:
                inner_result.append([group[0], group[1]])
            composition_sec.append([inner_result, compound[2], compound[1]])

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
        global NG2, new_group_dict
        NG2 = True
        NG = getattr(eval(groupA + '()'), groupB)
        WL = getattr(eval(groupA + '()'), groupB + '_wl')
        WL_ID = getattr(eval(groupA + '()'), groupB + '_wl_id')
        try:
            NG2 = getattr(eval(groupA + '()'), groupB + '_2')
            NG2_WL = getattr(eval(groupA + '()'), groupB + '_wl')
            NG2_WL_ID = getattr(eval(groupA + '()'), groupB + '_wl_id')
        except AttributeError:
            NG2 = False
        if NG2:
            new_group_dict = {'NG': NG, 'WL': WL, 'WL_ID': WL_ID, 'NG2': NG2, 'NG2_WL': NG2_WL, 'NG2_WL_ID': NG2_WL_ID}
        else:
            new_group_dict = {'NG': NG, 'WL': WL, 'WL_ID': WL_ID}

    def update_comp(composition, groups):
        global test_count, NG2, new_group_dict, low_group, in_primary
        NC, compoundA, compoundB = composition[groups[0][0]], composition[groups[0][0]], composition[groups[1][0]]
        swapped = False
        new_name = {}
        for group, count in compoundA[1] + compoundB[1]:
            if group in new_name:
                new_name[group] += count
            else:
                new_name[group] = count
        new_name = [[group, count] for group, count in new_name.items()]
        new_name.sort(key=lambda x: x[0])
        NG, WL, WL_ID = new_group_dict['NG'], new_group_dict['WL'], new_group_dict['WL_ID']
        if len(byproducts) == 0:
            byproducts.append([WL_ID, WL])
        else:
            for i in range(len(byproducts)):
                if byproducts[i][0] == WL_ID:
                    byproducts[i][1] = byproducts[i][1] + WL
                    break
                elif i == len(byproducts) - 1:
                    byproducts.append([WL_ID, WL])
                    break
        NW = compoundA[2][0] + compoundB[2][0] - WL
        NC = [[[group[0], group[1]] for group in NC[0]], new_name, [round(NW, 4)]]
        NC[0][groups[0][1]][0] = NG
        for species in NC[1]:
            name = sn_dict[species[0]]
            if name.cprgID[0] == NG:
                NC[0][groups[0][1]][1] = name.cprgID[1]
                swapped = True
                break
        if not swapped:
            NC[0][groups[0][1]][1] = 0
        old_groups = compoundB[0]
        if len(old_groups) == 1:
            pass
        else:
            del (old_groups[groups[1][1]])
            for sublist in old_groups:
                NC[0].append(sublist)
        if NG2 is not False:
            NG2 = new_group_dict['NG2']
            for species in NC[1]:
                name = sn_dict[species[0]]
                if name.cprgID[0] == NG2:
                    NC[0].append([NG2, name.cprgID[1]])
                    break
                else:
                    NC[0].append([NG2, 0])
        NC[0].sort(key=lambda x: x[0])
        composition[groups[0][0]] = NC
        del (composition[groups[1][0]])
        if __name__ == '__main__':
            window.update()
        if test_count >= test_interval:
            if in_primary:
                RXN_Status(composition, byproducts)
            else:
                RXN_Status_sec(composition, byproducts)
            test_count = 0

    def RXN_Status(composition, byproducts):
        global test_interval, in_situ_values, Xn_list, running, in_primary, comp_primary, comp_secondary, byproducts_primary, byproducts_secondary
        comp_secondary, byproducts_secondary = None, None
        comp_summary = collections.Counter([(tuple(tuple(i) for i in sublist[0]), tuple(tuple(i) for i in sublist[1]), sublist[2][0]) for sublist in composition])
        sum_comp = sum([comp_summary[key] * key[2] for key in comp_summary])
        sum_comp_2 = sum([comp_summary[key] * key[2]**2 for key in comp_summary])
        total_ct_temp = 0
        for key in comp_summary:
            total_ct_temp += comp_summary[key]
        Mw = sum_comp_2 / sum_comp
        Mn = sum_comp / total_ct_temp
        amine_ct, amine_ct, acid_ct, alcohol_ct, epoxide_ct, EHC_ct, IV_ct = 0, 0, 0, 0, 0, 0, 0
        for key in comp_summary:
            key_names = [i[0] for i in key[0]]
            Cl_ct = key_names.count('Cl')
            for group in key[0]:
                if group[0] == 'NH2' or group[0] == 'NH' or group[0] == 'N':
                    amine_ct += comp_summary[key]
                elif group[0] == 'COOH':
                    acid_ct += comp_summary[key]
                elif group[0] == 'POH' or group[0] == 'SOH':
                    alcohol_ct += comp_summary[key]
                    if 'Epi' in sn_dict:
                        sOHk = sn_dict['Epi'].cprgID[1]
                        if group[0] == 'SOH' and group[1] == sOHk and Cl_ct > 0:
                            Cl_ct -= 1
                            EHC_ct += comp_summary[key]
                elif group[0] == 'COC':
                    epoxide_ct += comp_summary[key]
                elif group[0] == 'aB_unsat' or group[0] == 'CC_3' or group[0] == 'CC_2' or group[0] == 'CC_1':
                    IV_ct += comp_summary[key]
        TAV = round((amine_ct * 56100) / sum_comp, 2)  # Calculate in-situ variables
        AV = round((acid_ct * 56100) / sum_comp, 2)
        OH = round((alcohol_ct * 56100) / sum_comp, 2)
        COC = round((epoxide_ct * 56100) / sum_comp, 2)
        EHC = round((EHC_ct * 35.453) / sum_comp * 100, 2)
        IV = round(((IV_ct * 2) * (127 / sum_comp) * 100), 2)
        variables = [TAV, AV, OH, COC, EHC, IV, Mw, Mn]  # in-situ variables added to list
        for i, variable in enumerate(variables):
            in_situ_values[i].append(variable)
        Xn_list.append(round((total_ct / workers) / total_ct_temp, 4))
        metrics = {'Amine Value': TAV, 'Acid Value': AV, 'OH Value': OH, 'Epoxide Value': COC, '% EHC': EHC, 'Iodine Value': IV}
        RXN_metric_value = metrics[end_metric_selection]
        if end_metric_value_upper >= RXN_metric_value >= end_metric_value_lower:
            test_interval = 1
        if end_metric_selection != '% EHC':
            if currentPID == PID_list[-1]:
                process_queue.put(round(((end_metric_value / RXN_metric_value) * 100), 2))
            if RXN_metric_value <= end_metric_value:
                comp_primary = tuple(composition)
                byproducts_primary = byproducts
                if RXN_EM_2_Active_status == False:
                    running = False
                else:
                    in_primary = False
                    for species in composition_sec:
                        composition.append(species)
                    test_interval = 40
        else:
            if currentPID == PID_list[-1]:
                process_queue.put(round(((EHC / end_metric_value) * 100), 2))
            if RXN_metric_value >= end_metric_value:
                comp_primary = tuple(composition)
                byproducts_primary = byproducts
                if RXN_EM_2_Active_status == False:
                    running = False
                else:
                    in_primary = False
                    for species in composition_sec:
                        composition.append(species)
                    test_interval = 40

    def RXN_Status_sec(composition, byproducts):
        global test_interval, in_situ_values_sec, Xn_list_sec, running, comp_secondary, byproducts_secondary
        comp_summary_2 = collections.Counter([(tuple(tuple(i) for i in sublist[0]), tuple(tuple(i) for i in sublist[1]), sublist[2][0]) for sublist in composition])
        sum_comp_2 = sum([comp_summary_2[key] * key[2] for key in comp_summary_2])
        sum_comp_2_2 = sum([comp_summary_2[key] * key[2] ** 2 for key in comp_summary_2])
        total_ct_temp_2 = 0
        for key in comp_summary_2:
            total_ct_temp_2 += comp_summary_2[key]
        Mw = sum_comp_2_2 / sum_comp_2
        Mn = sum_comp_2 / total_ct_temp_2
        amine_ct, amine_ct, acid_ct, alcohol_ct, epoxide_ct, EHC_ct, IV_ct = 0, 0, 0, 0, 0, 0, 0
        for key in comp_summary_2:
            key_names = [i[0] for i in key[0]]
            Cl_ct = key_names.count('Cl')
            for group in key[0]:
                if group[0] == 'NH2' or group[0] == 'NH' or group[0] == 'N':
                    amine_ct += comp_summary_2[key]
                elif group[0] == 'COOH':
                    acid_ct += comp_summary_2[key]
                elif group[0] == 'POH' or group[0] == 'SOH':
                    alcohol_ct += comp_summary_2[key]
                    if 'Epi' in sn_dict:
                        sOHk = sn_dict['Epi'].cprgID[1]
                        if group[0] == 'SOH' and group[1] == sOHk and Cl_ct > 0:
                            Cl_ct -= 1
                            EHC_ct += comp_summary_2[key]
                elif group[0] == 'COC':
                    epoxide_ct += comp_summary_2[key]
                elif group[0] == 'aB_unsat' or group[0] == 'CC_3' or group[0] == 'CC_2' or group[0] == 'CC_1':
                    IV_ct += comp_summary_2[key]
        TAV = round((amine_ct * 56100) / sum_comp_2, 2)
        AV = round((acid_ct * 56100) / sum_comp_2, 2)
        OH = round((alcohol_ct * 56100) / sum_comp_2, 2)
        COC = round((epoxide_ct * 56100) / sum_comp_2, 2)
        EHC = round((EHC_ct * 35.453) / sum_comp_2 * 100, 2)
        IV = round(((IV_ct * 2) * (127 / sum_comp_2) * 100), 2)
        variables = [TAV, AV, OH, COC, EHC, IV, Mw, Mn]
        for i, variable in enumerate(variables):
            in_situ_values_sec[i].append(variable)
        Xn_list_sec.append(round((total_ct_sec / workers) / total_ct_temp_2, 4))
        metrics = {'Amine Value': TAV, 'Acid Value': AV, 'OH Value': OH, 'Epoxide Value': COC, '% EHC': EHC,
                   'Iodine Value': IV}
        RXN_metric_value_2 = metrics[end_metric_selection_sec]
        if end_metric_value_upper_sec >= RXN_metric_value_2 >= end_metric_value_lower_sec:
            test_interval = 1
        if end_metric_selection_sec != '% EHC':
            if currentPID == PID_list[-1]:
                progress_queue_sec.put(round((end_metric_value_sec / RXN_metric_value_2) * 100), 2)
            if RXN_metric_value_2 <= end_metric_value_sec:
                comp_secondary = composition
                byproducts_secondary = byproducts
                running = False
        else:
            if currentPID == PID_list[-1]:
                progress_queue_sec.put(round(((EHC / end_metric_value_sec) * 100), 2))
            if RXN_metric_value_2 >= end_metric_value_sec:
                comp_secondary = composition
                byproducts_secondary = byproducts
                running = False

    while running:
        test_count += 1
        weights = []
        chemical = []
        for i, chemicals in enumerate(composition):
            index = 0
            for group in chemicals[0]:
                chemical.append([i, index, group[0]])
                weights.append(group[1])
                index += 1
        groups = random.choices(chemical, weights, k=2)
        start = time.time()
        while groups[0][0] == groups[1][0] or check_react(groups) is False:
            groups = random.choices(chemical, weights, k=2)
            if time.time() - start > 3:
                running = False
                return "Metric Error"
        stop = time.time()
        update_comp(composition, groups)
    if comp_secondary is None:
        return {"comp_primary": comp_primary, "in_situ_values": in_situ_values, "Xn_list": Xn_list, 'byproducts_primary': byproducts_primary}
    else:
        return {"comp_primary": comp_primary, "comp_secondary": comp_secondary, "in_situ_values": in_situ_values, 'in_situ_values_sec': in_situ_values_sec, "Xn_list": Xn_list, "Xn_list_sec": Xn_list_sec, 'byproducts_primary': byproducts_primary, 'byproducts_secondary': byproducts_secondary}


def update_metrics(TAV, AV, OH, EHC, COC, IV):
    RM.entries[8].delete(0, tkinter.END)
    RM.entries[8].insert(0, EHC)
    RM.entries[9].delete(0, tkinter.END)
    try:
        RM.entries[9].insert(0, round((3545.3 / EHC) - 36.4, 2))
    except ZeroDivisionError:
        RM.entries[9].insert(0, 'N/A')
    RM.entries[10].delete(0, tkinter.END)
    RM.entries[10].insert(0, AV)
    RM.entries[11].delete(0, tkinter.END)
    RM.entries[11].insert(0, TAV)
    RM.entries[12].delete(0, tkinter.END)
    RM.entries[12].insert(0, OH)
    RM.entries[13].delete(0, tkinter.END)
    RM.entries[13].insert(0, COC)
    RM.entries[14].delete(0, tkinter.END)
    RM.entries[14].insert(0, IV)


def update_metrics_sec(TAV, AV, OH, EHC, COC, IV):
    RM2.entries[8].delete(0, tkinter.END)
    RM2.entries[8].insert(0, EHC)
    RM2.entries[9].delete(0, tkinter.END)
    try:
        RM2.entries[9].insert(0, round((3545.3 / EHC) - 36.4, 2))
    except ZeroDivisionError:
        RM2.entries[9].insert(0, 'N/A')
    RM2.entries[10].delete(0, tkinter.END)
    RM2.entries[10].insert(0, AV)
    RM2.entries[11].delete(0, tkinter.END)
    RM2.entries[11].insert(0, TAV)
    RM2.entries[12].delete(0, tkinter.END)
    RM2.entries[12].insert(0, OH)
    RM2.entries[13].delete(0, tkinter.END)
    RM2.entries[13].insert(0, COC)
    RM2.entries[14].delete(0, tkinter.END)
    RM2.entries[14].insert(0, IV)

def RXN_Results(primary_comp_summary, byproducts_primary, in_situ_values, Xn_list):
    global rxn_summary_df, Xn, total_samples, byproducts_df
    comp_summary = collections.Counter([(tuple(tuple(i) for i in sublist[0]), tuple(tuple(i) for i in sublist[1]), sublist[2][0]) for sublist in primary_comp_summary])
    sum_comp = sum([comp_summary[key] * key[2] for key in comp_summary])
    RS = []
    for key in comp_summary:
        RS.append((key[0], key[1], key[2], comp_summary[key]))
    RS = [[[list(x) for x in i[0]], [list(y) for y in i[1]], i[2], i[3]] for i in RS]
    for key in RS:
        amine_ct, pTAV_ct, sTAV_ct, tTAV_ct, acid_ct, alcohol_ct, epoxide_ct, EHC_ct, IV_ct = 0, 0, 0, 0, 0, 0, 0, 0, 0
        key_names = [i[0] for i in key[0]]
        Cl_ct = key_names.count('Cl')
        for group in key[0]:
            if group[0] == 'NH2' or group[0] == 'NH' or group[0] == 'N':
                amine_ct += key[3]
                if group[0] == 'NH2':
                    pTAV_ct += key[3]
                elif group[0] == 'NH':
                    sTAV_ct += key[3]
                elif group[0] == 'N':
                    tTAV_ct += key[3]
            elif group[0] == 'COOH':
                acid_ct += key[3]
            elif group[0] == 'POH' or group[0] == 'SOH':
                alcohol_ct += key[3]
                if 'Epi' in sn_dict:
                    sOHk = sn_dict['Epi'].cprgID[1]
                    if group[0] == 'SOH' and group[1] == sOHk and Cl_ct > 0:
                        Cl_ct -= 1
                        EHC_ct += key[3]
            elif group[0] == 'COC':
                epoxide_ct += key[3]
            # elif group[0] == 'Cl' and 'SOH' in key_names:
            # EHC_ct += key[3]
            elif group[0] == 'aB_unsat' or group[0] == 'CC_3' or group[0] == 'CC_2' or group[0] == 'CC_1':
                IV_ct += key[3]
        key.append(round((amine_ct * 56100) / sum_comp, 2))
        key.append(round((pTAV_ct * 56100) / sum_comp, 2))
        key.append(round((sTAV_ct * 56100) / sum_comp, 2))
        key.append(round((tTAV_ct * 56100) / sum_comp, 2))
        key.append(round((acid_ct * 56100) / sum_comp, 2))
        key.append(round((alcohol_ct * 56100) / sum_comp, 2))
        key.append(round((epoxide_ct * 56100) / sum_comp, 2))
        key.append(round((EHC_ct * 35.453) / sum_comp * 100, 2))
        key.append(round(((IV_ct * 2) * (127 / sum_comp) * 100), 2))
    for key in RS:
        index = 0
        for group in key[1]:
            new_name = group[0] + '(' + str(group[1]) + ')'
            key[1][index] = new_name
            index += 1
        key[1] = '_'.join(key[1])
    rxn_summary_df = pandas.DataFrame(RS, columns=['Groups', 'Name', 'MW', 'Count', 'TAV', "1° TAV", "2° TAV", "3° TAV",
                                                   'AV', 'OH', 'COC', 'EHC,%', 'IV'])
    rxn_summary_df['MW'] = round(rxn_summary_df['MW'], 4)
    rxn_summary_df.drop(columns=['Groups'], inplace=True)
    rxn_summary_df.set_index('Name', inplace=True)
    rxn_summary_df.sort_values(by=['MW'], ascending=True, inplace=True)
    rxn_summary_df['Mass'] = rxn_summary_df['MW'] * rxn_summary_df['Count']
    rxn_summary_df['Mol,%'] = round(rxn_summary_df['Count'] / rxn_summary_df['Count'].sum() * 100, 4)
    rxn_summary_df['Wt,%'] = round(rxn_summary_df['Mass'] / rxn_summary_df['Mass'].sum() * 100, 4)
    sumNi = rxn_summary_df['Count'].sum()
    sumNiMi = (rxn_summary_df['Count'] * rxn_summary_df['MW']).sum()
    sumNiMi2 = (rxn_summary_df['Count'] * (rxn_summary_df['MW']) ** 2).sum()
    sumNiMi3 = (rxn_summary_df['Count'] * (rxn_summary_df['MW']) ** 3).sum()
    sumNiMi4 = (rxn_summary_df['Count'] * (rxn_summary_df['MW']) ** 4).sum()
    rxn_summary_df = rxn_summary_df[['Count', 'Mass', 'Mol,%', 'Wt,%', 'MW', 'TAV', '1° TAV', '2° TAV', '3° TAV', 'AV', 'OH', 'COC', 'EHC,%', 'IV']]
    rxn_summary_df['Mass'] = rxn_summary_df['Mass'] / total_samples
    rxn_summary_df.loc['Sum'] = round(rxn_summary_df.sum(), 3)
    rxn_summary_df = rxn_summary_df.groupby(['MW', 'Name']).sum()


    Mn = sumNiMi / sumNi
    Mw = sumNiMi2 / sumNiMi
    PDI = Mw / Mn
    Mz = sumNiMi3 / sumNiMi2
    Mz1 = sumNiMi4 / sumNiMi3

    WD.entries[6].delete(0, tkinter.END)
    WD.entries[6].insert(0, round(Mn, 4))
    WD.entries[7].delete(0, tkinter.END)
    WD.entries[7].insert(0, round(Mw, 4))
    WD.entries[8].delete(0, tkinter.END)
    WD.entries[8].insert(0, round(PDI, 4))
    WD.entries[9].delete(0, tkinter.END)
    WD.entries[9].insert(0, round(Mz, 4))
    WD.entries[10].delete(0, tkinter.END)
    WD.entries[10].insert(0, round(Mz1, 4))
    WD.entries[11].delete(0, tkinter.END)
    WD.entries[11].insert(0, round(total_ct / sumNi, 4))

    byproducts_df = pandas.DataFrame(byproducts_primary, columns=['Name', 'Mass'])
    byproducts_df.set_index('Name', inplace=True)
    byproducts_df['Mass'] = byproducts_df['Mass'] / total_samples
    byproducts_df['Wt, % (Of byproducts)'] = round(byproducts_df['Mass'] / byproducts_df['Mass'].sum() * 100, 4)
    byproducts_df['Wt, % (Of Final)'] = round(byproducts_df['Mass'] / rxn_summary_df['Mass'].sum() * 100, 4)
    byproducts_df['Wt, % (Of Initial)'] = round(byproducts_df['Mass'] / (starting_mass /total_samples) * 100, 4)

    Xn = pandas.DataFrame(in_situ_values[0], columns=['TAV'])
    Xn['AV'] = in_situ_values[1]
    Xn['OH'] = in_situ_values[2]
    Xn['COC'] = in_situ_values[3]
    Xn['EHC, %'] = in_situ_values[4]
    Xn['IV'] = in_situ_values[5]
    Xn['Xn'] = Xn_list
    Xn['P'] = -(1 / Xn['Xn']) + 1
    Xn['Mw'] = in_situ_values[6]
    Xn['Mn'] = in_situ_values[7]

    show_results(rxn_summary_df)
    show_byproducts(byproducts_df)
    show_Xn(Xn)

def RXN_Results_sec(secondary_comp_summary, byproducts_secondary, in_situ_values_sec, Xn_list_sec):
    global rxn_summary_df_2, Xn_2, total_samples, byproducts_df_2
    comp_summary_2 = collections.Counter(
        [(tuple(tuple(i) for i in sublist[0]), tuple(tuple(i) for i in sublist[1]), sublist[2][0]) for sublist in
         secondary_comp_summary])
    sum_comp_2 = sum([comp_summary_2[key] * key[2] for key in comp_summary_2])
    RS_2 = []
    for key in comp_summary_2:
        RS_2.append((key[0], key[1], key[2], comp_summary_2[key]))
    RS_2 = [[[list(x) for x in i[0]], [list(y) for y in i[1]], i[2], i[3]] for i in RS_2]
    for key in RS_2:
        amine_ct, pTAV_ct, sTAV_ct, tTAV_ct, acid_ct, alcohol_ct, epoxide_ct, EHC_ct, IV_ct = 0, 0, 0, 0, 0, 0, 0, 0, 0
        key_names = [i[0] for i in key[0]]
        Cl_ct = key_names.count('Cl')
        for group in key[0]:
            if group[0] == 'NH2' or group[0] == 'NH' or group[0] == 'N':
                amine_ct += key[3]
                if group[0] == 'NH2':
                    pTAV_ct += key[3]
                elif group[0] == 'NH':
                    sTAV_ct += key[3]
                elif group[0] == 'N':
                    tTAV_ct += key[3]
            elif group[0] == 'COOH':
                acid_ct += key[3]
            elif group[0] == 'POH' or group[0] == 'SOH':
                alcohol_ct += key[3]
                if 'Epi' in sn_dict:
                    sOHk = sn_dict['Epi'].cprgID[1]
                    if group[0] == 'SOH' and group[1] == sOHk and Cl_ct > 0:
                        Cl_ct -= 1
                        EHC_ct += key[3]
            elif group[0] == 'COC':
                epoxide_ct += key[3]
            # elif group[0] == 'Cl' and 'SOH' in key_names:
            # EHC_ct += key[3]
            elif group[0] == 'aB_unsat' or group[0] == 'CC_3' or group[0] == 'CC_2' or group[0] == 'CC_1':
                IV_ct += key[3]
        key.append(round((amine_ct * 56100) / sum_comp_2, 2))
        key.append(round((pTAV_ct * 56100) / sum_comp_2, 2))
        key.append(round((sTAV_ct * 56100) / sum_comp_2, 2))
        key.append(round((tTAV_ct * 56100) / sum_comp_2, 2))
        key.append(round((acid_ct * 56100) / sum_comp_2, 2))
        key.append(round((alcohol_ct * 56100) / sum_comp_2, 2))
        key.append(round((epoxide_ct * 56100) / sum_comp_2, 2))
        key.append(round((EHC_ct * 35.453) / sum_comp_2 * 100, 2))
        key.append(round(((IV_ct * 2) * (127 / sum_comp_2) * 100), 2))
    for key in RS_2:
        index = 0
        for group in key[1]:
            new_name = group[0] + '(' + str(group[1]) + ')'
            key[1][index] = new_name
            index += 1
        key[1] = '_'.join(key[1])
    rxn_summary_df_2 = pandas.DataFrame(RS_2,
                                        columns=['Groups', 'Name', 'MW', 'Count', 'TAV', "1° TAV", "2° TAV", "3° TAV",
                                                 'AV', 'OH', 'COC', 'EHC,%', 'IV'])
    rxn_summary_df_2['MW'] = round(rxn_summary_df_2['MW'], 4)
    rxn_summary_df_2.drop(columns=['Groups'], inplace=True)
    rxn_summary_df_2.set_index('Name', inplace=True)
    rxn_summary_df_2.sort_values(by=['MW'], ascending=True, inplace=True)
    rxn_summary_df_2['Mass'] = rxn_summary_df_2['MW'] * rxn_summary_df_2['Count']
    rxn_summary_df_2['Mol,%'] = round(rxn_summary_df_2['Count'] / rxn_summary_df_2['Count'].sum() * 100, 4)
    rxn_summary_df_2['Wt,%'] = round(rxn_summary_df_2['Mass'] / rxn_summary_df_2['Mass'].sum() * 100, 4)
    sumNi = rxn_summary_df_2['Count'].sum()
    sumNiMi = (rxn_summary_df_2['Count'] * rxn_summary_df_2['MW']).sum()
    sumNiMi2 = (rxn_summary_df_2['Count'] * (rxn_summary_df_2['MW']) ** 2).sum()
    sumNiMi3 = (rxn_summary_df_2['Count'] * (rxn_summary_df_2['MW']) ** 3).sum()
    sumNiMi4 = (rxn_summary_df_2['Count'] * (rxn_summary_df_2['MW']) ** 4).sum()
    rxn_summary_df_2 = rxn_summary_df_2[
        ['Count', 'Mass', 'Mol,%', 'Wt,%', 'MW', 'TAV', '1° TAV', '2° TAV', '3° TAV', 'AV', 'OH', 'COC', 'EHC,%', 'IV']]
    rxn_summary_df_2['Mass'] = rxn_summary_df_2['Mass'] / total_samples
    rxn_summary_df_2.loc['Sum'] = round(rxn_summary_df_2.sum(), 3)
    rxn_summary_df_2 = rxn_summary_df_2.groupby(['MW', 'Name']).sum()
    Mn = sumNiMi / sumNi
    Mw = sumNiMi2 / sumNiMi
    PDI = Mw / Mn
    Mz = sumNiMi3 / sumNiMi2
    Mz1 = sumNiMi4 / sumNiMi3
    WD2.entries[6].delete(0, tkinter.END)
    WD2.entries[6].insert(0, round(Mn, 4))
    WD2.entries[7].delete(0, tkinter.END)
    WD2.entries[7].insert(0, round(Mw, 4))
    WD2.entries[8].delete(0, tkinter.END)
    WD2.entries[8].insert(0, round(PDI, 4))
    WD2.entries[9].delete(0, tkinter.END)
    WD2.entries[9].insert(0, round(Mz, 4))
    WD2.entries[10].delete(0, tkinter.END)
    WD2.entries[10].insert(0, round(Mz1, 4))
    WD2.entries[11].delete(0, tkinter.END)
    WD2.entries[11].insert(0, round(total_ct_sec / sumNi, 4))

    byproducts_df_2 = pandas.DataFrame(byproducts_secondary, columns=['Name', 'Mass'])
    byproducts_df_2.set_index('Name', inplace=True)
    byproducts_df_2['Mass'] = byproducts_df_2['Mass'] / total_samples
    byproducts_df_2['Wt, % (Of byproducts)'] = round(byproducts_df_2['Mass'] / byproducts_df_2['Mass'].sum() * 100, 4)
    byproducts_df_2['Wt, % (Of Final)'] = round(byproducts_df_2['Mass'] / rxn_summary_df_2['Mass'].sum() * 100, 4)
    byproducts_df_2['Wt, % (Of Initial)'] = round(byproducts_df_2['Mass'] / (starting_mass / total_samples) * 100, 4)

    Xn_2 = pandas.DataFrame(in_situ_values_sec[0], columns=['TAV'])
    Xn_2['AV'] = in_situ_values_sec[1]
    Xn_2['OH'] = in_situ_values_sec[2]
    Xn_2['COC'] = in_situ_values_sec[3]
    Xn_2['EHC, %'] = in_situ_values_sec[4]
    Xn_2['IV'] = in_situ_values_sec[5]
    Xn_2['Xn'] = Xn_list_sec
    Xn_2['P'] = -(1 / Xn_2['Xn']) + 1
    Xn_2['Mw'] = in_situ_values_sec[6]
    Xn_2['Mn'] = in_situ_values_sec[7]

    show_results_sec(rxn_summary_df_2)
    show_byproducts_sec(byproducts_df_2)
    show_Xn_sec(Xn_2)

# -------------------------------------------Display Results---------------------------------------------------#

def show_results(rxn_summary_df):
    global results_table, frame_results
    try:
        results_table.destroy()
        frame_results.destroy()
    except NameError:
        pass
    frame_results = tkinter.Frame(tab2)
    x = ((window.winfo_screenwidth() - frame_results.winfo_reqwidth()) / 2) + 100
    y = (window.winfo_screenheight() - frame_results.winfo_reqheight()) / 2
    frame_results.place(x=x, y=y, anchor='center')
    results_table = Table(frame_results, dataframe=rxn_summary_df, showtoolbar=True, showstatusbar=True, showindex=True,
                          width=x, height=y, align='center', )
    # adjust results_table if it is too big
    if results_table.winfo_reqwidth() > window.winfo_screenwidth():
        results_table.width = window.winfo_screenwidth() - 100
    if results_table.winfo_reqheight() > window.winfo_screenheight():
        results_table.height = window.winfo_screenheight() - 100
    results_table.show()

def show_results_sec(rxn_summary_df_2):
    global results_table_2, frame_results_2
    try:
        results_table_2.destroy()
        frame_results_2.destroy()
    except NameError:
        pass
    frame_results_2 = tkinter.Frame(tab4)
    x = ((window.winfo_screenwidth() - frame_results_2.winfo_reqwidth()) / 2) + 100
    y = (window.winfo_screenheight() - frame_results_2.winfo_reqheight()) / 2
    frame_results_2.place(x=x, y=y, anchor='center')
    results_table_2 = Table(frame_results_2, dataframe=rxn_summary_df_2, showtoolbar=True, showstatusbar=True,
                            showindex=True,
                            width=x, height=y, align='center', )
    # adjust results_table if it is too big
    if results_table_2.winfo_reqwidth() > window.winfo_screenwidth():
        results_table_2.width = window.winfo_screenwidth() - 100
    if results_table_2.winfo_reqheight() > window.winfo_screenheight():
        results_table_2.height = window.winfo_screenheight() - 100
    results_table_2.show()

def show_byproducts(byproducts_df):
    global byproducts_table, frame_byproducts
    try:
        byproducts_table.destroy()
        frame_byproducts.destroy()
    except NameError:
        pass
    frame_byproducts = tkinter.Frame(tab2)
    frame_byproducts.grid(row=0, column=3, padx=(100, 0), pady=(20, 0))
    byproducts_table = Table(frame_byproducts, dataframe=byproducts_df, showtoolbar=False, showstatusbar=True,
                             showindex=True, width=600, height=100, align='center', maxcellwidth=1000)
    byproducts_table.show()

def show_byproducts_sec(byproducts_df_2):
    global byproducts_table_2, frame_byproducts_2
    try:
        byproducts_table_2.destroy()
        frame_byproducts_2.destroy()
    except NameError:
        pass
    frame_byproducts_2 = tkinter.Frame(tab4)
    frame_byproducts_2.grid(row=0, column=3, padx=(100, 0), pady=(20, 0))
    byproducts_table_2 = Table(frame_byproducts_2, dataframe=byproducts_df_2, showtoolbar=False, showstatusbar=True,
                               showindex=True, width=600, height=100, align='center', maxcellwidth=1000)
    byproducts_table_2.show()

def show_Xn(Xn):
    global Xn_table, frame_Xn
    try:
        Xn_table.destroy()
        frame_Xn.destroy()
    except NameError:
        pass
    frame_Xn = tkinter.Frame(tab3)
    frame_Xn.pack()
    Xn_table = Table(frame_Xn, dataframe=Xn, showtoolbar=True, showstatusbar=True, showindex=True, width=2000,
                     height=1000, align='center', maxcellwidth=1000)
    Xn_table.show()

def show_Xn_sec(Xn_2):
    global Xn_table_2, frame_Xn_2
    try:
        Xn_table_2.destroy()
        frame_Xn_2.destroy()
    except NameError:
        pass
    frame_Xn_2 = tkinter.Frame(tab5)
    frame_Xn_2.pack()
    Xn_table_2 = Table(frame_Xn_2, dataframe=Xn_2, showtoolbar=True, showstatusbar=True, showindex=True, width=2000,
                       height=1000, align='center', maxcellwidth=1000)
    Xn_table_2.show()

def show_APC(APC_Flow_Rate, APC_FWHM, APC_FWHM2, APC_comp, APC_df, label):
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(APC_df.index, APC_df['Sum'], linewidth=0.75)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Intensity (a.u.)')
    ax.set_title(f'{label} Theoretical APC')
    ax.set_xticks(np.arange(0, 12.25, 0.25))
    ax.set_xticklabels(np.arange(0, 12.25, 0.25), rotation=-35)
    ax.set_xlim(0, 12)

    textstr = f'Flow Rate: {APC_Flow_Rate:.1f} ml/min\nFWHM (100 MW): {APC_FWHM:.3f} min\nFWHM (100K MW): {APC_FWHM2:.3f} min'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    a = ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    time_textstr = f'Time (min): '
    b = ax.text(0.05, 0.75, time_textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    fig.canvas.draw()

    def on_move(event, b):
        xdata = event.xdata
        if xdata is not None:
            closest_peaks = APC_comp.iloc[(APC_comp['RT(min)'] - xdata).abs().argsort()[:5]]
            tolerence = 0.05
            peak_info = []
            for i, peak in closest_peaks.iterrows():
                if abs(peak['RT(min)'] - xdata) <= tolerence:
                    peak_info.append([peak['Name'], f"{peak['RT(min)']:.3f}", f"{10 ** peak['Log(MW)']:.2f}", f"{peak['Wt,%']:.2f}"])
            table.delete(*table.get_children())
            for info in peak_info:
                table.insert("", "end", values=info)
            b.set_text(f'Time (min): {xdata:.3f}')
        else:
            table.delete(*table.get_children())
            b.set_text(f'Time (min): ')
        fig.canvas.draw()

    fig.canvas.mpl_connect('motion_notify_event', lambda event: on_move(event, b))

    root = tkinter.Tk()
    root.title(f'Monte Karlo - {label} Theoretical APC')
    root.iconbitmap("testtube.ico")
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    toolbar.pack()
    columns = ("Name", "Retention Time (min)", "MW(g/mol)", "Wt,%")
    table = ttk.Treeview(root, columns=columns, show="headings", height=5)
    for col in columns:
        table.heading(col, text=col)
    table.pack()
    button_close = tkinter.Button(root, text="Close", command=root.destroy)
    button_close.pack()

    root.mainloop()

# -------------------------------------------Auxiliary Functions---------------------------------------------------#

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

def stop():
    global running, canceled_by_user
    if running:
        canceled_by_user = True
        running = False
    else:
        pass

def clear_last():
    cell = 16
    for i in range(RET.tableheight - 1):
        if RET.entries[16].get() == "":
            tkinter.messagebox.showinfo("Error", "Table is already empty")
            return
        else:
            if RET.entries[cell].get() != "":
                cell = cell + RET.tablewidth
            else:
                clear_index = i - 1
                clear_cell = cell - RET.tablewidth
                break
    RET.entries[clear_cell].delete(0, 'end')
    RET.entries[clear_cell].insert(0, "Clear")
    check_entry(entry=clear_cell, index=clear_index, cell=clear_cell)

def initialize_sim(workers):
    global total_ct, sn_dict, starting_mass, total_ct_sec, starting_mass_sec, end_metric_value, end_metric_value_sec, RXN_EM_2_Active_status, end_metric_selection, end_metric_selection_sec, starting_materials, \
        starting_materials_sec, total_samples
    starting_mass, starting_mass_sec, total_ct, total_ct_sec = 0, 0, 0, 0
    row_for_sec = RXN_EM_Entry_2_SR.current()
    try:
        end_metric_value = float(RXN_EM_Value.get())
        RXN_EM_2_Active_status = RXN_EM_2_Active.get()
        if RXN_EM_2_Active_status:
            end_metric_value_sec = float(RXN_EM_Value_2.get())
        else:
            end_metric_value_sec = 0
    except ValueError:
        messagebox.showinfo("Error", "Please enter a valid number for the end metric value(s)")
        return "Error"
    end_metric_selection, end_metric_selection_sec = str(RXN_EM.get()), str(RXN_EM_Entry_2.get())
    if "Metric" in end_metric_selection:
        messagebox.showinfo("Error", "Please select valid end metric(s)")
        if RXN_EM_2_Active_status and "Metric" in end_metric_selection_sec:
            messagebox.showinfo("Error", "Please select valid end metric(s)")
        return "Error"
    cell = 16
    index = 0
    sn_dict = {}
    starting_materials, starting_materials_sec = [], []
    try:
        for i in range(RET.tableheight - 1):
            if RET.entries[cell].get() != "" and RET.entries[cell + 1].get() != "":
                str_to_class(RDE[index]).assign(name=str_to_class(Entry_Reactants[index].get())(),
                                                mass=Entry_masses[index].get(),
                                                moles=round(float(RET.entries[cell + 3].get()), 6),
                                                prgID=RET.entries[cell + 4].get(), prgk=RET.entries[cell + 5].get(),
                                                cprgID=RET.entries[cell + 6].get(), cprgk=RET.entries[cell + 7].get(),
                                                srgID=RET.entries[cell + 8].get(), srgk=RET.entries[cell + 9].get(),
                                                csrgID=RET.entries[cell + 10].get(), csrgk=RET.entries[cell + 11].get(),
                                                trgID=RET.entries[cell + 12].get(), trgk=RET.entries[cell + 13].get(),
                                                ctrgID=RET.entries[cell + 14].get(), ctrgk=RET.entries[cell + 15].get(),
                                                ct=RXN_Samples.get())
                sn_dict[str_to_class(RDE[index]).sn] = str_to_class(RDE[index])
                count = RXN_Samples.get()
                moles_count = round(float(RET.entries[cell + 3].get()), 6)
                starting_mass_sec += float(float(Entry_masses[index].get()) * float(count))
                cell = cell + RET.tablewidth
                total_ct_sec += (float(count) * float(moles_count))
                if i >= row_for_sec and RXN_EM_2_Active.get() == True:
                    starting_materials_sec.append(str_to_class(RDE[index]).comp)
                else:
                    starting_mass += float(float(Entry_masses[index].get()) * float(count))
                    total_ct += (float(count) * float(moles_count))
                    starting_materials.append(str_to_class(RDE[index]).comp)
                index += 1
            else:
                break
        total_ct *= workers
        total_ct_sec *= workers
        starting_mass *= workers
        starting_mass_sec *= workers
        total_samples = int(RXN_Samples.get()) * workers
    except AttributeError as e:
        messagebox.showerror("Exception raised", str(e))
        pass

def multiprocessing_sim():
    if __name__ == "__main__":
        Buttons.Simulate.config(state="disabled", text="Running...")
        sim.progress['value'], sim.progress_2['value'] = 0, 0
        global running 
        running = True
        workers = NUM_OF_SIM.get()
        workers = int(workers)
        if initialize_sim(workers) == "Error":
            Buttons.Simulate.config(text="Simulate", state="normal")
            return
        progress_queue = multiprocessing.Manager().Queue()
        progress_queue_sec = multiprocessing.Manager().Queue()
        PID_list = multiprocessing.Manager().list()
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            results = [executor.submit(simulate, starting_materials, starting_materials_sec, end_metric_value, end_metric_selection, end_metric_value_sec, end_metric_selection_sec, sn_dict, RXN_EM_2_Active_status, total_ct, total_ct_sec, workers, progress_queue, PID_list, progress_queue_sec) for _ in range(workers)]
            while len(PID_list) < workers:
                pass
            while any(result.running() for result in results) and running is True:
                try:
                    progress = progress_queue.get_nowait()
                    sim.progress['value'] = progress
                except queue.Empty:
                    pass
                try:
                    progress_sec = progress_queue_sec.get_nowait()
                    sim.progress_2['value'] = progress_sec
                except queue.Empty:
                    pass
                window.update()
            if running is False and canceled_by_user is True:
                for result in results:
                    result.cancel()
                    result.result()
                    window.update()
                while not progress_queue.empty():
                    progress_queue.get()
                sim.progress['value'], sim.progress_2['value'] = 0, 0
                messagebox.showinfo("Simulation Cancelled", "Simulation cancelled by user")
                Buttons.Simulate.config(text="Simulate", state="normal")
                return
            concurrent.futures.wait(results)
            if results[0].result() == "Metric Error" or results[0].result() == "Zero Error":
                messagebox.showinfo("Error", "Please select valid end metric(s)")
                Buttons.Simulate.config(text="Simulate", state="normal")
                return
            consolidate_results(results)
            Buttons.Simulate.config(text="Simulate", state="normal")

def consolidate_results(results):
    primary_comp_summary, secondary_comp_summary = [], []
    byproducts_primary, byproducts_secondary = [], []
    in_situ_values = results[0].result()['in_situ_values']
    Xn_list = results[0].result()['Xn_list']
    if 'comp_secondary' in results[0].result():
        in_situ_values_sec = results[0].result()['in_situ_values_sec']
        Xn_list_sec = results[0].result()['Xn_list_sec']
        for _ in range(len(results)):
            for species in results[_].result()['comp_primary']:
                primary_comp_summary.append(species)
            for species in results[_].result()['comp_secondary']:
                secondary_comp_summary.append(species)
            for identifier, value in results[_].result()['byproducts_primary']:
                for item in byproducts_primary:
                    if item[0] == identifier:
                        item[1] += value
                        break
                else:
                    byproducts_primary.append([identifier, value])
            for identifier, value in results[_].result()['byproducts_secondary']:
                for item in byproducts_secondary:
                    if item[0] == identifier:
                        item[1] += value
                        break
                else:
                    byproducts_secondary.append([identifier, value])
        RXN_Results(primary_comp_summary, byproducts_primary, in_situ_values, Xn_list)
        RXN_Results_sec(secondary_comp_summary, byproducts_secondary, in_situ_values_sec, Xn_list_sec)
    else:
        for _ in range(len(results)):
            for species in results[_].result()['comp_primary']:
                primary_comp_summary.append(species)
            for identifier, value in results[_].result()['byproducts_primary']:
                for item in byproducts_primary:
                    if item[0] == identifier:
                        item[1] += value
                        break
                else:
                    byproducts_primary.append([identifier, value])
        RXN_Results(primary_comp_summary, byproducts_primary, in_situ_values, Xn_list)

def reset_entry_table():
    for i in range(RET.tableheight - 1):
        for j in range(RET.tablewidth):
            RET.entries[(i + 1) * RET.tablewidth + j].configure(state='normal')
            RET.entries[(i + 1) * RET.tablewidth + j].delete(0, 'end')
    sim.progress['value'] = 0
    sim.progress_2['value'] = 0
    for i in range(8, 15):
        RM.entries[i].delete(0, 'end')
        RM2.entries[i].delete(0, 'end')
    RXN_EM_2_Active.set(False)
    RXN_EM_Entry_2_SR.set("2° Start")
    RXN_EM_Value_2.delete(0, 'end')
    RXN_EM_Entry_2.insert(0, "")
    RXN_EM_Entry_2.set("2° End Metric")
    RXN_EM_Entry.set("1° End Metric")
    RXN_EM_Value.delete(0, 'end')
    RXN_EM_Entry.insert(0, "")

def check_entry(entry, index, cell):
    RET.entries[entry].get()
    if RET.entries[entry].get() not in Reactants and RET.entries[entry].get() != "":
        RET.entries[entry].delete(0, 'end')
        messagebox.showerror("Error", "Please enter a valid reactant")
    else:
        RET.update_table(index, cell)
        RET.update_rates(index, cell)

def quick_add():
    cell = 16
    for i in range(RET.tableheight - 1):
        if RET.entries[cell].get() != "" and RET.entries[cell + 1].get() != "":
            cell = cell + RET.tablewidth
        else:
            start_index = i
            break
    d = QuickAddWindow(window)
    if quick_add_comp is None:
        pass
    else:
        for i in range(len(quick_add_dict[quick_add_comp[0]])):
            RET.entries[cell].delete(0, 'end')
            RET.entries[cell + 1].delete(0, 'end')
            RET.entries[cell].insert(0, quick_add_dict[quick_add_comp[0]][i][0])
            new_mass = round(float(quick_add_dict[quick_add_comp[0]][i][1]) * float(quick_add_comp[1]), 4)
            RET.entries[cell + 1].insert(0, new_mass)
            check_entry(entry=cell, index=i + start_index, cell=cell)
            cell += RET.tablewidth

#-------------------------------------------------APC Functions----------------------------------------------------------#
def run_APC(rxn_summary_df, label):
    APCParametersWindow(window)
    if apc_params[0] != "" and apc_params[1] != "" and apc_params[2] != "" and apc_params[3] != "":
        APC_Flow_Rate = apc_params[0]
        APC_FWHM = apc_params[1]
        APC_FWHM2 = apc_params[2]
        APC_temp = apc_params[3]
        APC(APC_Flow_Rate, APC_FWHM, APC_FWHM2, APC_temp, rxn_summary_df, label)

    else:
        messagebox.showerror("Error", "Please enter valid parameters for flow rate, FWHM, and temp.")

def APC(APC_Flow_Rate, APC_FWHM, APC_FWHM2, APC_temp, rxn_summary_df, label):
    global APC_df, apc_params
    APC_comp = rxn_summary_df
    APC_comp = APC_comp.reset_index()
    APC_comp = APC_comp[['MW', 'Wt,%', 'Name']]
    APC_comp = APC_comp[:-1]
    if str(APC_temp) == "35.0":
        STD_Equation_params = np.array([0.0236, -0.6399, 6.5554 , -31.7505, 71.8922, -56.3224])
        Low_MW_Equation_params = np.array([-1.4443, 13.5759, -48.4406, 76.9940, -40.3772])
        High_MW_Equation_params = np.array([0.1606, -1.8779, 7.9957])
    elif str(APC_temp) == "55.0":
        STD_Equation_params = np.array([0.0264, -0.7016, 7.0829, -33.9565, 76.4028, -59.9091])
        Low_MW_Equation_params = np.array([-1.9097, 18.0005, -64.1695, 101.7605, -54.9071])
        High_MW_Equation_params = np.array([0.1621, -1.8972, 8.0633])
    APC_comp['Log(MW)'] = np.log10(APC_comp['MW'])
    APC_comp.loc[:, 'FWHM(min)'] = 0.000
    APC_comp.loc[:, 'RT(min)'] = 0.000
    MIN_MW = np.log10(621)
    MAX_MW = np.log10(290000)

    FWHM_slope = (APC_FWHM2 - APC_FWHM) / 99900
    FWHM_yint = APC_FWHM - (APC_FWHM2 - APC_FWHM) / 99900 * 100

    for i, row in APC_comp.iterrows():
        APC_comp.loc[i, 'FWHM(min)'] = FWHM_slope * row['Log(MW)'] + FWHM_yint
        if row['Log(MW)'] < MIN_MW:
            APC_comp.loc[i, 'RT(min)'] = np.polyval(Low_MW_Equation_params, row['Log(MW)']) / APC_Flow_Rate
        elif row['Log(MW)'] > MAX_MW:
            APC_comp.loc[i, 'RT(min)'] = np.polyval(High_MW_Equation_params, row['Log(MW)']) / APC_Flow_Rate
        else:
            APC_comp.loc[i, 'RT(min)'] = np.polyval(STD_Equation_params, row['Log(MW)']) / APC_Flow_Rate

    APC_columns = []
    for i in range(len(APC_comp)):
        APC_columns.append(APC_comp['Name'][i])
    APC_rows = list(range(0, 120001))
    APC_df = pandas.DataFrame(0, index=APC_rows, columns=APC_columns)
    APC_df.index.name = 'Time'
    APC_df.index = APC_df.index * 0.001

    def calc_apc_matrix(APC_comp, APC_df):
        times = APC_df.index.values
        apc_values = np.zeros((len(times), len(APC_comp)))
        for i, row in APC_comp.iterrows():
            apc_values[:, i] = np.exp(-0.5 * np.power((times - row['RT(min)']) / (row['FWHM(min)'] / 2.35), 2)) * (row['Wt,%'] * 0.5)
        return pandas.DataFrame(apc_values, index=times, columns=APC_comp['Name'])

    APC_df = calc_apc_matrix(APC_comp, APC_df)

    APC_df['Sum'] = APC_df.sum(axis=1)
    show_APC(APC_Flow_Rate, APC_FWHM, APC_FWHM2, APC_comp, APC_df, label)

# -------------------------------------------------Export Data Functions-------------------------------------------------#
def export_primary():
    filepath = filedialog.asksaveasfilename(defaultextension='.xlsx',
                                            filetypes=[("Excel xlsx", "*.xlsx"), ("Excel csv", "*.csv")])
    if filepath == "":
        return
    else:
        with pandas.ExcelWriter(filepath) as writer:
            rxn_summary_df.to_excel(writer, sheet_name='1_Summary', index=True)
            byproducts_df.to_excel(writer, sheet_name='1_Summary', index=True, startrow=0, startcol=18)
            Xn.to_excel(writer, sheet_name='1_In_Situ', index=True)
            data = []
            for row in range(RET.tableheight):
                row_data = []
                for column in range(RET.tablewidth):
                    entry = RET.entries[row * RET.tablewidth + column]
                    row_data.append(entry.get())
                data.append(row_data)
            df = pandas.DataFrame(data)
            df.to_excel(writer, sheet_name='1_Aux', index=False, header=False)

            data = []
            for column in range(WD.tablewidth):
                column_data = []
                for row in range(WD.tableheight):
                    entry = WD.entries[row + column * WD.tableheight]
                    column_data.append(entry.get())
                data.append(column_data)
            data = list(map(list, zip(*data)))
            df = pandas.DataFrame(data)
            df.to_excel(writer, sheet_name='1_Aux', index=False, header=False, startrow=0, startcol=18)

def export_secondary():
    filepath = filedialog.asksaveasfilename(defaultextension='.xlsx',
                                            filetypes=[("Excel xlsx", "*.xlsx"), ("Excel csv", "*.csv")])
    if filepath == "":
        return
    else:
        with pandas.ExcelWriter(filepath) as writer:
            rxn_summary_df_2.to_excel(writer, sheet_name='2_Summary', index=True)
            byproducts_df_2.to_excel(writer, sheet_name='2_Summary', index=True, startrow=0, startcol=18)
            Xn_2.to_excel(writer, sheet_name='2_In_Situ', index=True)
            data = []
            for row in range(RET.tableheight):
                row_data = []
                for column in range(RET.tablewidth):
                    entry = RET.entries[row * RET.tablewidth + column]
                    row_data.append(entry.get())
                data.append(row_data)
            df = pandas.DataFrame(data)
            df.to_excel(writer, sheet_name='2_Aux', index=False, header=False)

            data = []
            for column in range(WD2.tablewidth):
                column_data = []
                for row in range(WD2.tableheight):
                    entry = WD2.entries[row + column * WD2.tableheight]
                    column_data.append(entry.get())
                data.append(column_data)
            data = list(map(list, zip(*data)))
            df = pandas.DataFrame(data)
            df.to_excel(writer, sheet_name='2_Aux', index=False, header=False, startrow=0, startcol=18)

def export_all():
    filepath = filedialog.asksaveasfilename(defaultextension='.xlsx',
                                            filetypes=[("Excel xlsx", "*.xlsx"), ("Excel csv", "*.csv")])
    if filepath == "":
        return
    else:
        with pandas.ExcelWriter(filepath) as writer:
            rxn_summary_df.to_excel(writer, sheet_name='1_Summary', index=True)
            byproducts_df.to_excel(writer, sheet_name='1_Summary', index=True, startrow=0, startcol=18)
            Xn.to_excel(writer, sheet_name='1_In_Situ', index=True)
            rxn_summary_df_2.to_excel(writer, sheet_name='2_Summary', index=True)
            byproducts_df_2.to_excel(writer, sheet_name='2_Summary', index=True, startrow=0, startcol=18)
            Xn_2.to_excel(writer, sheet_name='2_In_Situ', index=True)
            data = []
            for row in range(RET.tableheight):
                row_data = []
                for column in range(RET.tablewidth):
                    entry = RET.entries[row * RET.tablewidth + column]
                    row_data.append(entry.get())
                data.append(row_data)
            df = pandas.DataFrame(data)
            df.to_excel(writer, sheet_name='1_Aux', index=False, header=False)
            df.to_excel(writer, sheet_name='2_Aux', index=False, header=False)

            data = []
            for column in range(WD.tablewidth):
                column_data = []
                for row in range(WD.tableheight):
                    entry = WD.entries[row + column * WD.tableheight]
                    column_data.append(entry.get())
                data.append(column_data)
            data = list(map(list, zip(*data)))
            df = pandas.DataFrame(data)
            df.to_excel(writer, sheet_name='1_Aux', index=False, header=False, startrow=0, startcol=18)

            data = []
            for column in range(WD2.tablewidth):
                column_data = []
                for row in range(WD2.tableheight):
                    entry = WD2.entries[row + column * WD2.tableheight]
                    column_data.append(entry.get())
                data.append(column_data)
            data = list(map(list, zip(*data)))
            df = pandas.DataFrame(data)
            df.to_excel(writer, sheet_name='2_Aux', index=False, header=False, startrow=0, startcol=18)

            workbook = writer.book
            sheet_names = ['1_Summary', '1_In_Situ', '1_Aux', '2_Summary', '2_In_Situ', '2_Aux']
            workbook._sheets.sort(key=lambda ws: sheet_names.index(ws.title))

# ---------------------------------------------------User-Interface----------------------------------------------#
if __name__ == "__main__":
    window = tkinter.Tk()
    style = ttk.Style(window)
    style.theme_use('clam')
    style.configure('TNotebook.Tab', background='#355C7D', foreground='#ffffff')
    style.configure("red.Horizontal.TProgressbar", troughcolor='green')
    style.map('TNotebook.Tab', background=[('selected', 'green3')], foreground=[('selected', '#000000')])
    window.iconbitmap("testtube.ico")
    window.title("Monte Karlo")
    window.geometry("{0}x{1}+0+0".format(window.winfo_screenwidth(), window.winfo_screenheight()))
    window.configure(background="#000000")

    tab_control = ttk.Notebook(window)
    tab1 = ttk.Frame(tab_control, style='TNotebook.Tab')
    tab2 = ttk.Frame(tab_control, style='TNotebook.Tab')
    tab3 = ttk.Frame(tab_control, style='TNotebook.Tab')
    tab4 = ttk.Frame(tab_control, style='TNotebook.Tab')
    tab5 = ttk.Frame(tab_control, style='TNotebook.Tab')
    tab_control.add(tab1, text='Reactor')
    tab_control.add(tab2, text='1° Reaction Results')
    tab_control.add(tab3, text='1° In-Situ Results')
    tab_control.add(tab4, text='2° Reaction Results')
    tab_control.add(tab5, text='2° In-Situ Results')
    tkinter.Grid.rowconfigure(window, 0, weight=1)
    tkinter.Grid.columnconfigure(window, 0, weight=1)
    tab_control.grid(row=0, column=0, sticky=tkinter.E + tkinter.W + tkinter.N + tkinter.S)

    menubar = tkinter.Menu(window, background="red")
    window.config(menu=menubar)
    filemenu1 = tkinter.Menu(menubar, tearoff=0)
    export_menu = tkinter.Menu(filemenu1, tearoff=0)
    filemenu2 = tkinter.Menu(menubar, tearoff=0)
    APC_Menu = tkinter.Menu(filemenu2, tearoff=0)
    filemenu3 = tkinter.Menu(menubar, tearoff=0)
    menubar.add_cascade(label='File', menu=filemenu1)
    menubar.add_cascade(label='Options', menu=filemenu2)
    menubar.add_cascade(label='Help', menu=filemenu3)
    filemenu1.add_cascade(label="Export", menu=export_menu)
    filemenu2.add_cascade(label="Calculate", menu=APC_Menu)
    export_menu.add_command(label='1° Reaction Results', command=export_primary)
    export_menu.add_command(label='2° Reaction Results', command=export_secondary)
    export_menu.add_command(label='All Reaction Results', command=export_all)
    APC_Menu.add_command(label='1° APC Chromatograph', command=lambda: run_APC(rxn_summary_df, label="1°"))
    APC_Menu.add_command(label='2° APC Chromatograph', command=lambda: run_APC(rxn_summary_df_2, label="2°"))
    filemenu1.add_command(label='Reset', command=reset_entry_table)
    filemenu1.add_command(label='Exit', command=window.destroy)
    filemenu2.add_command(label='Quick Add', command=quick_add)
    filemenu3.add_command(label='Help')

    Entry_Reactants = ['R1Reactant', 'R2Reactant', 'R3Reactant', 'R4Reactant', 'R5Reactant', 'R6Reactant', 'R7Reactant',
                       'R8Reactant', 'R9Reactant', 'R10Reactant', 'R11Reactant', 'R12Reactant', 'R13Reactant',
                       'R14Reactant', 'R15Reactant', 'R16Reactant', 'R17Reactant', 'R18Reactant', 'R19Reactant',]
    Entry_masses = ['R1mass', 'R2mass', 'R3mass', 'R4mass', 'R5mass', 'R6mass', 'R7mass', 'R8mass', 'R9mass', 'R10mass',
                    'R11mass', 'R12mass', 'R13mass', 'R14mass', 'R15mass', 'R16mass', 'R17mass', 'R18mass', 'R19mass',]
    RDE = ['R1Data', 'R2Data', 'R3Data', 'R4Data', 'R5Data', 'R6Data', 'R7Data', 'R8Data', 'R9Data', 'R10Data', 'R11Data',
           'R12Data', 'R13Data', 'R14Data', 'R15Data', 'R16Data', 'R17Data', 'R18Data', 'R19Data',]

    global starting_cell
    starting_cell = 16


    class APCParametersWindow(simpledialog.Dialog):
        def body(self, master):
            self.title("APC Parameters")
            self.iconbitmap("testtube.ico")
            Label(master, text="Flow Rate (ml/min):").grid(row=0, column=0)
            self.flow_rate = DoubleVar(value=1.0)  # Set default value to 1.0 ml/min
            self.e1 = Entry(master, width=18, textvariable=self.flow_rate)
            self.e1.grid(row=0, column=1)
            Label(master, text="100 MW FWHM (min):").grid(row=1, column=0)
            self.fwhm = DoubleVar(value=0.060) # Set default FWHM for 100 MW
            self.e2 = Entry(master, width=18, textvariable=self.fwhm)
            self.e2.grid(row=1, column=1)
            Label(master, text="100K MW FWHM (min):").grid(row=2, column=0)
            self.fwhm2 = DoubleVar(value=0.1) # Set default FWHM for 100K MW
            self.e3 = Entry(master, width=18, textvariable=self.fwhm2)
            self.e3.grid(row=2, column=1)
            Label(master, text="Temperature (°C):").grid(row=3, column=0)
            self.temp = DoubleVar(value=35) # Set default temperature to 35°C
            self.e4 = Entry(master, width=18, textvariable=self.temp)
            self.e4.grid(row=3, column=1)
            self.flow_rate.trace('w', self.update_fwhm)
            return self.e1

        def update_fwhm(self, *args):
            try:
                flow_rate = self.flow_rate.get()
                self.fwhm.set(0.06 / flow_rate)
                self.fwhm2.set(0.1 / flow_rate)
            except ZeroDivisionError:
                pass
            except ValueError:
                pass

        def buttonbox(self):
            box = Frame(self)
            w = Button(box, text="Submit", width=10, command=self.ok)
            w.pack(side=LEFT, padx=5, pady=5)
            w = Button(box, text="Cancel", width=10, command=self.cancel)
            w.pack(side=LEFT, padx=5, pady=5)
            self.bind("<Return>", self.ok)
            self.bind("<Escape>", self.cancel)
            box.pack()

        def cancel(self):
            global apc_params
            apc_params = None
            self.destroy()

        def ok(self):
            global apc_params
            apc_params = [self.flow_rate.get(), self.fwhm.get(), self.fwhm2.get(), self.temp.get()]
            self.destroy()

    class QuickAddWindow(simpledialog.Dialog):
        def body(self, master):
            self.title("Quick Add")
            Label(master, text="Reactant:").grid(row=0, column=0)
            Label(master, text="Mass (g):").grid(row=1, column=0)
            comp_to_add = tkinter.StringVar()
            self.e1 = AutocompleteCombobox(master, completevalues=quick_adds, textvariable=comp_to_add, width=15)
            self.e2 = Entry(master, width=18)
            self.e1.grid(row=0, column=1)
            self.e2.grid(row=1, column=1)
            return self.e1

        def buttonbox(self):
            box = Frame(self)
            w = Button(box, text="Submit", width=10, command=self.ok)
            w.pack(side=LEFT, padx=5, pady=5)
            w = Button(box, text="Cancel", width=10, command=self.cancel)
            w.pack(side=LEFT, padx=5, pady=5)
            self.bind("<Return>", self.ok)
            self.bind("<Escape>", self.cancel)
            box.pack()

        def cancel(self):
            global quick_add_comp
            quick_add_comp = None
            self.destroy()

        def ok(self):
            global quick_add_comp
            quick_add_comp = [self.e1.get(), float(self.e2.get())]
            self.destroy()

    class RxnEntryTable(tkinter.Frame):
        def __init__(self, master=tab1):
            tkinter.Frame.__init__(self, master)
            self.tablewidth = 16
            self.tableheight = 20
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
            self.entries[offset + 0].insert(0, "Component")
            self.entries[offset + 0].config(state="readonly", font=("Helvetica", 8, "bold"))
            self.entries[offset + 1].insert(0, "Mass (g)")
            self.entries[offset + 1].config(state="readonly", font=("Helvetica", 8, "bold"))
            self.entries[offset + 2].insert(0, "wt, %")
            self.entries[offset + 2].config(state="readonly", font=("Helvetica", 8, "bold"))
            self.entries[offset + 3].insert(0, "Moles")
            self.entries[offset + 3].config(state="readonly", font=("Helvetica", 8, "bold"))
            self.entries[offset + 4].insert(0, "1° - ID")
            self.entries[offset + 4].config(state="readonly", font=("Helvetica", 8, "bold"))
            self.entries[offset + 5].insert(0, "1° - K")
            self.entries[offset + 5].config(state="readonly", font=("Helvetica", 8, "bold"))
            self.entries[offset + 6].insert(0, "C1° - ID")
            self.entries[offset + 6].config(state="readonly", font=("Helvetica", 8, "bold"))
            self.entries[offset + 7].insert(0, "1° - Child K")
            self.entries[offset + 7].config(state="readonly", font=("Helvetica", 8, "bold"))
            self.entries[offset + 8].insert(0, "2° - ID")
            self.entries[offset + 8].config(state="readonly", font=("Helvetica", 8, "bold"))
            self.entries[offset + 9].insert(0, "2° - K")
            self.entries[offset + 9].config(state="readonly", font=("Helvetica", 8, "bold"))
            self.entries[offset + 10].insert(0, "C2° - ID")
            self.entries[offset + 10].config(state="readonly", font=("Helvetica", 8, "bold"))
            self.entries[offset + 11].insert(0, "2° - Child K")
            self.entries[offset + 11].config(state="readonly", font=("Helvetica", 8, "bold"))
            self.entries[offset + 12].insert(0, "3° - ID")
            self.entries[offset + 12].config(state="readonly", font=("Helvetica", 8, "bold"))
            self.entries[offset + 13].insert(0, "3° - K")
            self.entries[offset + 13].config(state="readonly", font=("Helvetica", 8, "bold"))
            self.entries[offset + 14].insert(0, "C3° - ID")
            self.entries[offset + 14].config(state="readonly", font=("Helvetica", 8, "bold"))
            self.entries[offset + 15].insert(0, "3° - Child K")
            self.entries[offset + 15].config(state="readonly", font=("Helvetica", 8, "bold"))
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
            if self.entries[cell + 1].get() == "" or self.entries[cell + 1].get() == "0":
                self.entries[cell + 1].delete(0, tkinter.END)
                self.entries[cell + 2].config(state="normal")
                self.entries[cell + 2].delete(0, tkinter.END)
            if self.entries[cell].get() == "Clear":
                self.entries[cell].delete(0, tkinter.END)
                self.entries[cell + 1].delete(0, tkinter.END)
                self.entries[cell + 2].config(state="normal")
                self.entries[cell + 2].delete(0, tkinter.END)
                self.entries[cell + 3].delete(0, tkinter.END)
                self.entries[cell + 4].config(state="normal")
                self.entries[cell + 4].delete(0, tkinter.END)
                self.entries[cell + 5].delete(0, tkinter.END)
                self.entries[cell + 6].config(state="normal")
                self.entries[cell + 6].delete(0, tkinter.END)
                self.entries[cell + 7].delete(0, tkinter.END)
                self.entries[cell + 8].config(state="normal")
                self.entries[cell + 8].delete(0, tkinter.END)
                self.entries[cell + 9].delete(0, tkinter.END)
                self.entries[cell + 10].config(state="normal")
                self.entries[cell + 10].delete(0, tkinter.END)
                self.entries[cell + 11].delete(0, tkinter.END)
                self.entries[cell + 12].config(state="normal")
                self.entries[cell + 12].delete(0, tkinter.END)
                self.entries[cell + 13].delete(0, tkinter.END)
                self.entries[cell + 14].config(state="normal")
                self.entries[cell + 14].delete(0, tkinter.END)
                self.entries[cell + 15].delete(0, tkinter.END)
            else:
                if self.entries[cell].get() != "" and self.entries[cell + 1].get() != "":
                    a = str_to_class(Entry_Reactants[index].get())()
                    molesA = float(Entry_masses[index].get()) / float(a.mw)
                    self.entries[cell + 3].delete(0, tkinter.END)
                    self.entries[cell + 3].insert(0, str(round(molesA, 6)))

            def sum_mass():
                total = 0
                for entry in Entry_masses:
                    if entry.get() != "":
                        total = total + float(entry.get())
                return total

            def weight_percent():
                cell = 17
                index = 0
                for i in range(self.tableheight - 1):
                    if Entry_masses[index].get() != "":
                        self.entries[cell + 1].config(state="normal")
                        self.entries[cell + 1].delete(0, tkinter.END)
                        self.entries[cell + 1].insert(0,
                                                      str(round((float(Entry_masses[index].get()) / sum_mass()) * 100,
                                                                3)))
                        self.entries[cell + 1].config(state="readonly")
                    cell = cell + self.tablewidth
                    index = index + 1

            weight_percent()

        def update_rates(self, index, cell):
            if self.entries[cell].get() != "Clear" and self.entries[cell].get() != "":
                a = str_to_class(Entry_Reactants[index].get())()
                self.entries[cell + 4].config(state="normal")
                self.entries[cell + 4].delete(0, tkinter.END)
                self.entries[cell + 4].insert(0, str(a.prgID))
                self.entries[cell + 4].config(state="readonly")
                self.entries[cell + 5].delete(0, tkinter.END)
                self.entries[cell + 5].insert(0, str(a.prgk))
                self.entries[cell + 6].config(state="normal")
                self.entries[cell + 6].delete(0, tkinter.END)
                self.entries[cell + 6].insert(0, str(a.cprgID))
                self.entries[cell + 6].config(state="readonly")
                self.entries[cell + 7].delete(0, tkinter.END)
                self.entries[cell + 7].insert(0, str(a.cprgk))
                self.entries[cell + 8].config(state="normal")
                self.entries[cell + 8].delete(0, tkinter.END)
                self.entries[cell + 8].insert(0, str(a.srgID))
                self.entries[cell + 8].config(state="readonly")
                self.entries[cell + 9].delete(0, tkinter.END)
                self.entries[cell + 9].insert(0, str(a.srgk))
                self.entries[cell + 10].config(state="normal")
                self.entries[cell + 10].delete(0, tkinter.END)
                self.entries[cell + 10].insert(0, str(a.csrgID))
                self.entries[cell + 10].config(state="readonly")
                self.entries[cell + 11].delete(0, tkinter.END)
                self.entries[cell + 11].insert(0, str(a.csrgk))
                self.entries[cell + 12].config(state="normal")
                self.entries[cell + 12].delete(0, tkinter.END)
                self.entries[cell + 12].insert(0, str(a.trgID))
                self.entries[cell + 12].config(state="readonly")
                self.entries[cell + 13].delete(0, tkinter.END)
                self.entries[cell + 13].insert(0, str(a.trgk))
                self.entries[cell + 14].config(state="normal")
                self.entries[cell + 14].delete(0, tkinter.END)
                self.entries[cell + 14].insert(0, str(a.ctrgID))
                self.entries[cell + 14].config(state="readonly")
                self.entries[cell + 15].delete(0, tkinter.END)
                self.entries[cell + 15].insert(0, str(a.ctrgk))
            else:
                pass


    global RXN_Type, RXN_Samples, RXN_EOR, RXN_EM, RXN_EM_Value, NUM_OF_SIM
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
            self.tableheight = 4
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
            self.entries[2].delete(0, tkinter.END)
            self.entries[2].insert(0, "# of samples =")
            self.entries[2].config(state="readonly")
            self.entries[3].delete(0, tkinter.END)
            self.entries[3].insert(0, "# of Simulations =")
            self.entries[3].config(state="readonly")
            self.user_entry()

        def user_entry(self):
            global RXN_Type, RXN_Samples, RXN_EOR, RXN_EM, RXN_EM_Value, RXN_EM_2, RXN_EM_Entry_2, RXN_EM_2_SR, RXN_EM_Entry_2_SR, RXN_EM_2_Active, RXN_EM_2_Check, RXN_EM_Value_2, RXN_EM_Entry, NUM_OF_SIM
            RXN_EM = tkinter.StringVar()
            reactants_list = []
            RXN_EM_Entry = AutocompleteCombobox(self, completevalues=End_Metrics, width=15, textvariable=RXN_EM)
            RXN_EM_Entry.grid(row=0, column=0)
            RXN_EM_Entry.insert(0, "1º End Metric")
            RXN_EM_Entry.config(justify="center", state="readonly")
            RXN_EM_Value = self.entries[4]
            RXN_EM_2_Active = tkinter.BooleanVar()
            RXN_EM_2_Check = tkinter.Checkbutton(self, text="2º Active?", variable=RXN_EM_2_Active, onvalue=True,
                                                 offvalue=False)
            RXN_EM_2_Check.grid(row=0, column=2)
            RXN_EM_2 = tkinter.StringVar()
            RXN_EM_Entry_2 = AutocompleteCombobox(self, completevalues=End_Metrics, width=15, textvariable=RXN_EM_2)
            RXN_EM_Entry_2.grid(row=1, column=0)
            RXN_EM_Entry_2.insert(0, "2º End Metric")
            RXN_EM_Entry_2.config(justify="center", state="readonly")
            RXN_EM_Value_2 = self.entries[5]
            RXN_EM_2_SR = tkinter.StringVar()
            RXN_EM_Entry_2_SR = AutocompleteCombobox(self, completevalues=reactants_list, width=15,
                                                     textvariable=RXN_EM_2_SR)
            RXN_EM_Entry_2_SR.grid(row=1, column=2)
            if RXN_EM_Entry_2_SR.get() == "":
                RXN_EM_Entry_2_SR.insert(0, "2º Start")
            RXN_EM_Entry_2_SR.config(justify="center")
            RXN_Samples = tkinter.StringVar()
            RXN_Samples_Entry = AutocompleteCombobox(self, completevalues=Num_Samples, width=15,
                                                     textvariable=RXN_Samples)
            RXN_Samples_Entry.insert(0, "5000")
            RXN_Samples_Entry.grid(row=2, column=1)
            RXN_Samples_Entry.config(justify="center")


            NUM_OF_SIM = tkinter.StringVar()
            Core_options = [str(i) for i in range(1, multiprocessing.cpu_count() -1)]
            NUM_OF_SIM_Entry = AutocompleteCombobox(self, completevalues=Core_options, width=15,
                                                    textvariable=NUM_OF_SIM)
            NUM_OF_SIM_Entry.insert(0, str(int(multiprocessing.cpu_count()*0.75)))
            NUM_OF_SIM_Entry.grid(row=3, column=1)
            NUM_OF_SIM_Entry.config(justify="center")

            self.get_reactants()

        def get_reactants(self):
            global reactants_list, RXN_EM_2_SR, RXN_EM_Entry_2_SR
            reactants_list = []
            cell = 16
            for i in range(RET.tableheight - 1):
                if RET.entries[cell].get() == "" or RET.entries[cell].get() == "Clear":
                    cell += RET.tablewidth
                    pass
                else:
                    reactants_list.append(RET.entries[cell].get())
                    cell += RET.tablewidth
            reactants_list = [f'{index + 1}: {reactant}' for index, reactant in enumerate(reactants_list)]
            RXN_EM_Entry_2_SR.config(values=reactants_list, state="readonly")


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
            self.tableheight = 8
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
            self.entries[5].delete(0, tkinter.END)
            self.entries[5].insert(0, "COC Value =")
            self.entries[5].config(state="readonly")
            self.entries[6].delete(0, tkinter.END)
            self.entries[6].insert(0, "Iodine Value =")
            self.entries[6].config(state="readonly")


    class RxnMetrics_sec(tkinter.Frame):
        def __init__(self, master=tab1):
            tkinter.Frame.__init__(self, master)
            self.tablewidth = None
            self.tableheight = None
            self.entries = None
            self.grid(row=2, column=1, padx=(500, 0), pady=5)
            self.create_table()

        def create_table(self):
            self.entries = {}
            self.tableheight = 8
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
            self.entries[5].delete(0, tkinter.END)
            self.entries[5].insert(0, "COC Value =")
            self.entries[5].config(state="readonly")
            self.entries[6].delete(0, tkinter.END)
            self.entries[6].insert(0, "Iodine Value =")
            self.entries[6].config(state="readonly")


    class WeightDist(tkinter.Frame):
        def __init__(self, master=tab2):
            tkinter.Frame.__init__(self, master)
            self.tablewidth = None
            self.tableheight = None
            self.entries = None
            self.grid(row=0, column=0, padx=15, pady=15)
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
                    self.entries[counter].config(justify="center", width=15)
                    counter += 1
            self.table_labels()

        def table_labels(self):
            self.entries[0].delete(0, tkinter.END)
            self.entries[0].insert(0, "Mn (Number Average) =")
            self.entries[0].config(width=25)
            self.entries[0].config(state="readonly")
            self.entries[1].delete(0, tkinter.END)
            self.entries[1].insert(0, "Mw (Weight Average) =")
            self.entries[1].config(width=25)
            self.entries[1].config(state="readonly")
            self.entries[2].delete(0, tkinter.END)
            self.entries[2].insert(0, "PDI (Dispersity Index) =")
            self.entries[2].config(width=25)
            self.entries[2].config(state="readonly")
            self.entries[3].delete(0, tkinter.END)
            self.entries[3].insert(0, "Mz =")
            self.entries[3].config(width=25)
            self.entries[3].config(state="readonly")
            self.entries[4].delete(0, tkinter.END)
            self.entries[4].insert(0, "Mz + 1 =")
            self.entries[4].config(width=25)
            self.entries[4].config(state="readonly")
            self.entries[5].delete(0, tkinter.END)
            self.entries[5].insert(0, "Xn DOP = ")
            self.entries[5].config(width=25)
            self.entries[5].config(state="readonly")


    class WeightDist_2(tkinter.Frame):
        def __init__(self, master=tab4):
            tkinter.Frame.__init__(self, master)
            self.tablewidth = None
            self.tableheight = None
            self.entries = None
            self.grid(row=0, column=0, padx=15, pady=15)
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
                    self.entries[counter].config(justify="center", width=15)
                    counter += 1
            self.table_labels()

        def table_labels(self):
            self.entries[0].delete(0, tkinter.END)
            self.entries[0].insert(0, "Mn (Number Average) =")
            self.entries[0].config(width=25)
            self.entries[0].config(state="readonly")
            self.entries[1].delete(0, tkinter.END)
            self.entries[1].insert(0, "Mw (Weight Average) =")
            self.entries[1].config(width=25)
            self.entries[1].config(state="readonly")
            self.entries[2].delete(0, tkinter.END)
            self.entries[2].insert(0, "PDI (Dispersity Index) =")
            self.entries[2].config(width=25)
            self.entries[2].config(state="readonly")
            self.entries[3].delete(0, tkinter.END)
            self.entries[3].insert(0, "Mz =")
            self.entries[3].config(width=25)
            self.entries[3].config(state="readonly")
            self.entries[4].delete(0, tkinter.END)
            self.entries[4].insert(0, "Mz + 1 =")
            self.entries[4].config(width=25)
            self.entries[4].config(state="readonly")
            self.entries[5].delete(0, tkinter.END)
            self.entries[5].insert(0, "Xn DOP = ")
            self.entries[5].config(width=25)
            self.entries[5].config(state="readonly")


    class Buttons(tkinter.Frame):
        def __init__(self, master=tab1):
            tkinter.Frame.__init__(self, master)
            self.Simulate = None
            self.tablewidth = None
            self.tableheight = None
            self.entries = None
            self.grid(row=1, column=0, padx=5, pady=5)
            self.create_table()

        def create_table(self):
            self.entries = {}
            self.tableheight = 4
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
            self.Simulate = tkinter.Button(self, text="Simulate", command=multiprocessing_sim, width=15, bg="Green")
            self.Simulate.grid(row=0, column=0)
            stop_button = tkinter.Button(self, text="Terminate", command=stop, width=15, bg="Red")
            stop_button.grid(row=1, column=0)
            clear_last_row = tkinter.Button(self, text="Clear Last", command=clear_last, width=15, bg="Yellow")
            clear_last_row.grid(row=2, column=0)
            self.Reset = tkinter.Button(self, text="Reset", command=reset_entry_table, width=15, bg="Orange")
            self.Reset.grid(row=3, column=0)


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
            self.tableheight = 2
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
            self.entries[0].insert(0, "1º Simulation Status")
            self.entries[0].config(state="readonly")
            self.entries[1].delete(0, tkinter.END)
            self.entries[1].insert(0, "2º Simulation Status")
            self.entries[1].config(state="readonly")
            self.add_buttons()

        def add_buttons(self):
            self.progress = ttk.Progressbar(self, orient="horizontal", length=300, mode="determinate",
                                            style="red.Horizontal.TProgressbar")
            self.progress.grid(row=0, column=1)
            self.progress_2 = ttk.Progressbar(self, orient="horizontal", length=300, mode="determinate",
                                              style="red.Horizontal.TProgressbar")
            self.progress_2.grid(row=1, column=1)


    RET = RxnEntryTable()
    WD = WeightDist()
    WD2 = WeightDist_2()
    RD = RxnDetails()
    RM = RxnMetrics()
    RM2 = RxnMetrics_sec()
    Buttons = Buttons()
    sim = SimStatus()

    # run update_table if user changes value in RET
    for i in range(16, 305, 16):
        RET.entries[i].bind("<FocusOut>", lambda *args, entry=i, index=int(i/16-1), cell=i: check_entry(entry, index, cell))

    # run update_table if user changes value in RET
    for i in range(0, 19):
        Entry_masses[i].bind("<KeyRelease>", lambda *args, index=i, cell=int(i*16+16): RET.update_table(index, cell))


    window.bind('<Control-s>', lambda *args: initialize_sim())
    RXN_EM_Entry_2_SR.bind('<Enter>', lambda *args: RD.get_reactants())
    window.bind('<Control-a>', lambda *args: quick_add())
    window.bind('<Control-e>', lambda *args: reset_entry_table())
    window.bind('<Control-q>', lambda *args: quit())
    window.bind('<Control-r>', lambda *args: reset_entry_table())
    window.bind('<Control-l>', lambda *args: clear_last())
    RXN_EM_Entry.bind('<KeyRelease>', lambda *args: RXN_EM_Value.focus())
    RXN_EM_Entry_2.bind('<KeyRelease>', lambda *args: RXN_EM_Value_2.focus())

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
    R15Data = R15Data()
    R16Data = R16Data()
    R17Data = R17Data()
    R18Data = R18Data()
    R19Data = R19Data()

    rg = reactive_groups()

    window.mainloop()
