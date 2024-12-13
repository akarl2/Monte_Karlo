import collections
import queue
import multiprocessing
import os
import random
import sys
import tkinter
import time
from Cluster_Analysis import ClusterAnalysis
import sv_ttk
from tkinter import *
import mplcursors
from scipy.integrate import trapz, simps
from matplotlib.widgets import Cursor, Slider, SpanSelector
from mpl_toolkits import *
from matplotlib.backends.backend_tkagg import *
import matplotlib.backends.backend_tkagg as tkagg
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from openpyxl import Workbook
import openpyxl
import customtkinter
from tkinter import ttk, messagebox, simpledialog, filedialog
import pandas
import concurrent.futures
import platform

import Correlation_Matrix
import scipy
from ttkwidgets.autocomplete import AutocompleteCombobox

from Correlation_Matrix import calc_corr_matrix
from Database import *
from Logistic_Reg import run_regression
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
from nn_Builder import NeuralNetworkArchitectureBuilder
from Reactants import R1Data, R2Data, R3Data, R4Data, R5Data, R6Data, R7Data, R8Data, R9Data, R10Data, R11Data, R12Data, \
    R13Data, R14Data, R15Data, R16Data, R17Data, R18Data, R19Data

# Runs the simulation
global running, emo_a, results_table, frame_results, expanded_results, groupA, groupB, test_count, test_interval, \
    total_ct, total_ct_sec, sn_dist, TAV, AV, OH, COC, EHC, in_situ_values, in_situ_values_sec, byproducts, \
    frame_byproducts, low_group, RXN_EM_2, RXN_EM_Entry_2, RXN_EM_2_SR, reactants_list, RXN_EM_2_SR, RXN_EM_Entry_2_SR, results_table_2, \
    frame_results_2, byproducts_table_2, frame_byproducts_2, RXN_EM_2_Active, RXN_EM_2_Check, RXN_EM_Value_2, in_primary, quick_add, quick_add_comp, \
    RXN_EM_Value, RXN_EM_Entry, rxn_summary_df, rxn_summary_df_2, Xn, Xn_2, end_metric_value, end_metric_value_sec, RXN_EM_2_Active_status, end_metric_selection, \
    end_metric_selection_sec, starting_mass_sec, starting_mass, sn_dict, samples_value, total_samples, canceled_by_user, metrics, RXN_EM_Operator, RXN_EM_Operator_2, ks_mod, k_groups


# -------------------------------------------Simulate---------------------------------------------------#
def simulate(starting_materials, starting_materials_sec, end_metric_value, end_metric_selection, end_metric_value_sec,
             end_metric_selection_sec, sn_dict, RXN_EM_2_Active_status, total_ct, total_ct_sec, workers, process_queue,
             PID_list, progress_queue_sec, RXN_EM_Operator_sel, RXN_EM_Operator_2_sel, ks_mod, k_groups):
    PID_list.append(os.getpid())
    currentPID = os.getpid()
    time.sleep(1)
    rg = reactive_groups()
    global test_count, test_interval, sn_dist, in_situ_values, byproducts, running, in_primary, in_situ_values_sec, quick_add, comp_primary, comp_secondary
    in_situ_values = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    in_situ_values_sec = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    byproducts, composition, composition_sec = [], [], []
    running, in_primary = True, True
    test_count = 0
    test_interval = 50
    try:
        end_metric_value_upper = end_metric_value * 1.15
        end_metric_value_lower = end_metric_value * 0.85
        if RXN_EM_2_Active_status:
            end_metric_value_upper_sec = end_metric_value_sec * 1.15
            end_metric_value_lower_sec = end_metric_value_sec * 0.85
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
        is_EHC = False
        global groupA, groupB
        groupA = groups[0][2]
        groupB = groups[1][2]
        # if 'Epi' in sn_dict:
        #     sOHk = sn_dict['Epi'].cprgID[1]
        #     if groupA == "SOH" and composition[groups[1][0]][0][groups[1][1]][1] == sOHk:
        #         is_EHC = True
        #         if groupB == "OH" and is_EHC:
        #             return True
        try:
            if groupB in getattr(rg, groupA):
                new_group(groupA, groupB)
                return True
            else:
                return False
        except TypeError:
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
        global test_count, NG2, new_group_dict
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
            RXN_Status(composition, byproducts)
            test_count = 0

    def RXN_Status(composition, byproducts):
        global test_interval, in_situ_values, running, in_primary, comp_primary, comp_secondary, byproducts_primary, byproducts_secondary, metrics
        comp_secondary, byproducts_secondary = None, None
        if not in_primary:
            comp_summary = collections.Counter(
                [(tuple(tuple(i) for i in sublist[0]), tuple(tuple(i) for i in sublist[1]), sublist[2][0]) for sublist
                 in composition])
            sum_comp = sum([comp_summary[key] * key[2] for key in comp_summary])
        else:
            comp_summary = collections.Counter(
                [(tuple(tuple(i) for i in sublist[0]), tuple(tuple(i) for i in sublist[1]), sublist[2][0]) for sublist
                 in composition])
            sum_comp = sum([comp_summary[key] * key[2] for key in comp_summary])

        sum_comp_2 = sum([comp_summary[key] * key[2] ** 2 for key in comp_summary])
        total_ct_temp = sum(comp_summary[key] for key in comp_summary)
        Mw = sum_comp_2 / sum_comp
        Mn = sum_comp / total_ct_temp

        amine_ct, acid_ct, alcohol_ct, epoxide_ct, EHC_ct, IV_ct, NH2_ct, NH_ct, N_ct, totCl_ct, POH_ct, SOH_ct = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        for key in comp_summary:
            key_names = [i[0] for i in key[0]]
            totCl_ct += key_names.count('Cl') * comp_summary[key]
            NH2_ct += key_names.count('NH2') * comp_summary[key] + key_names.count('α_NH2') * comp_summary[key]
            NH_ct += key_names.count('NH') * comp_summary[key]
            N_ct += key_names.count('N') * comp_summary[key]
            amine_ct = NH2_ct + NH_ct + N_ct
            acid_ct += key_names.count('COOH') * comp_summary[key]
            epoxide_ct += key_names.count('COC') * comp_summary[key]
            IV_ct += (key_names.count('aB_unsat') + key_names.count('CC_3') + key_names.count('CC_2') + key_names.count('CC_1')) * comp_summary[key]
            Cl_ct = key_names.count('Cl')
            POH_ct += key_names.count('POH') * comp_summary[key]
            SOH_ct += key_names.count('SOH') * comp_summary[key]
            alcohol_ct = POH_ct + SOH_ct
            for group in key[0]:
                if group[0] == 'POH' or group[0] == 'SOH':
                    if 'Epi' in sn_dict and group[0] == 'SOH' and group[1] == sn_dict['Epi'].cprgID[1] and Cl_ct > 0:
                        Cl_ct -= 1
                        EHC_ct += comp_summary[key]

        TAV = round((amine_ct * 56100) / sum_comp, 2)
        p_TAV = round((NH2_ct * 56100) / sum_comp, 2)
        s_TAV = round((NH_ct * 56100) / sum_comp, 2)
        t_TAV = round((N_ct * 56100) / sum_comp, 2)
        AV = round((acid_ct * 56100) / sum_comp, 2)
        OH = round((alcohol_ct * 56100) / sum_comp, 2)
        POH = round((POH_ct * 56100) / sum_comp, 2)
        SOH = round((SOH_ct * 56100) / sum_comp, 2)
        COC = round((epoxide_ct * 56100) / sum_comp, 2)
        EHC = round((EHC_ct * 35.453) / sum_comp * 100, 2)
        IV = round(((IV_ct * 2) * (127 / sum_comp) * 100), 2)
        Cl = round((totCl_ct * 35.453) / sum_comp * 100, 2)
        if not in_primary:
            Xn = round((total_ct_sec / workers) / total_ct_temp, 4)
        else:
            Xn = round((total_ct / workers) / total_ct_temp, 4)

        metrics = {'Amine Value': TAV, 'Acid Value': AV, 'OH Value': OH, 'Epoxide Value': COC, '% EHC': EHC,
                   'Iodine Value': IV, 'MW': Mw, 'Mn': Mn, '1° TAV': p_TAV, '2° TAV': s_TAV, '3° TAV': t_TAV, 'Xn': Xn,
                   '% Cl': Cl, '1° OH': POH, '2° OH': SOH}

        if not in_primary: 
            if not in_primary:
                for i, (metric_name, variable) in enumerate(metrics.items()):
                    in_situ_values_sec[i].append(variable)
                RXN_metric_value = metrics[end_metric_selection_sec]
            if (RXN_EM_Operator_2_sel == '<=' and RXN_metric_value <= end_metric_value_upper_sec) or (
                    RXN_EM_Operator_2_sel == '>=' and RXN_metric_value >= end_metric_value_lower_sec):
                test_interval = 1
            if end_metric_selection_sec != '% EHC':
                if currentPID == PID_list[-1]:
                    progress_queue_sec.put(round((end_metric_value_sec / RXN_metric_value) * 100), 2)
                if (RXN_EM_Operator_2_sel == '<=' and RXN_metric_value <= end_metric_value_sec) or (
                        RXN_EM_Operator_2_sel == '>=' and RXN_metric_value >= end_metric_value_sec):
                    comp_secondary = tuple(composition)
                    byproducts_secondary = byproducts
                    running = False
        else:
            for i, (metric_name, variable) in enumerate(metrics.items()):
                in_situ_values[i].append(variable)
            RXN_metric_value = metrics[end_metric_selection]
            if (RXN_EM_Operator_sel == '<=' and RXN_metric_value <= end_metric_value_upper) or (
                    RXN_EM_Operator_sel == '>=' and RXN_metric_value >= end_metric_value_lower):
                test_interval = 1
            if end_metric_selection != '% EHC':
                if currentPID == PID_list[-1]:
                    process_queue.put(round((end_metric_value / RXN_metric_value) * 100), 2)
                if (RXN_EM_Operator_sel == '<=' and RXN_metric_value <= end_metric_value) or (
                        RXN_EM_Operator_sel == '>=' and RXN_metric_value >= end_metric_value):
                    comp_primary = tuple(composition)
                    byproducts_primary = byproducts
                    if not RXN_EM_2_Active_status:
                        running = False
                    else:
                        in_primary = False
                        for species in composition_sec:
                            composition.append(species)
                        test_interval = 50

    while running:
        test_count += 1
        start = time.time()
        conditions_met = False
        outer_weights = [len(groups[0]) for groups in composition]
        while conditions_met is False:
            a_index, b_index = random.choices(range(len(composition)), weights=outer_weights)[0], random.choices(range(len(composition)), weights=outer_weights)[0]
            a_i_index, b_i_index = random.randint(0, len(composition[a_index][0]) - 1), random.randint(0, len(composition[b_index][0]) - 1)
            a_group, b_group = composition[a_index][0][a_i_index][0], composition[b_index][0][b_i_index][0]
            groups = [(a_index, a_i_index, a_group), (b_index, b_i_index, b_group)]
            if groups[0][0] == groups[1][0] or check_react(groups) is False:
                conditions_met = False
            else:
                groups_pre_sort = groups
                groups = [groups[0][2], groups[1][2]]
                groups.sort()
                random_number = random.randint(1, 100)
                if groups in k_groups and random_number <= ks_mod[k_groups.index(groups)]:
                    groups = groups_pre_sort
                    conditions_met = True
            if time.time() - start > 10:
                running = False
                return "Metric Error"
        update_comp(composition, groups)


    if comp_secondary is None:
        return {"comp_primary": comp_primary, "in_situ_values": in_situ_values,
                'byproducts_primary': byproducts_primary}
    else:
        return {"comp_primary": comp_primary, "comp_secondary": comp_secondary, "in_situ_values": in_situ_values,
                'in_situ_values_sec': in_situ_values_sec, 'byproducts_primary': byproducts_primary,
                'byproducts_secondary': byproducts_secondary}


def RXN_Results(primary_comp_summary, byproducts_primary, in_situ_values):
    global rxn_summary_df, Xn, total_samples, byproducts_df, metrics
    comp_summary = collections.Counter(
        [(tuple(tuple(i) for i in sublist[0]), tuple(tuple(i) for i in sublist[1]), sublist[2][0]) for sublist in
         primary_comp_summary])
    sum_comp = sum([comp_summary[key] * key[2] for key in comp_summary])
    RS = []
    for key in comp_summary:
        RS.append((key[0], key[1], key[2], comp_summary[key]))
    RS = [[[list(x) for x in i[0]], [list(y) for y in i[1]], i[2], i[3]] for i in RS]
    for key in RS:
        amine_ct, NH2_ct, NH_ct, N_ct, acid_ct, alcohol_ct, epoxide_ct, EHC_ct, IV_ct, totalCl_ct, POH_ct, SOH_ct = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        key_names = [i[0] for i in key[0]]
        totalCl_ct += key_names.count('Cl') * key[3]
        NH2_ct += key_names.count('NH2') * key[3] + key_names.count('α_NH2') * key[3]
        NH_ct += key_names.count('NH') * key[3]
        N_ct += key_names.count('N') * key[3]
        amine_ct = NH2_ct + NH_ct + N_ct
        acid_ct += key_names.count('COOH') * key[3]
        POH_ct += key_names.count('POH') * key[3]
        SOH_ct += key_names.count('SOH') * key[3]
        alcohol_ct = POH_ct + SOH_ct
        epoxide_ct += key_names.count('COC') * key[3]
        IV_ct += (key_names.count('aB_unsat') + key_names.count('CC_3') + key_names.count('CC_2') + key_names.count(
            'CC_1')) * key[3]
        Cl_ct = key_names.count('Cl')
        for group in key[0]:
            if group[0] == 'POH' or group[0] == 'SOH':
                if 'Epi' in sn_dict and group[0] == 'SOH' and group[1] == sn_dict['Epi'].cprgID[1] and totalCl_ct > 0:
                    Cl_ct -= 1
                    EHC_ct += key[3]

        key.append(round((amine_ct * 56100) / sum_comp, 2))
        key.append(round((NH2_ct * 56100) / sum_comp, 2))
        key.append(round((NH_ct * 56100) / sum_comp, 2))
        key.append(round((N_ct * 56100) / sum_comp, 2))
        key.append(round((acid_ct * 56100) / sum_comp, 2))
        key.append(round((alcohol_ct * 56100) / sum_comp, 2))
        key.append(round((epoxide_ct * 56100) / sum_comp, 2))
        key.append(round((EHC_ct * 35.453) / sum_comp * 100, 2))
        key.append(round(((IV_ct * 2) * (127 / sum_comp) * 100), 2))
        key.append(round((totalCl_ct * 35.453) / sum_comp * 100, 2))
        key.append(round((POH_ct * 56100) / sum_comp, 2))
        key.append(round((SOH_ct * 56100) / sum_comp, 2))

    for key in RS:
        index = 0
        for group in key[1]:
            new_name = group[0] + '(' + str(group[1]) + ')'
            key[1][index] = new_name
            index += 1
        key[1] = '_'.join(key[1])
    rxn_summary_df = pandas.DataFrame(RS, columns=['Groups', 'Name', 'MW', 'Count', 'TAV', "1° TAV", "2° TAV", "3° TAV",
                                                   'AV', 'OH', 'COC', 'EHC,%', 'IV', 'Cl, %', "1° OH", "2° OH"])
    rxn_summary_df['MW'] = round(rxn_summary_df['MW'], 4)
    rxn_summary_df.drop(columns=['Groups'], inplace=True)
    rxn_summary_df.set_index('Name', inplace=True)
    rxn_summary_df.sort_values(by=['MW'], ascending=True, inplace=True)
    rxn_summary_df['Mass'] = rxn_summary_df['MW'] * rxn_summary_df['Count']
    rxn_summary_df['Mol,%'] = round(rxn_summary_df['Count'] / rxn_summary_df['Count'].sum() * 100, 4)
    rxn_summary_df['Wt,%'] = round(rxn_summary_df['Mass'] / rxn_summary_df['Mass'].sum() * 100, 4)
    rxn_summary_df['p*Mw'] = rxn_summary_df['MW'] * (rxn_summary_df['Wt,%'] / 100)
    rxn_summary_df['p*Mw2'] = (rxn_summary_df['MW'] ** 2) * (rxn_summary_df['Wt,%'] / 100)
    rxn_summary_df['p*Count'] = rxn_summary_df['Count'] * (rxn_summary_df['Wt,%'] / 100)
    rxn_summary_df['p*Count2'] = (rxn_summary_df['Count'] ** 2) * (rxn_summary_df['Wt,%'] / 100)
    Mw_variance = ((rxn_summary_df['p*Mw2'].sum()) - (rxn_summary_df['p*Mw'].sum() ** 2))
    Mn_variance = ((rxn_summary_df['p*Count2'].sum()) - (rxn_summary_df['p*Count'].sum() ** 2))

    rxn_summary_df['XiMi'] = (rxn_summary_df['Mol,%']/100) * rxn_summary_df['MW']
    sumNiMi2 = (rxn_summary_df['Count'] * (rxn_summary_df['MW']) ** 2).sum()
    sumNiMi3 = (rxn_summary_df['Count'] * (rxn_summary_df['MW']) ** 3).sum()
    sumNiMi4 = (rxn_summary_df['Count'] * (rxn_summary_df['MW']) ** 4).sum()

    Mn = rxn_summary_df['XiMi'].sum()
    Mw = rxn_summary_df['p*Mw'].sum()
    PDI = Mw / Mn
    Mz = sumNiMi3 / sumNiMi2
    Mz1 = sumNiMi4 / sumNiMi3
    DOP = total_ct / rxn_summary_df['Count'].sum()


    rxn_summary_df = rxn_summary_df[
        ['Count', 'Mass', 'Mol,%', 'Wt,%', 'MW', 'TAV', '1° TAV', '2° TAV', '3° TAV', 'AV', 'OH', 'COC', 'EHC,%', 'IV',
         'Cl, %', "1° OH", "2° OH"]]
    rxn_summary_df['Mass'] = rxn_summary_df['Mass'] / total_samples
    rxn_mass = rxn_summary_df['Mass'].sum()
    rxn_summary_df.loc['Sum'] = round(rxn_summary_df.sum(), 3)
    rxn_summary_df = rxn_summary_df.groupby(['MW', 'Name']).sum()

    Mw_std_dev = math.sqrt(Mw_variance)
    Mw_std_err = Mw_std_dev / math.sqrt(total_samples)
    Mw_low_95, Mw_high_95 = Mw - (1.96 * Mw_std_err), Mw + (1.96 * Mw_std_err)

    Mn_std_dev = math.sqrt(Mn_variance)
    Mn_std_err = Mn_std_dev / math.sqrt(total_samples)
    Mn_low_95, Mn_high_95 = Mn - (1.96 * Mn_std_err), Mn + (1.96 * Mn_std_err)

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
    WD.entries[11].insert(0, round(DOP, 4))
    WD.entries[12].delete(0, tkinter.END)
    WD.entries[12].insert(0, round(Mn_low_95, 4))
    WD.entries[13].delete(0, tkinter.END)
    WD.entries[13].insert(0, round(Mw_low_95, 4))
    WD.entries[18].delete(0, tkinter.END)
    WD.entries[18].insert(0, round(Mn_high_95, 4))
    WD.entries[19].delete(0, tkinter.END)
    WD.entries[19].insert(0, round(Mw_high_95, 4))

    byproducts_df = pandas.DataFrame(byproducts_primary, columns=['Name', 'Mass'])
    byproducts_df.set_index('Name', inplace=True)
    byproducts_df['Mass'] = byproducts_df['Mass'] / total_samples
    byproducts_df['Wt, % (Of byproducts)'] = round(byproducts_df['Mass'] / byproducts_df['Mass'].sum() * 100, 4)
    byproducts_df['Wt, % (Of Final)'] = round(byproducts_df['Mass'] / (rxn_mass + byproducts_df['Mass']) * 100, 4)
    byproducts_df['Wt, % (Of Initial)'] = round(byproducts_df['Mass'] / (starting_mass / total_samples) * 100, 4)

    Xn = pandas.DataFrame(in_situ_values[0], columns=['TAV'])
    Xn['AV'] = in_situ_values[1]
    Xn['OH'] = in_situ_values[2]
    Xn['COC'] = in_situ_values[3]
    Xn['EHC, %'] = in_situ_values[4]
    Xn['IV'] = in_situ_values[5]
    Xn['Mw'] = in_situ_values[6]
    Xn['Mn'] = in_situ_values[7]
    Xn['1° TAV'] = in_situ_values[8]
    Xn['2° TAV'] = in_situ_values[9]
    Xn['3° TAV'] = in_situ_values[10]
    Xn['Xn'] = in_situ_values[11]
    Xn['% Cl'] = in_situ_values[12]
    Xn['POH'] = in_situ_values[13]
    Xn['SOH'] = in_situ_values[14]
    Xn['P'] = -(1 / Xn['Xn']) + 1

    show_results(rxn_summary_df)
    show_byproducts(byproducts_df)
    show_Xn(Xn)


def RXN_Results_sec(secondary_comp_summary, byproducts_secondary, in_situ_values_sec):
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
        amine_ct, NH2_ct, NH_ct, N_ct, acid_ct, alcohol_ct, epoxide_ct, EHC_ct, IV_ct, totCl_ct, POH_ct, SOH_ct = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        key_names = [i[0] for i in key[0]]
        totCl_ct += key_names.count('Cl') * key[3]
        NH2_ct += key_names.count('NH2') * key[3] + key_names.count('α_NH2') * key[3]
        NH_ct += key_names.count('NH') * key[3]
        N_ct += key_names.count('N') * key[3]
        amine_ct = NH2_ct + NH_ct + N_ct
        acid_ct += key_names.count('COOH') * key[3]
        POH_ct += key_names.count('POH') * key[3]
        SOH_ct += key_names.count('SOH') * key[3]
        alcohol_ct = POH_ct + SOH_ct
        epoxide_ct += key_names.count('COC') * key[3]
        IV_ct += (key_names.count('aB_unsat') + key_names.count('CC_3') + key_names.count('CC_2') + key_names.count(
            'CC_1')) * key[3]
        Cl_ct = key_names.count('Cl')
        for group in key[0]:
            if group[0] == 'POH' or group[0] == 'SOH':
                if 'Epi' in sn_dict and group[0] == 'SOH' and group[1] == sn_dict['Epi'].cprgID[1] and totCl_ct > 0:
                    Cl_ct -= 1
                    EHC_ct += key[3]
        key.append(round((amine_ct * 56100) / sum_comp_2, 2))
        key.append(round((NH2_ct * 56100) / sum_comp_2, 2))
        key.append(round((NH_ct * 56100) / sum_comp_2, 2))
        key.append(round((N_ct * 56100) / sum_comp_2, 2))
        key.append(round((acid_ct * 56100) / sum_comp_2, 2))
        key.append(round((alcohol_ct * 56100) / sum_comp_2, 2))
        key.append(round((epoxide_ct * 56100) / sum_comp_2, 2))
        key.append(round((EHC_ct * 35.453) / sum_comp_2 * 100, 2))
        key.append(round(((IV_ct * 2) * (127 / sum_comp_2) * 100), 2))
        key.append(round((totCl_ct * 35.453) / sum_comp_2 * 100, 2))
        key.append(round((POH_ct * 56100) / sum_comp_2, 2))
        key.append(round((SOH_ct * 56100) / sum_comp_2, 2))

    for key in RS_2:
        index = 0
        for group in key[1]:
            new_name = group[0] + '(' + str(group[1]) + ')'
            key[1][index] = new_name
            index += 1
        key[1] = '_'.join(key[1])
    rxn_summary_df_2 = pandas.DataFrame(RS_2,
                                        columns=['Groups', 'Name', 'MW', 'Count', 'TAV', "1° TAV", "2° TAV", "3° TAV",
                                                 'AV', 'OH', 'COC', 'EHC,%', 'IV', 'Cl, %', "1° OH", "2° OH"])
    rxn_summary_df_2['MW'] = round(rxn_summary_df_2['MW'], 4)
    rxn_summary_df_2.drop(columns=['Groups'], inplace=True)
    rxn_summary_df_2.set_index('Name', inplace=True)
    rxn_summary_df_2.sort_values(by=['MW'], ascending=True, inplace=True)
    rxn_summary_df_2['Mass'] = rxn_summary_df_2['MW'] * rxn_summary_df_2['Count']
    rxn_summary_df_2['Mol,%'] = round(rxn_summary_df_2['Count'] / rxn_summary_df_2['Count'].sum() * 100, 4)
    rxn_summary_df_2['Wt,%'] = round(rxn_summary_df_2['Mass'] / rxn_summary_df_2['Mass'].sum() * 100, 4)
    rxn_summary_df_2['p*Mw'] = rxn_summary_df_2['MW'] * (rxn_summary_df_2['Wt,%'] / 100)
    rxn_summary_df_2['p*Mw2'] = (rxn_summary_df_2['MW'] ** 2) * (rxn_summary_df_2['Wt,%'] / 100)
    rxn_summary_df_2['p*Count'] = rxn_summary_df_2['Count'] * (rxn_summary_df_2['Wt,%'] / 100)
    rxn_summary_df_2['p*Count2'] = (rxn_summary_df_2['Count'] ** 2) * (rxn_summary_df_2['Wt,%'] / 100)
    rxn_summary_df_2['XiMi'] = (rxn_summary_df_2['Mol,%'] / 100) * rxn_summary_df_2['MW']

    Mw_variance = ((rxn_summary_df_2['p*Mw2'].sum()) - (rxn_summary_df_2['p*Mw'].sum() ** 2))
    Mn_variance = ((rxn_summary_df_2['p*Count2'].sum()) - (rxn_summary_df_2['p*Count'].sum() ** 2))

    sumNiMi2 = (rxn_summary_df_2['Count'] * (rxn_summary_df_2['MW']) ** 2).sum()
    sumNiMi3 = (rxn_summary_df_2['Count'] * (rxn_summary_df_2['MW']) ** 3).sum()
    sumNiMi4 = (rxn_summary_df_2['Count'] * (rxn_summary_df_2['MW']) ** 4).sum()

    Mn = rxn_summary_df_2['XiMi'].sum()
    Mw = rxn_summary_df_2['p*Mw'].sum()
    PDI = Mw / Mn
    Mz = sumNiMi3 / sumNiMi2
    Mz1 = sumNiMi4 / sumNiMi3
    DOP = total_ct_sec / rxn_summary_df_2['Count'].sum()

    rxn_summary_df_2 = rxn_summary_df_2[
        ['Count', 'Mass', 'Mol,%', 'Wt,%', 'MW', 'TAV', '1° TAV', '2° TAV', '3° TAV', 'AV', 'OH', 'COC', 'EHC,%', 'IV',
         'Cl, %', "1° OH", "2° OH"]]
    rxn_summary_df_2['Mass'] = rxn_summary_df_2['Mass'] / total_samples
    rxn_mass = rxn_summary_df_2['Mass'].sum()
    rxn_summary_df_2.loc['Sum'] = round(rxn_summary_df_2.sum(), 3)
    rxn_summary_df_2 = rxn_summary_df_2.groupby(['MW', 'Name']).sum()



    Mw_std_dev = math.sqrt(Mw_variance)
    Mw_std_err = Mw_std_dev / math.sqrt(total_samples)
    Mw_low_95, Mw_high_95 = Mw - (1.96 * Mw_std_err), Mw + (1.96 * Mw_std_err)

    Mn_std_dev = math.sqrt(Mn_variance)
    Mn_std_err = Mn_std_dev / math.sqrt(total_samples)
    Mn_low_95, Mn_high_95 = Mn - (1.96 * Mn_std_err), Mn + (1.96 * Mn_std_err)

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
    WD2.entries[11].insert(0, round(DOP, 4))
    WD2.entries[12].delete(0, tkinter.END)
    WD2.entries[12].insert(0, round(Mn_low_95, 4))
    WD2.entries[13].delete(0, tkinter.END)
    WD2.entries[13].insert(0, round(Mw_low_95, 4))
    WD2.entries[18].delete(0, tkinter.END)
    WD2.entries[18].insert(0, round(Mn_high_95, 4))
    WD2.entries[19].delete(0, tkinter.END)
    WD2.entries[19].insert(0, round(Mw_high_95, 4))

    byproducts_df_2 = pandas.DataFrame(byproducts_secondary, columns=['Name', 'Mass'])
    byproducts_df_2.set_index('Name', inplace=True)
    byproducts_df_2['Mass'] = byproducts_df_2['Mass'] / total_samples
    byproducts_df_2['Wt, % (Of byproducts)'] = round(byproducts_df_2['Mass'] / byproducts_df_2['Mass'].sum() * 100, 4)
    byproducts_df_2['Wt, % (Of Final)'] = round(byproducts_df_2['Mass'] / (rxn_mass + byproducts_df_2['Mass']) * 100, 4)
    byproducts_df_2['Wt, % (Of Initial)'] = round(byproducts_df_2['Mass'] / (starting_mass_sec / total_samples) * 100, 4)

    Xn_2 = pandas.DataFrame(in_situ_values_sec[0], columns=['TAV'])
    Xn_2['AV'] = in_situ_values_sec[1]
    Xn_2['OH'] = in_situ_values_sec[2]
    Xn_2['COC'] = in_situ_values_sec[3]
    Xn_2['EHC, %'] = in_situ_values_sec[4]
    Xn_2['IV'] = in_situ_values_sec[5]
    Xn_2['Mw'] = in_situ_values_sec[6]
    Xn_2['Mn'] = in_situ_values_sec[7]
    Xn_2['1° TAV'] = in_situ_values_sec[8]
    Xn_2['2° TAV'] = in_situ_values_sec[9]
    Xn_2['3° TAV'] = in_situ_values_sec[10]
    Xn_2['Xn'] = in_situ_values_sec[11]
    Xn_2['% Cl'] = in_situ_values_sec[12]
    Xn_2['POH'] = in_situ_values_sec[13]
    Xn_2['SOH'] = in_situ_values_sec[14]
    Xn_2['P'] = -(1 / Xn_2['Xn']) + 1

    show_results_sec(rxn_summary_df_2)
    show_byproducts_sec(byproducts_df_2)
    show_Xn_sec(Xn_2)


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
        starting_materials_sec, total_samples, RXN_EM_Operator, RXN_EM_Operator_2, RXN_EM_Operator_sel, RXN_EM_Operator_2_sel, ks_mod, k_groups
    starting_mass, starting_mass_sec, total_ct, total_ct_sec = 0, 0, 0, 0
    row_for_sec = RXN_EM_Entry_2_SR.current()
    try:
        end_metric_value = float(RXN_EM_Value.get())
        RXN_EM_2_Active_status = RXN_EM_2_Active.get()
        RXN_EM_Operator_sel = RXN_EM_Operator.get()
        RXN_EM_Operator_2_sel = RXN_EM_Operator_2.get()
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
            elif RET.entries[cell].get() != "" and RET.entries[cell + 1].get() == "":
                messagebox.showinfo("Error", "Please enter a mass for each reactant")
                return "Error"
            else:
                break
        start = 3
        ks = []
        k_groups = []
        for i in range(CT.tableheight - 1):
            if CT.entries[start].get() != "" and CT.entries[start + 1].get() != "" and CT.entries[start + 2].get() != "":
                ks.append(float(CT.entries[start + 2].get()))
                k_groups.append([CT.entries[start].get(), CT.entries[start + 1].get()])
            elif CT.entries[start].get() != "" and CT.entries[start + 1].get() != "" and CT.entries[start + 2].get() == "":
                messagebox.showinfo("Error", "Please enter valid k values")
                return "Error"
            start += CT.tablewidth
        for i in range(len(k_groups)):
            for j in range(len(k_groups[i])):
                k_groups[i][j] = k_groups[i][j].replace("₂", "2")
        max_value = max(ks)
        ks_mod = [(int((x / max_value) * 100)) for x in ks]
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
            results = [executor.submit(simulate, starting_materials, starting_materials_sec, end_metric_value,
                                       end_metric_selection, end_metric_value_sec, end_metric_selection_sec, sn_dict,
                                       RXN_EM_2_Active_status, total_ct, total_ct_sec, workers, progress_queue,
                                       PID_list, progress_queue_sec, RXN_EM_Operator_sel, RXN_EM_Operator_2_sel, ks_mod, k_groups) for _
                       in range(workers)]
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
    if 'comp_secondary' in results[0].result():
        in_situ_values_sec = results[0].result()['in_situ_values_sec']
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
        RXN_Results(primary_comp_summary, byproducts_primary, in_situ_values)
        RXN_Results_sec(secondary_comp_summary, byproducts_secondary, in_situ_values_sec)
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
        RXN_Results(primary_comp_summary, byproducts_primary, in_situ_values)


def reset_entry_table():
    for i in range(RET.tableheight - 1):
        for j in range(RET.tablewidth):
            RET.entries[(i + 1) * RET.tablewidth + j].configure(state='normal')
            RET.entries[(i + 1) * RET.tablewidth + j].delete(0, 'end')
    sim.progress['value'] = 0
    sim.progress_2['value'] = 0
    RXN_EM_2_Active.set(False)
    RXN_EM_Entry_2_SR.set("2° Start")
    RXN_EM_Value_2.delete(0, 'end')
    RXN_EM_Entry_2.insert(0, "")
    RXN_EM_Entry_2.set("2° End Metric")
    RXN_EM_Entry.set("1° End Metric")
    RXN_EM_Value.delete(0, 'end')
    RXN_EM_Entry.insert(0, "")

    for i in range(len(CT.entries)):
        if i > 2:
            CT.entries[i].config(state='normal')
            CT.entries[i].delete(0, tkinter.END)


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


# -------------------------------------------------APC Functions----------------------------------------------------------#
def run_APC(rxn_summary_df, label):
    APCParametersWindow(window)
    if apc_params[0] != "" and apc_params[1] != "" and apc_params[2] != "" and apc_params[3] != "":
        APC_Flow_Rate = apc_params[0]
        APC_FWHM = apc_params[1]
        APC_FWHM2 = apc_params[2]
        APC_temp = apc_params[3]
        APC_Solvent = apc_params[4]
        APC(APC_Flow_Rate, APC_FWHM, APC_FWHM2, APC_temp, APC_Solvent, rxn_summary_df, label)
    else:
        messagebox.showerror("Error", "Please enter valid parameters for flow rate, FWHM, and temp.")


def APC(APC_Flow_Rate, APC_FWHM, APC_FWHM2, APC_temp, APC_Solvent, rxn_summary_df, label):
    global APC_df, apc_params
    APC_comp = rxn_summary_df
    APC_comp = APC_comp.reset_index()
    APC_comp = APC_comp[['MW', 'Wt,%', 'Name']]
    APC_comp = APC_comp[:-1]
    if str(APC_temp) == "35" and APC_Solvent == "THF":
        STD_Equation_params = np.array([0.0236, -0.6399, 6.5554, -31.7505, 71.8922, -56.3224])
        Low_MW_Equation_params = np.array([-1.4443, 13.5759, -48.4406, 76.9940, -40.3772])
        High_MW_Equation_params = np.array([0.1411, -1.6901, 7.5488])
    elif str(APC_temp) == "55" and APC_Solvent == "THF":
        STD_Equation_params = np.array([0.0264, -0.7016, 7.0829, -33.9565, 76.4028, -59.9091])
        Low_MW_Equation_params = np.array([-1.9097, 18.0005, -64.1695, 101.7605, -54.9071])
        High_MW_Equation_params = np.array([0.1417, -1.7016, 7.5978])
    elif APC_Solvent == "Water":
        STD_Equation_params = np.array([-0.08236, 1.3094, -7.7095, 20.5979, -26.0739, 24.0511])
    APC_comp['Log(MW)'] = np.log10(APC_comp['MW'])
    APC_comp.loc[:, 'FWHM(min)'] = 0.000
    APC_comp.loc[:, 'RT(min)'] = 0.000

    if APC_Solvent == "THF":
        MIN_MW = np.log10(621)
        MAX_MW = np.log10(290000)
    elif APC_Solvent == "Water":
        MIN_MW = np.log10(1)
        MAX_MW = np.log10(1000000)

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
    if APC_Solvent == "THF":
        APC_rows = list(range(0, 120001))
    elif APC_Solvent == "Water":
        APC_rows = list(range(12000, 25001))
    APC_df = pandas.DataFrame(0, index=APC_rows, columns=APC_columns)
    APC_df.index.name = 'Time'
    APC_df.index = APC_df.index * 0.001

    def calc_apc_matrix(APC_comp, APC_df):
        times = APC_df.index.values
        apc_values = np.zeros((len(times), len(APC_comp)))
        for i, row in APC_comp.iterrows():
            apc_values[:, i] = np.exp(-0.5 * np.power((times - row['RT(min)']) / (row['FWHM(min)'] / 2.35), 2)) * (
                    row['Wt,%'] * 0.5)
        return pandas.DataFrame(apc_values, index=times, columns=APC_comp['Name'])

    APC_df = calc_apc_matrix(APC_comp, APC_df)

    APC_df['Sum'] = APC_df.sum(axis=1)
    show_APC(APC_Flow_Rate, APC_FWHM, APC_FWHM2, APC_Solvent, APC_comp, APC_df, label)


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
                for column in range(4):
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
            df.to_excel(writer, sheet_name='1_Aux', index=False, header=False, startrow=0, startcol=11)

            ks_mod_df = pandas.DataFrame(ks_mod)
            k_groups_df = pandas.DataFrame(k_groups)
            ks_mod_df.columns = ['Reactivity (0 - 100)']
            k_groups_df.columns = ['Group A', 'Group B']
            ks_mod_df.to_excel(writer, sheet_name='1_Aux', index=False, header=True, startcol=6)
            k_groups_df.to_excel(writer, sheet_name='1_Aux', index=False, header=True, startcol=7)

            workbook = writer.book
            worksheet = workbook['1_Aux']
            worksheet.sheet_state = 'visible'
            columns_to_adjust = ['A', 'G', 'L']
            for column in columns_to_adjust:
                worksheet.column_dimensions[column].width = 20

            APC_selected_columns = APC_df.iloc[:, -1]
            APC_selected_columns.to_excel(writer, sheet_name='1_APC', index=True)


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

            ks_mod_df = pandas.DataFrame(ks_mod)
            k_groups_df = pandas.DataFrame(k_groups)
            ks_mod_df.columns = ['Reactivity (0 - 100)']
            k_groups_df.columns = ['Group A', 'Group B']
            ks_mod_df.to_excel(writer, sheet_name='2_Aux', index=False, header=True, startcol=6)
            k_groups_df.to_excel(writer, sheet_name='2_Aux', index=False, header=True, startcol=7)

            workbook = writer.book
            worksheet = workbook['2_Aux']
            worksheet.sheet_state = 'visible'
            columns_to_adjust = ['A', 'G', 'L']
            for column in columns_to_adjust:
                worksheet.column_dimensions[column].width = 20

            APC_selected_columns = APC_df.iloc[:, -1]
            APC_selected_columns.to_excel(writer, sheet_name='1_APC', index=True)


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

            ks_mod_df = pandas.DataFrame(ks_mod)
            k_groups_df = pandas.DataFrame(k_groups)
            ks_mod_df.columns = ['Reactivity (0 - 100)']
            k_groups_df.columns = ['Group A', 'Group B']
            ks_mod_df.to_excel(writer, sheet_name='1_Aux', index=False, header=True, startcol=6)
            k_groups_df.to_excel(writer, sheet_name='1_Aux', index=False, header=True, startcol=7)

            workbook = writer.book
            worksheet = workbook['1_Aux']
            worksheet.sheet_state = 'visible'
            columns_to_adjust = ['A', 'G', 'L']
            for column in columns_to_adjust:
                worksheet.column_dimensions[column].width = 20

            APC_selected_columns = APC_df.iloc[:, -1]
            APC_selected_columns.to_excel(writer, sheet_name='1_APC', index=True)

            workbook = writer.book
            sheet_names = ['1_Summary', '1_In_Situ', '1_Aux', '2_Summary', '2_In_Situ', '2_Aux', '1_APC']
            workbook._sheets.sort(key=lambda ws: sheet_names.index(ws.title))

# -------------------------------------------Display Results---------------------------------------------------#
def show_results(rxn_summary_df):
    global results_table, frame_results
    try:
        results_table.destroy()
        frame_results.destroy()
    except NameError:
        pass
    frame_results = tkinter.Frame(tab2)
    frame_results.place(relx=0.5, rely=0.6, anchor='center', relwidth=0.8, relheight=0.6)
    results_table = Table(frame_results, dataframe=rxn_summary_df, showtoolbar=True, showstatusbar=True, showindex=True, align='center')
    results_table.show()

def show_results_sec(rxn_summary_df_2):
    global results_table_2, frame_results_2
    try:
        results_table_2.destroy()
        frame_results_2.destroy()
    except NameError:
        pass
    frame_results_2 = tkinter.Frame(tab4)
    frame_results_2.place(relx=0.5, rely=0.6, anchor='center', relwidth=0.8, relheight=0.6)
    results_table_2 = Table(frame_results_2, dataframe=rxn_summary_df_2, showtoolbar=True, showstatusbar=True, showindex=True, align='center', )
    results_table_2.show()


def show_byproducts(byproducts_df):
    global byproducts_table, frame_byproducts
    try:
        byproducts_table.destroy()
        frame_byproducts.destroy()
    except NameError:
        pass
    frame_byproducts = tkinter.Frame(tab2)
    frame_byproducts.place(relx=0.5, rely=0.15, anchor='center', relwidth=0.3, relheight=0.15)
    byproducts_table = Table(frame_byproducts, dataframe=byproducts_df, showtoolbar=False, showstatusbar=True, showindex=True, align='center', maxcellwidth=1000)
    byproducts_table.show()


def show_byproducts_sec(byproducts_df_2):
    global byproducts_table_2, frame_byproducts_2
    try:
        byproducts_table_2.destroy()
        frame_byproducts_2.destroy()
    except NameError:
        pass
    frame_byproducts_2 = tkinter.Frame(tab4)
    frame_byproducts_2.place(relx=0.5, rely=0.15, anchor='center', relwidth=0.3, relheight=0.15)
    byproducts_table_2 = Table(frame_byproducts_2, dataframe=byproducts_df_2, showtoolbar=False, showstatusbar=True, showindex=True, align='center', maxcellwidth=1000)
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


def show_APC(APC_Flow_Rate, APC_FWHM, APC_FWHM2, APC_Solvent, APC_comp, APC_df, label):
    global start_time, end_time, dragging_start, dragging_end
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(APC_df.index, APC_df['Sum'], linewidth=0.75)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Intensity (a.u.)')
    ax.set_title(f'{label} Theoretical APC')
    if APC_Solvent == "THF":
        ax.set_xticks(np.arange(0, 12.25, 0.25))
        ax.set_xticklabels(np.arange(0, 12.25, 0.25), rotation=-35)
        ax.set_xlim(0, 12)
        start_time = 1.0
        end_time = 11.0
    elif APC_Solvent == "Water":
        ax.set_xticks(np.arange(12, 25.25, 0.25))
        ax.set_xticklabels(np.arange(12, 25.25, 0.25), rotation=-35)
        ax.set_xlim(12, 25)
        start_time = 13.0
        end_time = 24.0

    textstr = f'Solvent: {APC_Solvent} \nFlow Rate: {APC_Flow_Rate:.1f} ml/min\nFWHM (100 MW): {APC_FWHM:.3f} min\nFWHM (100K MW): {APC_FWHM2:.3f} min'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    a = ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    time_textstr = f'Time, min: '
    b = ax.text(0.05, 0.7, time_textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ingegrated_area = f'Integrated Area, %: '
    c = ax.text(0.05, 0.6, ingegrated_area, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    # Create vertical bars for start and end times
    start_time_line = ax.axvline(start_time, color='blue', linestyle='-', linewidth=1, picker=True)
    end_time_line = ax.axvline(end_time, color='red', linestyle='-', linewidth=1, picker=True)

    fig.canvas.draw()

    def on_move(event):
        xdata = event.xdata
        if xdata is not None:
            closest_peaks = APC_comp.iloc[(APC_comp['RT(min)'] - xdata).abs().argsort()[:5]]
            tolerence = 0.05
            peak_info = []
            for i, peak in closest_peaks.iterrows():
                if abs(peak['RT(min)'] - xdata) <= tolerence:
                    peak_info.append(
                        [peak['Name'], f"{peak['RT(min)']:.3f}", f"{10 ** peak['Log(MW)']:.2f}", f"{peak['Wt,%']:.2f}"])
            table.delete(*table.get_children())
            for info in peak_info:
                table.insert("", "end", values=info)
            b.set_text(f'Time, min: {xdata:.3f}')
        else:
            table.delete(*table.get_children())
            b.set_text(f'Time, min: ')
        fig.canvas.draw()

    def calculate_area():
        global start_time, end_time

        x = np.array(APC_df.index)
        y = np.array(APC_df['Sum'])

        # Filter data based on the time range
        x_times = sorted([start_time, end_time])
        indicies = np.where((x > x_times[0]) & (x < x_times[1]))

        x_range = x[indicies]
        y_range = y[indicies]

        area_range = simps(y_range, x_range)
        total_area = simps(y, x)
        area_percent = area_range / total_area * 100
        c.set_text(f'Integrated Area, %: {area_percent:.2f}')

        fig.canvas.draw()

    fig.canvas.mpl_connect('motion_notify_event', lambda event: on_move(event))

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

    # Variables to track the dragging state
    dragging_start = False
    dragging_end = False

    # Function to handle mouse button press events
    def on_press(event):
        global dragging_start, dragging_end
        if event.button == 1:
            if start_time_line.contains(event)[0]:
                dragging_start = True
            elif end_time_line.contains(event)[0]:
                dragging_end = True

    # Function to handle mouse button release events
    def on_release(event):
        global dragging_start, dragging_end
        if event.button == 1:
            dragging_start = False
            dragging_end = False

    # Function to handle mouse motion events (dragging)
    def on_motion(event):
        global start_time, end_time
        if dragging_start == False and dragging_end == False:
            return
        elif dragging_start:
            start_time = event.xdata
            start_time_line.set_xdata([start_time, start_time])
        elif dragging_end:
            end_time = event.xdata
            end_time_line.set_xdata([end_time, end_time])
        calculate_area()

        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    root.mainloop()

# ---------------------------------------------------User-Interface----------------------------------------------#
if __name__ == "__main__":
    window = tkinter.Tk()
    style = ttk.Style(window)
    # Set theme based on the operating system
    if platform.system() == "Darwin":  # macOS
        style.theme_use('clam')  # or another theme that looks good on macOS
    else:  # Assume Windows or Linux
        style.theme_use('clam')  # Choose a different theme for Windows/Linux
    style.configure('TNotebook.Tab', background='#355C7D', foreground='#ffffff')
    style.configure("red.Horizontal.TProgressbar", troughcolor='green')
    style.map('TNotebook.Tab', background=[('selected', 'green3')], foreground=[('selected', '#000000')])
    # sv_ttk.set_theme("dark")
    window.iconbitmap("testtube.ico")
    window.title("Monte Karlo")
    window.configure(background="#000000")

    tab_control = ttk.Notebook(window)
    tab1 = ttk.Frame(tab_control, style='TNotebook.Tab')
    tab2 = ttk.Frame(tab_control, style='TNotebook.Tab')
    tab3 = ttk.Frame(tab_control, style='TNotebook.Tab')
    tab4 = ttk.Frame(tab_control, style='TNotebook.Tab')
    tab5 = ttk.Frame(tab_control, style='TNotebook.Tab')
    tab6 = ttk.Frame(tab_control, style='TNotebook.Tab')
    tab_control.add(tab1, text='Reactor')
    tab_control.add(tab2, text='1° Reaction Results')
    tab_control.add(tab3, text='1° In-Situ Results')
    tab_control.add(tab4, text='2° Reaction Results')
    tab_control.add(tab5, text='2° In-Situ Results')
    tab_control.add(tab6, text='Machine Learning')
    tkinter.Grid.rowconfigure(window, 0, weight=1)
    tkinter.Grid.columnconfigure(window, 0, weight=1)
    tab_control.grid(row=0, column=0, sticky=tkinter.E + tkinter.W + tkinter.N + tkinter.S)

    # Get the screen width and height
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    # Set the window size to a percentage of the screen size
    screen_size = 0.8  # 80% of the screen
    window_width = int(screen_width * screen_size)
    window_height = int(screen_height * screen_size)
    window.geometry(f"{window_width}x{window_height}+0+0")  # "+0+0" positions the window in the top-left corner


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
                       'R14Reactant', 'R15Reactant', 'R16Reactant', 'R17Reactant', 'R18Reactant', 'R19Reactant', ]
    Entry_masses = ['R1mass', 'R2mass', 'R3mass', 'R4mass', 'R5mass', 'R6mass', 'R7mass', 'R8mass', 'R9mass', 'R10mass',
                    'R11mass', 'R12mass', 'R13mass', 'R14mass', 'R15mass', 'R16mass', 'R17mass', 'R18mass', 'R19mass', ]
    RDE = ['R1Data', 'R2Data', 'R3Data', 'R4Data', 'R5Data', 'R6Data', 'R7Data', 'R8Data', 'R9Data', 'R10Data',
           'R11Data',
           'R12Data', 'R13Data', 'R14Data', 'R15Data', 'R16Data', 'R17Data', 'R18Data', 'R19Data', ]

    global starting_cell
    starting_cell = 16


    class APCParametersWindow(simpledialog.Dialog):
        def body(self, master):
            self.title("APC Parameters")
            self.iconbitmap("testtube.ico")

            Label(master, text="Flow Rate (ml/min):").grid(row=0, column=0)
            self.flow_rate = DoubleVar(value=0.6)  # Set default value to 1.0 ml/min
            self.e1 = Entry(master, width=18, textvariable=self.flow_rate)
            self.e1.grid(row=0, column=1)

            Label(master, text="100 MW FWHM (min):").grid(row=1, column=0)
            self.fwhm = DoubleVar(value=0.060)  # Set default FWHM for 100 MW
            self.e2 = Entry(master, width=18, textvariable=self.fwhm)
            self.e2.grid(row=1, column=1)

            Label(master, text="100K MW FWHM (min):").grid(row=2, column=0)
            self.fwhm2 = DoubleVar(value=0.1)  # Set default FWHM for 100K MW
            self.e3 = Entry(master, width=18, textvariable=self.fwhm2)
            self.e3.grid(row=2, column=1)

            Label(master, text="Temperature (°C):").grid(row=3, column=0)
            self.temp = IntVar(value=35)  # Set default temperature to 35°C
            self.temp_choices = [35, 55]
            self.combobox = ttk.Combobox(master, values=self.temp_choices, textvariable=self.temp)
            self.combobox.grid(row=3, column=1)
            self.combobox.config(width=15)

            Label(master, text="Solvent:").grid(row=4, column=0)
            self.solvent = StringVar(value="THF")  # Set default solvent to THF
            self.solvent_choices = ["THF", "Water"]
            self.combobox_solvent = ttk.Combobox(master, values=self.solvent_choices, textvariable=self.solvent)
            self.combobox_solvent.grid(row=4, column=1)
            self.combobox_solvent.config(width=15)

            self.flow_rate.trace('w', self.update_fwhm)
            self.solvent.trace('w', self.update_fwhm)

            return self.e1

        def update_fwhm(self, *args):
            try:
                flow_rate = self.flow_rate.get()
                if self.solvent.get() == "Water":
                    self.flow_rate.set(0.5)
                    self.fwhm.set(0.35)
                    self.fwhm2.set(0.35)
                elif self.solvent.get() == "THF":
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
            apc_params = [self.flow_rate.get(), self.fwhm.get(), self.fwhm2.get(), self.temp.get(), self.solvent.get()]
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
            self.cell_width = 10
            self.rel_width = 0.65
            self.rel_height = 0.5
            self.tablewidth = 16
            self.tableheight = 20
            self.entries = None
            self.place(relx=0.55, rely=0.5, anchor=CENTER, relwidth=self.rel_width, relheight=self.rel_height)  # Place the frame in the middle of the window
            self.create_table()

        def create_table(self):
            self.entries = {}
            counter = 0
            for row in range(self.tableheight):
                for column in range(self.tablewidth):
                    self.entries[counter] = tkinter.Entry(self)
                    self.entries[counter].grid(row=row, column=column, sticky="nsew")
                    self.entries[counter].config(justify="center")
                    counter += 1
            self.tabel_labels()

            for row in range(self.tableheight):
                tkinter.Grid.rowconfigure(self, row, weight=1)
            for column in range(self.tablewidth):
                tkinter.Grid.columnconfigure(self, column, weight=3)
                if column == 0:
                    self.columnconfigure(column, weight=1)

        def tabel_labels(self):
            labels = [
                "Component", "Mass (g)", "wt, %", "Moles", "1° - ID", "1° - K", "C1° - ID",
                "1° - Child K", "2° - ID", "2° - K", "C2° - ID", "2° - Child K", "3° - ID",
                "3° - K", "C3° - ID", "3° - Child K"]
            offset = 0
            for i, label in enumerate(labels):
                self.entries[offset + i].insert(0, label)
                self.entries[offset + i].config(state="readonly", font=("Helvetica", 8, "bold"))
            self.user_entry()

        def user_entry(self):
            cell = starting_cell
            for index in range(self.tableheight - 1):
                Entry_Reactants[index] = tkinter.StringVar()
                self.entries[cell] = AutocompleteCombobox(self, completevalues=Reactants, width=24,textvariable=Entry_Reactants[index])
                self.entries[cell].grid(row=index + 1, column=0, sticky="nsew")
                self.entries[cell].config(justify="center")
                Entry_masses[index] = self.entries[cell + 1]
                cell += self.tablewidth

        def update_table(self, index, cell):
            # If the next cell is empty or contains "0", clear it and enable the following cell
            if self.entries[cell + 1].get() in ("", "0"):
                self.entries[cell + 1].delete(0, tkinter.END)
                self.entries[cell + 2].config(state="normal")
                self.entries[cell + 2].delete(0, tkinter.END)

            # If the current cell is "Clear", clear subsequent cells and enable following cells
            if self.entries[cell].get() == "Clear":
                self.entries[cell].delete(0, tkinter.END)
                for i in range(1, 16):
                    self.entries[cell + i].config(state="normal")
                    self.entries[cell + i].delete(0, tkinter.END)

            else:
                if self.entries[cell].get() != "" and self.entries[cell + 1].get() != "":
                    a = str_to_class(Entry_Reactants[index].get())()
                    molesA = float(Entry_masses[index].get()) / float(a.mw)
                    self.entries[cell + 3].delete(0, tkinter.END)
                    self.entries[cell + 3].insert(0, str(round(molesA, 6)))
                    for i in range(5, 15, 2):
                        self.entries[cell + i].config(state="normal")

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
                        self.entries[cell + 1].insert(0,str(round((float(Entry_masses[index].get()) / sum_mass()) * 100, 3)))
                        self.entries[cell + 1].config(state="readonly")
                    cell = cell + self.tablewidth
                    index = index + 1

            weight_percent()

        def get_rgID_combinations(self):
            unique_rgIDs = set()
            cell = 20
            for i in range(self.tableheight - 1):
                unique_rgIDs.update(filter(lambda x: x not in (None, "", "None"), [self.entries[cell + j].get() for j in range(0, 12, 2)]))
                cell += self.tablewidth
            unique_combinations = set(itertools.combinations(sorted(unique_rgIDs), 2))
            reaction_combinations = set()
            for combination in unique_combinations:
                rg1, rg2 = combination
                rg1 = rg1.replace("₂", "2")
                rg2 = rg2.replace("₂", "2")
                combination_initial = combination
                if rg1 == rg2:
                    pass
                else:
                    rg1_attr = getattr(rg, rg1, None)
                    rg2_attr = getattr(rg, rg2, None)
                    if (rg1_attr is not None and rg2 in rg1_attr) or (rg2_attr is not None and rg1 in rg2_attr):
                        reaction_combinations.add(combination_initial)
            return list(reaction_combinations)

        def update_rates(self, index, cell):
            if self.entries[cell].get() != "Clear" and self.entries[cell].get() != "":
                a = str_to_class(Entry_Reactants[index].get())()
                attributes = ["prgID", "prgk", "cprgID", "cprgk", "srgID", "srgk",
                              "csrgID", "csrgk", "trgID", "trgk", "ctrgID", "ctrgk"]

                for i, attr in enumerate(attributes, start=4):
                    self.entries[cell + i].config(state="normal")
                    self.entries[cell + i].delete(0, tkinter.END)
                    self.entries[cell + i].insert(0, str(getattr(a, attr)))
                    self.entries[cell + i].config(state="readonly")
            else:
                pass

            combinations = self.get_rgID_combinations()

            # Update the combinations_table
            if CT:
                CT.display_combinations(combinations)


    class combinations_table(tkinter.Frame):
        def __init__(self, master=tab1):
            tkinter.Frame.__init__(self, master)
            self.tablewidth = 3
            self.tableheight = 20
            self.entries = None
            self.place(relx=0.15, rely=0.5, anchor=CENTER, relwidth=0.13, relheight=0.5)
            self.create_table()


        def create_table(self):
            self.entries = {}
            counter = 0
            for row in range(self.tableheight):
                for column in range(self.tablewidth):
                    entry = tkinter.Entry(self, justify="center")
                    entry.grid(row=row, column=column, sticky="nsew")
                    self.entries[counter] = entry
                    counter += 1
            self.tabel_labels()

            for row in range(self.tableheight):
                tkinter.Grid.rowconfigure(self, row, weight=1)
            for column in range(self.tablewidth):
                tkinter.Grid.columnconfigure(self, column, weight=3)

        def tabel_labels(self):
            labels = ["ID - 1", "ID - 2", "K"]
            for i, label in enumerate(labels):
                self.entries[i].insert(0, label)
                self.entries[i].config(state="readonly", font=("Helvetica", 8, "bold"))

        def display_combinations(self, combinations):
            for i in range(len(self.entries)):
                if i > 2:
                    self.entries[i].config(state='normal')
                    self.entries[i].delete(0, tkinter.END)
            for i in range(len(combinations)):
                index = (i + 1) * self.tablewidth + 2
                self.entries[index].insert(0, 1)
                for j in range(len(combinations[i])):
                    self.entries[(i + 1) * self.tablewidth + j].insert(0, combinations[i][j])
                    self.entries[(i + 1) * self.tablewidth + j].config(state="readonly")


    class DataFrameEditor(tkinter.Frame):
        def __init__(self, master=tab6):
            tkinter.Frame.__init__(self, master, background="#355C7D")

            # Initialize the DataFrame with 30 rows and placeholder column headers
            self.data = pandas.DataFrame("", index=range(30),
                                         columns=[f"Column {i + 1}" for i in range(14)])  # Placeholder headers

            # Create a frame for the pandastable
            self.table_frame = tkinter.Frame(self)
            self.table_frame.place(relx=0.5, rely=0.5, anchor='center', relwidth=0.9, relheight=0.8)

            # Initialize the pandastable with the DataFrame
            self.table = Table(self.table_frame, dataframe=self.data, showtoolbar=True, showstatusbar=True,
                               showindex=True)
            self.table.show()

            # Pack the main frame to ensure it's visible
            self.pack(fill=tkinter.BOTH, expand=True)

            # Create a single frame for the top row of buttons
            self.button_frame = tkinter.Frame(self, background="#355C7D")
            self.button_frame.pack(side=tkinter.TOP, fill=tkinter.X, pady=10)

            # Add buttons to the top row
            self.run_button = tkinter.Button(self.button_frame, text="Run Logistic Regression",
                                             command=self.prompt_for_columns, background="#355C7D",
                                             highlightthickness=0, borderwidth=0, relief="flat")
            self.run_button.pack(side=tkinter.LEFT, padx=5)

            self.show_corr_matrix = tkinter.Button(
                self.button_frame,
                text="Calculate Correlation Matrix",
                command=lambda: calc_corr_matrix(self.table.model.df, self.master),
                background="#355C7D",
                highlightthickness=0,
                borderwidth=0,
                relief="flat"
            )
            self.show_corr_matrix.pack(side=tkinter.LEFT, padx=5)

            self.neural_network_button = tkinter.Button(self.button_frame, text="Neural Network",
                                                        command=self.machine_learning, background="#355C7D",
                                                        highlightthickness=0, borderwidth=0, relief="flat")
            self.neural_network_button.pack(side=tkinter.LEFT, padx=5)

            self.one_hot_button = tkinter.Button(self.button_frame, text="One-Hot Encode Columns",
                                                 command=self.one_hot_encode_columns, background="#355C7D",
                                                 highlightthickness=0, borderwidth=0, relief="flat")
            self.one_hot_button.pack(side=tkinter.LEFT, padx=5)

            #Cluster analysis button

            self.cluster_button = tkinter.Button(self.button_frame, text="Cluster Analysis",
                                                    command=self.cluster_analysis, background="#355C7D",
                                                    highlightthickness=0, borderwidth=0, relief="flat")

            self.cluster_button.pack(side=tkinter.LEFT, padx=5)



        def one_hot_encode_columns(self):
            # Get current DataFrame
            current_data = self.table.model.df

            # Create a new popup window to select columns for one-hot encoding
            popup = tkinter.Toplevel(self)
            popup.title("Select Columns for One-Hot Encoding")

            # Column selection
            tkinter.Label(popup, text="Select columns:").pack(anchor=tkinter.W, padx=10, pady=5)
            self.columns_to_encode_vars = {}
            for col in current_data.columns:
                var = tkinter.BooleanVar()
                checkbutton = tkinter.Checkbutton(popup, text=col, variable=var)
                checkbutton.pack(anchor=tkinter.W, padx=10)
                self.columns_to_encode_vars[col] = var

            # Confirm button to apply one-hot encoding
            confirm_button = tkinter.Button(popup, text="Apply One-Hot Encoding",
                                            command=lambda: self.apply_one_hot_encoding(current_data, popup))
            confirm_button.pack(pady=10)

        def apply_one_hot_encoding(self, data, popup):
            # Get selected columns
            selected_columns = [col for col, var in self.columns_to_encode_vars.items() if var.get()]
            if not selected_columns:
                messagebox.showwarning("No Columns Selected", "Please select at least one column.")
                return

            # Apply one-hot encoding
            encoded_data = pandas.get_dummies(data, columns=selected_columns, dtype=int)

            # Update the DataFrame in the table
            self.table.updateModel(TableModel(encoded_data))
            self.table.redraw()

            # Close the popup
            popup.destroy()


        def prompt_for_columns(self):
            # Dynamically gather the column headers from the current DataFrame in the Pandas Table
            current_data = self.table.model.df  # Access the current DataFrame directly from the table's model
            column_headers = list(current_data.columns)

            # Create a new popup window to select X and Y columns
            popup = tkinter.Toplevel(self)
            popup.title("Select X and Y Columns")

            # X Columns selection (multiple)
            tkinter.Label(popup, text="Select X columns:").pack(anchor=tkinter.W, padx=10, pady=5)
            self.x_columns_vars = {}
            for col in column_headers:
                var = tkinter.BooleanVar()
                checkbutton = tkinter.Checkbutton(popup, text=col, variable=var)
                checkbutton.pack(anchor=tkinter.W, padx=10)
                self.x_columns_vars[col] = var

            # Y Column selection (single)
            tkinter.Label(popup, text="Select Y column:").pack(anchor=tkinter.W, padx=10, pady=10)
            self.y_column_var = tkinter.StringVar()
            # Set the default value for the Y dropdown to the last column
            self.y_column_var.set(column_headers[-1])  # Set default to the last column
            y_dropdown = ttk.Combobox(popup, textvariable=self.y_column_var, values=column_headers)
            y_dropdown.pack(anchor=tkinter.W, padx=10)

            #give the user the option to select StandardScaler
            tkinter.Label(popup, text="Standard Scaler:").pack(anchor=tkinter.W, padx=10, pady=5)
            self.scaler_var = tkinter.BooleanVar(value=False)
            scaler_checkbutton = tkinter.Checkbutton(popup, text="Apply Standard Scaler",
                                                    variable=self.scaler_var)
            scaler_checkbutton.pack(anchor=tkinter.W, padx=10)

            # Additional input for learning rate, iterations, and loss method
            tkinter.Label(popup, text="Learning Rate (alpha):").pack(anchor=tkinter.W, padx=10, pady=5)
            self.learning_rate_var = tkinter.DoubleVar(value=0.0001)  # Default value
            learning_rate_entry = tkinter.Entry(popup, textvariable=self.learning_rate_var)
            learning_rate_entry.pack(anchor=tkinter.W, padx=10)

            tkinter.Label(popup, text="Max Iterations:").pack(anchor=tkinter.W, padx=10, pady=5)
            self.max_iter_var = tkinter.IntVar(value=1000000)  # Default value
            max_iter_entry = tkinter.Entry(popup, textvariable=self.max_iter_var)
            max_iter_entry.pack(anchor=tkinter.W, padx=10)

            tkinter.Label(popup, text="Loss Method:").pack(anchor=tkinter.W, padx=10, pady=5)
            self.loss_method_var = tkinter.StringVar(value="log_loss")  # Default value
            loss_method_dropdown = ttk.Combobox(popup, textvariable=self.loss_method_var,
                                                values=["log_loss", "hinge", "squared_hinge"])
            loss_method_dropdown.pack(anchor=tkinter.W, padx=10)

            # Penalty Method selection
            tkinter.Label(popup, text="Penalty Method:").pack(anchor=tkinter.W, padx=10, pady=5)
            self.penalty_method_var = tkinter.StringVar(value="None")  # Default value as a string
            penalty_method_dropdown = ttk.Combobox(popup, textvariable=self.penalty_method_var,
                                                   values=["None", "l2", "l1", "elasticnet"])
            penalty_method_dropdown.pack(anchor=tkinter.W, padx=10)

            tkinter.Label(popup, text="Polynomial Degree:").pack(anchor=tkinter.W, padx=10, pady=5)
            self.degree_var = tkinter.IntVar(value=1)  # Default value is 1
            degree_combobox = ttk.Combobox(popup, textvariable=self.degree_var, values=[1, 2, 3], state="readonly")
            degree_combobox.pack(anchor=tkinter.W, padx=10)

            # Train/Test Split Checkbox
            tkinter.Label(popup, text="Train/Test Split:").pack(anchor=tkinter.W, padx=10, pady=5)
            self.train_test_split_var = tkinter.BooleanVar(value=False)
            train_test_split_checkbutton = tkinter.Checkbutton(popup, text="Apply Train/Test Split",
                                                               variable=self.train_test_split_var)
            train_test_split_checkbutton.pack(anchor=tkinter.W, padx=10)

            # Train/Test Split Fraction input
            tkinter.Label(popup, text="Test Fraction (0.0 - 1.0):").pack(anchor=tkinter.W, padx=10, pady=5)
            self.test_fraction_var = tkinter.DoubleVar(value=0.2)  # Default test size as 20%
            test_fraction_entry = tkinter.Entry(popup, textvariable=self.test_fraction_var)
            test_fraction_entry.pack(anchor=tkinter.W, padx=10)

            # Add button to confirm selection and run logistic regression
            confirm_button = tkinter.Button(popup, text="Run Logistic Regression",
                                            command=lambda: self.run_logistic_regression())
            confirm_button.pack(pady=10)

        def run_logistic_regression(self):
            # Gather the selected X and Y columns
            x_columns = [col for col, var in self.x_columns_vars.items() if var.get()]
            y_column = self.y_column_var.get()

            # Get the current DataFrame
            current_data = self.table.model.df  # Access the current DataFrame directly from the table's model

            # Create a new DataFrame with the selected X and Y columns
            selected_data = current_data[x_columns + [y_column]]

            # Get learning rate, iterations, loss method, and penalty
            alpha = self.learning_rate_var.get()
            max_iter = self.max_iter_var.get()
            loss_method = self.loss_method_var.get()
            penalty_method = self.penalty_method_var.get()
            degree_var = self.degree_var.get()
            to_scale = self.scaler_var.get()

            # Convert "None" string to None
            penalty_method = None if penalty_method == "None" else penalty_method

            # Get Train/Test split info
            apply_train_test_split = self.train_test_split_var.get()
            test_size = self.test_fraction_var.get()

            # Run regression with/without train-test split
            if apply_train_test_split:
                # If train/test split is selected, call the function with the test_size
                run_regression(selected_data, alpha=alpha, max_iter=max_iter, loss=loss_method, penalty=penalty_method,
                               test_size=test_size, degree=degree_var, to_scale=to_scale)
            else:
                # Run without splitting the data
                run_regression(selected_data, alpha=alpha, max_iter=max_iter, loss=loss_method, penalty=penalty_method, degree=degree_var, to_scale=to_scale)

        def machine_learning(self):
            self.pack(fill=tkinter.BOTH, expand=True)

            # Dynamically gather column headers from the current DataFrame in the Pandas Table
            current_data = self.table.model.df  # Access DataFrame directly from the table's model
            column_headers = list(current_data.columns)

            # Create a new popup window for X and Y column selection
            self.popup = tkinter.Toplevel(self)
            self.popup.title("Build Neural Network")

            # X Columns selection (multiple)
            tkinter.Label(self.popup, text="Select X columns:").pack(anchor=tkinter.W, padx=10, pady=5)
            self.x_columns_vars = {}
            for col in column_headers:
                var = tkinter.BooleanVar()
                checkbutton = tkinter.Checkbutton(self.popup, text=col, variable=var)
                checkbutton.pack(anchor=tkinter.W, padx=10)
                self.x_columns_vars[col] = var

            # Button to add another Y column
            add_y_button = tkinter.Button(self.popup, text="Add Another Y Column",
                                     command=lambda: self.add_y_column_selection(column_headers))
            add_y_button.pack(padx=10, pady=5, anchor=tkinter.W)

            # Y Column selection (multiple)
            tkinter.Label(self.popup, text="Select Y column(s):").pack(anchor=tkinter.W, padx=10, pady=10)
            self.y_columns_vars = []
            self.add_y_column_selection(column_headers)

            # Train/Test Split Checkbox
            tkinter.Label(self.popup, text="Train/Test Split:").pack(anchor=tkinter.W, padx=10, pady=5)
            self.train_test_split_var_nn = tkinter.BooleanVar(value=False)
            train_test_split_checkbutton_nn = tkinter.Checkbutton(self.popup, text="Apply Train/Test Split",
                                                               variable=self.train_test_split_var_nn)
            train_test_split_checkbutton_nn.pack(anchor=tkinter.W, padx=10)

            # Train/Test Split Fraction input
            tkinter.Label(self.popup, text="Test Fraction (0.0 - 1.0):").pack(anchor=tkinter.W, padx=10, pady=5)
            self.test_fraction_var_nn = tkinter.DoubleVar(value=0.2)  # Default test size as 20%
            test_fraction_entry_nn = tkinter.Entry(self.popup, textvariable=self.test_fraction_var_nn)
            test_fraction_entry_nn.pack(anchor=tkinter.W, padx=10)

        def add_y_column_selection(self, column_headers):
            y_var = tkinter.StringVar()
            y_var.set(column_headers[-1])  # Default to last column
            self.y_columns_vars.append(y_var)

            y_dropdown = ttk.Combobox(self.popup, textvariable=y_var, values=column_headers)
            y_dropdown.pack(anchor=tkinter.W, padx=10)

            # Confirm button to proceed to the next step for configuring NN layers
            confirm_button = tkinter.Button(self.popup, text="Confirm Selection",
                                       command=lambda: self.confirm_nn_selection(column_headers))
            confirm_button.pack(pady=10, anchor=tkinter.W, side=tkinter.BOTTOM)

        def confirm_nn_selection(self, column_headers):
            # Gather selected X and Y columns
            selected_x_columns = [col for col, var in self.x_columns_vars.items() if var.get()]
            selected_y_columns = [y_var.get() for y_var in self.y_columns_vars if y_var.get()]

            # Confirm selections with the user
            if not selected_x_columns:
                messagebox.showwarning("No X Columns Selected", "Please select at least one X column.")
                return

            if not selected_y_columns:
                messagebox.showwarning("No Y Columns Selected", "Please select at least one Y column.")
                return

            # Combine selected X and Y data
            NN_PD_DATA = self.table.model.df[selected_x_columns + selected_y_columns]

            # Clean the data by removing '\xa0' (non-breaking space) and converting to numeric
            NN_PD_DATA = NN_PD_DATA.map(lambda x: str(x).replace('\xa0', '').strip())  # Remove non-breaking space
            NN_PD_DATA = NN_PD_DATA.apply(pandas.to_numeric, errors='coerce')  # Convert to numeric, invalid values become NaN

            # Drop rows with NaN values across any selected column (ensuring alignment)
            NN_PD_DATA = NN_PD_DATA.dropna(subset=selected_x_columns + selected_y_columns)

            NN_PD_DATA_X = NN_PD_DATA[selected_x_columns]
            NN_PD_DATA_Y = NN_PD_DATA[selected_y_columns]

            # Convert to NumPy arrays
            X_data = NN_PD_DATA_X.to_numpy()
            y_data = NN_PD_DATA_Y.to_numpy()

            # Open a new popup for configuring the neural network
            nn_popup = tkinter.Toplevel(self)

            #include train_test_split if selecte
            if self.train_test_split_var_nn.get():
                test_size = self.test_fraction_var_nn.get()
                nn_builder = NeuralNetworkArchitectureBuilder(nn_popup, X_data, y_data, NN_PD_DATA_X, NN_PD_DATA_Y, train_test_split_var=test_size)
                nn_builder.configure_nn_popup()
            else:
                nn_builder = NeuralNetworkArchitectureBuilder(nn_popup, X_data, y_data, NN_PD_DATA_X, NN_PD_DATA_Y)
                nn_builder.configure_nn_popup()

        def cluster_analysis(self):
            self.pack(fill=tkinter.BOTH, expand=True)

            # Dynamically gather column headers from the current DataFrame in the Pandas Table
            self.full_dataset = self.table.model.df
            column_headers = list(self.full_dataset.columns)

            # Create a new popup for column selection
            self.popup = tkinter.Toplevel(self)
            self.popup.title("Cluster Analysis")

            # X Columns selection (multiple)
            tkinter.Label(self.popup, text="Select X columns:").pack(anchor=tkinter.W, padx=10, pady=5)
            self.x_columns_vars = {}
            for col in column_headers:
                var = tkinter.BooleanVar()
                checkbutton = tkinter.Checkbutton(self.popup, text=col, variable=var)
                checkbutton.pack(anchor=tkinter.W, padx=10)
                self.x_columns_vars[col] = var

            # Create a dropdown to select the clustering method
            tkinter.Label(self.popup, text="Clustering Method:").pack(anchor=tkinter.W, padx=10, pady=5)
            self.cluster_method_var = tkinter.StringVar(value="KMeans")  # Default value is KMeans
            cluster_method_combobox = ttk.Combobox(self.popup, textvariable=self.cluster_method_var,
                                                   values=["KMeans", "Agglomerative", "DBSCAN"], state="readonly")
            cluster_method_combobox.pack(anchor=tkinter.W, padx=10)

            # Create a dropdown for the number of random starts
            tkinter.Label(self.popup, text="Number of Random Starts:").pack(anchor=tkinter.W, padx=10, pady=5)
            self.random_starts_var = tkinter.IntVar(value=10)  # Default value is 10
            random_starts_combobox = ttk.Combobox(self.popup, textvariable=self.random_starts_var,
                                                  values=[str(i) for i in range(1, 101)], state="readonly")
            random_starts_combobox.pack(anchor=tkinter.W, padx=10)

            # Create a button to send the data to Cluster Analysis
            confirm_button = tkinter.Button(self.popup, text="Run Cluster Analysis",
                                            command=lambda: self.run_cluster_analysis())
            confirm_button.pack(pady=10)

        def run_cluster_analysis(self):
            # Gather the selected X columns
            selected_x_columns = [col for col, var in self.x_columns_vars.items() if var.get()]
            if not selected_x_columns:
                tkinter.messagebox.showerror("Error", "Please select at least one column for clustering.")
                return

            # Extract data for clustering
            cluster_data = self.table.model.df[selected_x_columns].to_numpy()  # Convert to numpy array

            # Get the clustering method
            cluster_method = self.cluster_method_var.get()

            # Get the number of random starts
            random_starts = self.random_starts_var.get()

            # Create a new popup for cluster tabs
            cluster_popup = tkinter.Toplevel(self.master)

            #get the data in a pandas dataframe with headers
            cluster_data_PD = pandas.DataFrame(cluster_data, columns=selected_x_columns)

            cluster_popup.title("Cluster Analysis")

            # Create the ClusterAnalysis instance
            cluster_analysis = ClusterAnalysis(
                master=cluster_popup,
                data=cluster_data,
                n_clusters=2,  # Start with 2 clusters
                random_starts=random_starts,
                data_PD=cluster_data_PD,
                full_dataset=self.full_dataset,  # Assuming `cluster_data` is your full dataset
                cluster_method=cluster_method
            )


    global RXN_Type, RXN_Samples, RXN_EOR, RXN_EM, RXN_EM_Value, NUM_OF_SIM


    class RxnDetails(tkinter.Frame):
        def __init__(self, master=tab1):
            tkinter.Frame.__init__(self, master)
            self.tableheight = None
            self.tablewidth = None
            self.entries = None
            self.place(relx=0.01, rely=0.01, anchor=NW, relwidth=0.3, relheight=0.1)
            self.create_table()

        def create_table(self):
            self.entries = {}
            self.tableheight = 4
            self.tablewidth = 3
            counter = 0
            for column in range(self.tablewidth):
                for row in range(self.tableheight):
                    self.entries[counter] = tkinter.Entry(self)
                    self.entries[counter].grid(row=row, column=column, sticky="nsew")
                    # self.entries[counter].insert(0, str(counter))
                    self.entries[counter].config(justify="center", width=18)
                    counter += 1
            self.table_labels()

            for row in range(self.tableheight):
                tkinter.Grid.rowconfigure(self, row, weight=1)
            for column in range(self.tablewidth):
                tkinter.Grid.columnconfigure(self, column, weight=1)

        def table_labels(self):
            self.entries[2].delete(0, tkinter.END)
            self.entries[2].insert(0, "# of samples =")
            self.entries[2].config(state="readonly")
            self.entries[3].delete(0, tkinter.END)
            self.entries[3].insert(0, "# of Simulations =")
            self.entries[3].config(state="readonly")
            self.user_entry()

        def user_entry(self):
            global RXN_Type, RXN_Samples, RXN_EOR, RXN_EM, RXN_EM_Value, RXN_EM_2, RXN_EM_Entry_2, RXN_EM_2_SR, RXN_EM_Entry_2_SR, RXN_EM_2_Active, RXN_EM_2_Check, RXN_EM_Value_2, RXN_EM_Entry, NUM_OF_SIM, RXN_EM_Operator, RXN_EM_Operator_2
            RXN_EM = tkinter.StringVar()
            RXN_EM.set("1º End Metric")
            reactants_list = []
            RXN_EM_Entry = AutocompleteCombobox(self, completevalues=End_Metrics, width=15, textvariable=RXN_EM)
            RXN_EM_Entry.grid(row=0, column=0, sticky="nsew")
            RXN_EM_Entry.config(justify="center", state="readonly")
            RXN_EM_Value = self.entries[8]

            RXN_EM_Operator = tkinter.StringVar()
            RXN_EM_Operator.set("<=")
            RXN_EM_Operator_Entry = AutocompleteCombobox(self, completevalues=["<=", ">="], width=15,
                                                         textvariable=RXN_EM_Operator)
            RXN_EM_Operator_Entry.grid(row=0, column=1, sticky="nsew")
            RXN_EM_Operator_Entry.config(justify="center", state="readonly")

            RXN_EM_2_Active = tkinter.BooleanVar()
            RXN_EM_2_Check = tkinter.Checkbutton(self, text="2º Active?", variable=RXN_EM_2_Active, onvalue=True,
                                                 offvalue=False)
            RXN_EM_2_Check.grid(row=0, column=3, sticky="nsew")
            RXN_EM_2 = tkinter.StringVar()
            RXN_EM_Entry_2 = AutocompleteCombobox(self, completevalues=End_Metrics, width=15, textvariable=RXN_EM_2)
            RXN_EM_Entry_2.grid(row=1, column=0, sticky="nsew")
            RXN_EM_Entry_2.insert(0, "2º End Metric")
            RXN_EM_Entry_2.config(justify="center", state="readonly")
            RXN_EM_Value_2 = self.entries[9]

            RXN_EM_Operator_2 = tkinter.StringVar()
            RXN_EM_Operator_2.set("<=")
            RXN_EM_Operator_Entry_2 = AutocompleteCombobox(self, completevalues=["<=", ">="], width=15,
                                                           textvariable=RXN_EM_Operator_2)
            RXN_EM_Operator_Entry_2.grid(row=1, column=1, sticky="nsew")
            RXN_EM_Operator_Entry_2.config(justify="center", state="readonly")

            RXN_EM_2_SR = tkinter.StringVar()
            RXN_EM_Entry_2_SR = AutocompleteCombobox(self, completevalues=reactants_list, width=15,
                                                     textvariable=RXN_EM_2_SR)
            RXN_EM_Entry_2_SR.grid(row=1, column=3, sticky="nsew")
            if RXN_EM_Entry_2_SR.get() == "":
                RXN_EM_Entry_2_SR.insert(0, "2º Start")
            RXN_EM_Entry_2_SR.config(justify="center")
            RXN_Samples = tkinter.StringVar()
            RXN_Samples_Entry = AutocompleteCombobox(self, completevalues=Num_Samples, width=15,
                                                     textvariable=RXN_Samples)
            RXN_Samples_Entry.insert(0, "10000")
            RXN_Samples_Entry.grid(row=2, column=1, sticky="nsew")
            RXN_Samples_Entry.config(justify="center")

            NUM_OF_SIM = tkinter.StringVar()
            Core_options = [str(i) for i in range(1, multiprocessing.cpu_count() - 1)]
            NUM_OF_SIM_Entry = AutocompleteCombobox(self, completevalues=Core_options, width=15,
                                                    textvariable=NUM_OF_SIM)
            NUM_OF_SIM_Entry.insert(0, str(int(multiprocessing.cpu_count() * 0.75)))
            NUM_OF_SIM_Entry.grid(row=3, column=1, sticky="nsew")
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
            self.tablewidth = 4
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
            labels = [
                "Mn (Number Average) =",
                "Mw (Weight Average) =",
                "PDI (Dispersity Index) =",
                "Mz =",
                "Mz + 1 =",
                "Xn DOP ="
            ]

            for i, label_text in enumerate(labels):
                self.entries[i].delete(0, tkinter.END)
                self.entries[i].insert(0, label_text)
                self.entries[i].config(width=25, state="readonly")


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
            self.tablewidth = 4
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
            labels = [
                "Mn (Number Average) =",
                "Mw (Weight Average) =",
                "PDI (Dispersity Index) =",
                "Mz =",
                "Mz + 1 =",
                "Xn DOP ="]

            for i, label_text in enumerate(labels):
                self.entries[i].delete(0, tkinter.END)
                self.entries[i].insert(0, label_text)
                self.entries[i].config(width=25, state="readonly")


    class Buttons(tkinter.Frame):
        def __init__(self, master=tab1):
            tkinter.Frame.__init__(self, master)
            self.place(relx=0.01, rely=0.12, anchor=NW, relwidth=0.08, relheight=0.1)
            self.add_buttons()

        def add_buttons(self):
            self.Simulate = tkinter.Button(self, text="Simulate", command=multiprocessing_sim, bg="Green", relief="groove")
            self.Simulate.grid(row=0, column=0, sticky="nsew")
            self.stop_button = tkinter.Button(self, text="Terminate", command=stop, bg="Red", relief="groove")
            self.stop_button.grid(row=1, column=0, sticky='nsew')
            self.clear_last_row = tkinter.Button(self, text="Clear Last", command=clear_last, bg="Yellow", relief="groove")
            self.clear_last_row.grid(row=2, column=0, sticky='nsew')
            self.Reset = tkinter.Button(self, text="Reset", command=reset_entry_table, bg="Orange", relief="groove")
            self.Reset.grid(row=3, column=0, sticky='nsew')
            self.grid_columnconfigure(0, weight=1)
            for i in range(4):
                self.grid_rowconfigure(i, weight=1)


    class SimStatus(tkinter.Frame):
        def __init__(self, master=tab1):
            tkinter.Frame.__init__(self, master)
            self.tablewidth = None
            self.tableheight = None
            self.progress = None
            self.entries = None
            self.place(relx=0.5, rely=0.18, anchor=CENTER, relwidth=0.3, relheight=0.06)
            self.create_table()

        def create_table(self):
            self.entries = {}
            self.tableheight = 2
            self.tablewidth = 2
            counter = 0
            for column in range(self.tablewidth):
                for row in range(self.tableheight):
                    self.entries[counter] = tkinter.Entry(self)
                    self.entries[counter].grid(row=row, column=column, sticky="nsew")
                    self.entries[counter].insert(0, str(counter))
                    self.entries[counter].config(justify="center", width=18)
                    counter += 1
            self.tabel_labels()

            self.grid_columnconfigure(0, weight=1)
            self.grid_columnconfigure(1, weight=4)
            for i in range(2):
                self.grid_rowconfigure(i, weight=1)

        def tabel_labels(self):
            self.entries[0].delete(0, tkinter.END)
            self.entries[0].insert(0, "1º Simulation Status")
            self.entries[0].config(state="readonly")
            self.entries[1].delete(0, tkinter.END)
            self.entries[1].insert(0, "2º Simulation Status")
            self.entries[1].config(state="readonly")
            self.add_buttons()

        def add_buttons(self):
            self.progress = ttk.Progressbar(self, orient="horizontal", mode="determinate", style="red.Horizontal.TProgressbar")
            self.progress.grid(row=0, column=1, sticky="nsew")
            self.progress_2 = ttk.Progressbar(self, orient="horizontal", mode="determinate", style="red.Horizontal.TProgressbar")
            self.progress_2.grid(row=1, column=1, sticky="nsew")

    RET = RxnEntryTable()
    WD = WeightDist()
    WD2 = WeightDist_2()
    RD = RxnDetails()
    Buttons = Buttons()
    sim = SimStatus()
    CT = combinations_table()
    DFE = DataFrameEditor()

    # run update_table if user changes value in RET
    for i in range(16, 305, 16):
        RET.entries[i].bind("<FocusOut>",
                            lambda *args, entry=i, index=int(i / 16 - 1), cell=i: check_entry(entry, index, cell))

    # run update_table if user changes value in RET
    for i in range(0, 19):
        Entry_masses[i].bind("<KeyRelease>",
                             lambda *args, index=i, cell=int(i * 16 + 16): RET.update_table(index, cell))

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