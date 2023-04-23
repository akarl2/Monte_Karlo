def simulate(starting_materials, starting_materials_sec, end_metric_value, end_metric_selection, end_metric_value_sec, end_metric_selection_sec, sn_dict, RXN_EM_2_Active_status, total_ct, total_ct_sec, workers, process_queue, PID_list, progress_queue_sec):
    PID_list.append(os.getpid())
    time.sleep(1)
    rg = reactive_groups()
    global test_count, test_interval, sn_dist, in_situ_values, Xn_list, byproducts, running, in_primary, in_situ_values_sec, Xn_list_sec, quick_add, comp_primary, comp_secondary
    in_situ_values = [[], [], [], [], [], [], [], [], []]
    in_situ_values_sec = [[], [], [], [], [], [], [], [], []]
    Xn_list, Xn_list_sec, byproducts, composition, composition_sec = [], [], [], [], []
    running, in_primary = True, True
    test_count = 0
    test_interval = 40
    process_queue.put(0)

    #Extra code....#

    update_metrics(TAV, AV, OH, EHC, COC, IV):
        if os.getpid() == PID_list[-1]:



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


def multiprocessing_sim():
    if __name__ == "__main__":
        global running
        running = True
        #workers = 1
        workers = int(os.cpu_count() * .75)
        initialize_sim(workers)
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
                    window.update()
                except queue.Empty:
                    pass
                try:
                    progress_sec = progress_queue_sec.get_nowait()
                    sim.progress_2['value'] = progress_sec
                    window.update()
                except queue.Empty:
                    pass
            if running is False and canceled_by_user is True:
                for result in results:
                    result.cancel()
                    result.result()
                    window.update()
                while not progress_queue.empty():
                    progress_queue.get()
                sim.progress['value'] = 0
                messagebox.showinfo("Simulation Cancelled", "Simulation cancelled by user")
                return
            concurrent.futures.wait(results)
            consolidate_results(results)


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

    #Extra code....#

    RET = RxnEntryTable()
    WD = WeightDist()
    WD2 = WeightDist_2()
    RD = RxnDetails()
    RM = RxnMetrics()
    RM2 = RxnMetrics_sec()
    Buttons = Buttons()
    sim = SimStatus()