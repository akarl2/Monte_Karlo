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
    except ValueError:
        messagebox.showinfo("Error", "Please enter a valid number for the end metric value(s)")
        return "Error"

def multiprocessing_sim():
    if __name__ == "__main__":
        Buttons.Simulate.config(state="disabled", text="Running...")
        sim.progress['value'] = 0
        sim.progress_2['value'] = 0
        global running
        running = True
        #workers = 1
        workers = int(os.cpu_count() * .85)
        initialize_sim(workers)
        if initialize_sim(workers) == "Error":
            Buttons.Simulate.config(text="Simulate", state="normal")
            return

multiprocessing_sim()