from ezc3d import c3d
from stan_utils import *

# Export EMG data to a csv file
def importEMG(input_dir, file_name, output_dir):
    filename = os.path.join(input_dir, file_name)
    c = c3d(filename)

    analogs = c['data']['analogs']
    data=analogs[0]
    labels_list = c['parameters']['ANALOG']['LABELS']['value']

    emg_data = []
    emg_list=[]
    for i, l in enumerate(labels_list):
        if 'Voltage' in l:
            emg_list.append(l[8:])
            emg_data.append(data[i])

    emg_df = pd.DataFrame()        
    for i, label in enumerate(emg_list):
        signal = calculate_emg_linear_envelope(emg_data[i])
        emg_df[label] = signal.tolist()
    to_write = os.path.join(output_dir, 'emg.csv')
    emg_df.to_csv(to_write, index=False)
    return emg_df