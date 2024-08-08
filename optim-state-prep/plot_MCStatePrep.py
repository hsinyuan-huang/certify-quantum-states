import seaborn as sns
sns.set(font="Avenir", style="ticks")
import matplotlib.pyplot as plt
import pandas as pd
import os
import subprocess
os.system("make") # Update C++ codes

if os.path.exists("pickle-dataframe-plot-MCStatePrep.pkl") == False:
    full_data = []
    for n in [5, 50]:
        for seed in [131313131]:
            for typ in [0, 1]:
                proc = subprocess.Popen(["./StatePrep", str(n), str(seed), str(typ)], stdout=subprocess.PIPE)
                (out, err) = proc.communicate()
                data_list = list([(float(strn.split(" ")[0]), float(strn.split(" ")[1])) for strn in (out).decode('UTF-8').split('\n')[:-2]])
                print(data_list)
                for step, (fid, shadow_o) in enumerate(data_list):
                    full_data.append((n, seed, typ, step, fid, shadow_o))
    pd_data = pd.DataFrame(data=full_data, columns = ['System size', 'Seed', 'Type', 'Steps', 'Fidelity', 'Shadow Fidelity'])
    pd_data.to_pickle("pickle-dataframe-plot-MCStatePrep.pkl")
else:
    pd_data = pd.read_pickle("pickle-dataframe-plot-MCStatePrep.pkl")

pd_data = pd_data.rename(columns={'Shadow Overlap': 'Shadow Overlap'})
pd_data['Type'] = pd_data['Type'].map({0: 'Trained w/ fidelity', 1: 'Trained w/ shadow ove.'})

enmax_palette = sns.color_palette("deep").as_hex()
color_codes_wanted = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey', 'yellow', 'cyan']
c = lambda x: enmax_palette[color_codes_wanted.index(x)]

for sys_size in [50]:
    plt.figure(figsize=(2.6, 5))
    ax = sns.lineplot(x = 'Steps', y = 'Fidelity', hue = 'Type', data=pd_data[(pd_data['System size'] == sys_size)],\
                palette=sns.color_palette([c('purple'), c('red')]), \
                lw=1, style = 'Type', dashes=[(1,1), (1,0)])
    plt.xlabel('Optimization Steps')
    plt.legend(loc='upper left', bbox_to_anchor=(-0.4, 1.2),
              ncol=1, shadow=False)
    plt.ylim(-0.06, 1.04)
    plt.tight_layout()
    plt.savefig("plot_MCStatePrep_fidelity_n={}.png".format(sys_size), dpi=800)

    plt.figure(figsize=(2.6, 5))
    ax = sns.lineplot(x = 'Steps', y = 'Shadow Overlap', hue = 'Type', data=pd_data[(pd_data['System size'] == sys_size)],\
                palette=sns.color_palette([c('purple'), c('red')]), \
                lw=1, style = 'Type', dashes=[(1,1), (1,0)])
    plt.xlabel('Optimization Steps')
    plt.ylim(-0.06, 1.04)
    plt.legend(loc='upper left', bbox_to_anchor=(-0.35, 1.2),
              ncol=1, shadow=False)
    plt.tight_layout()
    plt.savefig("plot_MCStatePrep_shadowF_n={}.png".format(sys_size), dpi=800)

full_data = []
seed = 131313131
for n in [6, 50]:
    proc = subprocess.Popen(["./StatePrep", str(n), str(seed), "2"], stdout=subprocess.PIPE)
    (out, err) = proc.communicate()
    data_list = list([(float(strn.split(" ")[0]), float(strn.split(" ")[1])) for strn in (out).decode('UTF-8').split('\n')[:-1]])
    print(data_list)
    for step, (fid, shadow_o) in enumerate(data_list):
        full_data.append((n, seed, "Fidelity", step, fid))
        full_data.append((n, seed, "Shadow Overlap", step, shadow_o))
pd_data = pd.DataFrame(data=full_data, columns = ['System size', 'Seed', 'Type', 'Steps', 'Value'])

for n in [6, 50]:
    plt.figure(figsize=(2.6, 5))
    ax = sns.lineplot(x = 'Steps', y = 'Value', hue='Type', data=pd_data[(pd_data['System size'] == n)],\
                palette=sns.color_palette([c('purple'), c('red')]), \
                lw=1, style = 'Type', dashes=[(2,2), (1,1)], markers=['o', 's'], markersize=5)
    plt.xlabel('State Construction Steps')
    plt.ylabel('Estimated Value')
    plt.legend(loc='upper left', bbox_to_anchor=(-0.1, 1.2),
                ncol=1, shadow=False)
    plt.ylim(-0.06, 1.04)
    plt.tight_layout()
    plt.savefig("plot_MCStatePrep_Construct_n={}.png".format(n), dpi=800)
