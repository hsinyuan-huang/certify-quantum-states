import seaborn as sns
sns.set(font="Avenir", style="ticks", font_scale=1.3)
import matplotlib.pyplot as plt
import pandas as pd
import os
import subprocess
import numpy as np
import scipy
X = np.matrix([[0,1],[1,0]])
Y = np.matrix([[0,-1j],[1j,0]])
Z = np.matrix([[1,0],[0,-1]])
os.system("make") # Update C++ codes

N = 50

if os.path.exists("pickle-dataframe-plot.pkl") == False:
    data = []
    for n in [4, 12, 20]:
        for p in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            for id in range(500):
                proc = subprocess.Popen(["./XEBvsShadowFid-Haar", str(n), str(N), str(p), str(id)], stdout=subprocess.PIPE)
                (out, err) = proc.communicate()
                XEB, SFid, Fid = (float(strn.split(" ")[1]) for strn in (out).decode('UTF-8').split('\n')[:3])

                data.append(["Haar", "Pure White Noise", n, N, p, id, XEB, SFid, Fid])

            for id in range(500):
                proc = subprocess.Popen(["./XEBvsShadowFid-Haar-general-err", str(n), str(N), str(p), str(id)], stdout=subprocess.PIPE)
                (out, err) = proc.communicate()
                XEB, SFid, Fid = (float(strn.split(" ")[1]) for strn in (out).decode('UTF-8').split('\n')[:3])

                data.append(["Haar", "Coherent Noise", n, N, p, id, XEB, SFid, Fid])

            for id in range(500):
                proc = subprocess.Popen(["./XEBvsShadowFid-Phase", str(n), str(N), str(p), str(id)], stdout=subprocess.PIPE)
                (out, err) = proc.communicate()
                XEB, SFid, Fid = (float(strn.split(" ")[1]) for strn in (out).decode('UTF-8').split('\n')[:3])

                data.append(["Phase", "Pure White Noise", n, N, p, id, XEB, SFid, Fid])

            for id in range(300):
                proc = subprocess.Popen(["./XEBvsShadowFid-Phase-general-err", str(n), str(N), str(p), str(id)], stdout=subprocess.PIPE)
                (out, err) = proc.communicate()
                XEB, SFid, Fid = (float(strn.split(" ")[1]) for strn in (out).decode('UTF-8').split('\n')[:3])

                data.append(["Phase", "Coherent Noise", n, N, p, id, XEB, SFid, Fid])

    pd_data = pd.DataFrame(data=data, columns = ['State Type', 'Type of Noise', 'Size', 'Samples', 'Error', 'Seed', "XEB", "Shad-Fid", "Fid"])
    pd_data.to_pickle("pickle-dataframe-plot.pkl")
else:
    pd_data = pd.read_pickle("pickle-dataframe-plot.pkl")

if os.path.exists("pickle-new-dataframe-plot-additional.pkl") == False:
    data = []
    for n in [4, 12, 20]:
        for p in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            for id in range(500):
                proc = subprocess.Popen(["./XEBvsShadowFid-Haar-dephasing", str(n), str(N), str(p), str(id)], stdout=subprocess.PIPE)
                (out, err) = proc.communicate()
                XEB, SFid, Fid = (float(strn.split(" ")[1]) for strn in (out).decode('UTF-8').split('\n')[:3])

                data.append(["Haar", "Dephasing Noise", n, N, p, id, XEB, SFid, Fid])

            for id in range(500):
                proc = subprocess.Popen(["./XEBvsShadowFid-Phase-dephasing", str(n), str(N), str(p), str(id)], stdout=subprocess.PIPE)
                (out, err) = proc.communicate()
                XEB, SFid, Fid = (float(strn.split(" ")[1]) for strn in (out).decode('UTF-8').split('\n')[:3])

                data.append(["Phase", "Dephasing Noise", n, N, p, id, XEB, SFid, Fid])

    pd_data_additional = pd.DataFrame(data=data, columns = ['State Type', 'Type of Noise', 'Size', 'Samples', 'Error', 'Seed', "XEB", "Shad-Fid", "Fid"])
    pd_data_additional.to_pickle("pickle-dataframe-plot-additional.pkl")
else:
    pd_data_additional = pd.read_pickle("pickle-dataframe-plot-additional.pkl")

# Wanted palette details
enmax_palette = sns.color_palette("deep").as_hex()
color_codes_wanted = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey', 'yellow', 'cyan']
c = lambda x: enmax_palette[color_codes_wanted.index(x)]

for n, color1, color2 in [(4, 'blue', 'yellow'), (12, 'green', 'red'), (20, 'cyan', 'orange')]:
    for state_type in ["Haar", "Phase"]:
        for exper_type, exper_type_name in [("Pure White Noise", "White Noise"), ("Coherent Noise", "Coherent Noise")]:
            plt.figure(figsize=(4.2, 3.75))
            subdata = pd_data[(pd_data['Type of Noise']==exper_type) & (pd_data['State Type']==state_type) & (pd_data['Size']==n)]

            # Fidelity
            sns_plot = sns.pointplot(x="Error", y="Fid", data=subdata, color=(0/255, 51/255, 102/255), linestyles=":", marker="", linewidth=0.75, label="Truth")

            # XEB
            sns_plot = sns.pointplot(x="Error", y="XEB", data=subdata, palette=sns.color_palette([c(color1)]), linestyles="", markersize = 4, hue="Size")

            # Shadow Fidelity
            sns_plot = sns.pointplot(x="Error", y="Shad-Fid", data=subdata, palette=sns.color_palette([c(color2)]), linestyles="", markers="D", markersize = 4, hue="Size")
            sns_plot.set_xticks([0, 2, 4, 6, 8, 10])

            handles, labels = sns_plot.get_legend_handles_labels()
            labels = ['True Fidelity', 'XEB', 'Shadow Overlap']
            sns_plot.legend(handles, labels, loc='lower left')

            sns_plot.set(xlabel=exper_type_name, ylabel='Estimated Fidelity')

            ylim = sns_plot.get_ylim();
            plt.ylim(0.0, max(1.06, ylim[1]))

            plt.title(r'Hilbert space d = $2^{{{}}}$ ({})'.format(n, state_type))

            plt.tight_layout()
            plt.savefig("plot_{}_{}_{}.png".format(n, exper_type_name, state_type), dpi=800)
            plt.clf()

# For additional experiments

for n, color1, color2 in [(4, 'blue', 'yellow'), (12, 'green', 'red'), (20, 'cyan', 'orange')]:
    for state_type in ["Haar", "Phase"]:
        for exper_type, exper_type_name in [("Dephasing Noise", "Dephasing Noise")]:
            plt.figure(figsize=(4.2, 3.75))
            subdata = pd_data_additional[(pd_data_additional['Type of Noise']==exper_type) & (pd_data_additional['State Type']==state_type) & (pd_data_additional['Size']==n)]

            # Fidelity
            sns_plot = sns.pointplot(x="Error", y="Fid", data=subdata, color=(0/255, 51/255, 102/255), linestyles=":", marker="", linewidth=0.75, label="Truth")

            # XEB
            sns_plot = sns.pointplot(x="Error", y="XEB", data=subdata, palette=sns.color_palette([c(color1)]), linestyles="", markersize = 4, hue="Size")

            # Shadow Fidelity
            sns_plot = sns.pointplot(x="Error", y="Shad-Fid", data=subdata, palette=sns.color_palette([c(color2)]), linestyles="", markers="D", markersize = 4, hue="Size")
            sns_plot.set_xticks([0, 2, 4, 6, 8, 10])

            handles, labels = sns_plot.get_legend_handles_labels()
            labels = ['True Fidelity', 'XEB', 'Shadow Overlap']
            sns_plot.legend(handles, labels, loc='lower left')

            sns_plot.set(xlabel=exper_type_name, ylabel='Estimated Fidelity')

            ylim = sns_plot.get_ylim();
            plt.ylim(0.0, max(1.06, ylim[1]))

            plt.title(r'Hilbert space d = $2^{{{}}}$ ({})'.format(n, state_type))

            plt.tight_layout()
            plt.savefig("plot_{}_{}_{}.png".format(n, exper_type_name, state_type), dpi=800)
            plt.clf()
