import seaborn as sns
sns.set(font="Avenir", style="ticks")
import matplotlib.pyplot as plt
import pandas as pd
import os
import subprocess
import numpy as np
import scipy

for type, filename in enumerate(["structured-40qubit-state-NN.txt", "pseudo-random-120qubit-state-NN.txt"]):
    data, Fid_data = [], []
    best_step = 0
    final_fid = 0
    true_purity, init_purity, trained_purity = [], [], []

    with open(filename, "r") as f:
        lines = f.readlines()
        true_purity = [float(x) for x in lines[1].split(" ")]
        init_purity = [float(x) for x in lines[3].split(" ")]
        trained_purity = [float(x) for x in lines[-2].split(" ")]
        final_fid = float(lines[-1].split(" ")[3])

        lines = lines[4:-3]

        for i in range(0, len(lines), 7):
            steps = int(lines[i].split(" ")[0])+1 # from 0-based to 1-based
            best_or_not = (lines[i].split(" ")[1] == 'Best\n')
            Tlogloss = float(lines[i+1].split(' ')[1])
            Vlogloss = float(lines[i+2].split(' ')[1])
            TShadowF = float(lines[i+3].split(' ')[1])
            VShadowF = float(lines[i+4].split(' ')[1])
            ShadowF = float(lines[i+5].split(' ')[1])
            Fidelity = float(lines[i+6].split(' ')[1])

            if best_or_not == True:
                best_step = steps

            if Fidelity > -999:
                Fid_data.append([steps, Fidelity])

            data.append([steps, Tlogloss, Vlogloss, TShadowF, VShadowF, ShadowF])

    data = np.array(data)
    Fid_data = np.array(Fid_data)

    # Wanted palette details
    enmax_palette = sns.color_palette("deep").as_hex()
    color_codes_wanted = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey', 'yellow', 'cyan']
    c = lambda x: enmax_palette[color_codes_wanted.index(x)]


    # fig, ax = plt.subplots(figsize=(8, 2.4))
    fig, ax = plt.subplots(figsize=(8, 3.2 * 8 / 6.5))
    ax.set_xlabel("Neural Network Training Steps (SGD)")
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)

    # plt.axvline(x=best_step, linewidth=1, color='k')
    # plt.text(best_step, 0.45, 'Best validation score', rotation=270, fontsize=10)

    plt.plot(data[:, 0], data[:, 1], label='Shadow-based log loss (Training)', color=c('blue'), linewidth=1, linestyle='--')
    plt.plot(data[:, 0], data[:, 2], label='Shadow-based log loss (Validation)', color=c('cyan'), linewidth=1, linestyle='--')
    plt.legend(loc='upper left', bbox_to_anchor=(-0.005, 1.28),
              ncol=1, shadow=False)
    ax.set_ylabel("Log Loss")

    ax2 = ax.twinx()
    plt.scatter(Fid_data[:, 0], Fid_data[:, 1], label='Fidelity', color=c('red'), marker='*', zorder=1000, s=40)
    plt.plot(data[:, 0], data[:, 5], label='Shadow Overlap', color=c('orange'))
    # plt.plot(data[:, 0], data[:, 3], label='Shadow Fidelity (Tr)', color=c('yellow'), linewidth=1, linestyle='-.')
    # plt.plot(data[:, 0], data[:, 4], label='Shadow Fidelity (Val)', color=c('grey'), linewidth=1, linestyle=':')
    # plt.legend(loc='upper left', bbox_to_anchor=(0.39, 1.28),
    #           ncol=1, shadow=False)
    ax2.set_ylabel("Estimated Fidelity", rotation=270, labelpad=15)

    # if type == 1:
        # ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig("plot_NN_{}_loss.png".format(filename.split(" ")[0]), dpi=800)
    plt.clf()

    fig = plt.figure(figsize=(8, 2.8))
    if len(true_purity) < 50:
        marker_s = 12
    else:
        marker_s = 12

    # plt.scatter(range(1, len(true_purity)+1), true_purity, label='Ground Truth', color=c('grey'), marker='o', s=4, zorder=30)
    plt.plot(range(1, len(true_purity)), true_purity[1:], label='Ground Truth', color=c('grey'), linewidth=1, linestyle=':', zorder=30)
    plt.scatter(range(1, len(true_purity)), init_purity[1:], label='Randomly Init. NQS', color=c('yellow'), marker='^', s=marker_s)
    plt.scatter(range(1, len(true_purity)), trained_purity[1:], label='Trained NQS (Fidelity = {0:.2f})'.format(final_fid), color=c('orange'), marker='D', s=marker_s)

    plt.ylim(-0.05, 1.05)
    plt.xlabel("Subsystem = { 1, 2, ..., i }")
    plt.ylabel("Estimated Purity")
    plt.legend(loc='upper left', bbox_to_anchor=(0.023, 1.3),
              ncol=3, shadow=False)

    plt.tight_layout()
    plt.savefig("plot_NN_{}_purity.png".format(filename.split(" ")[0]), dpi=800)
    plt.clf()

    print("plot_NN_{}_purity.png".format(filename.split(" ")[0]))
    print("Final NQS has Fidelity {}".format(final_fid))
