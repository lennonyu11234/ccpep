import pandas as pd
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
import argparse


def moving_average(data, window_size):
    if len(data) < window_size:
        return data
    return [sum(data[i:i+window_size])/float(window_size) for i in range(len(data)-window_size+1)]


def draw_gibbs_area(gibbs_csv, mer_xvg, area_xvg):
    time, PC1, PC2 = [], [], []
    with open(mer_xvg, 'r') as f:
        lines = f.readlines()
        for line in lines:
            row = line.split()
            time.append(float(row[0]))
            PC1.append(float(row[1]))
            PC2.append(float(row[2]))
    csv_df = pd.read_csv(gibbs_csv)
    points = csv_df[['PC1', 'PC2']].values
    values = csv_df['G (kJ/mol)'].values.reshape(-1, 1)
    grid_x, grid_y = PC1, PC2
    interpolated_values = griddata(points, values, (grid_x, grid_y), method='nearest')
    time_gibbs = []
    for i in interpolated_values:
        time_gibbs.append(float(i))

    psa, sasa, nsa = [], [], []
    with open(area_xvg, 'r') as f1:
        lines = f1.readlines()
        for line in lines[26:]:
            row = line.split()
            psa.append(float(row[3]))
            nsa.append(float(row[2]))
            sasa.append(float(row[1]))

    window_size = 10
    x_smooth = time[:-window_size] + [time[-1]]
    y_smooth = moving_average(time_gibbs, window_size)
    sasa_smooth = moving_average(sasa, window_size)
    psa_smooth = moving_average(psa, window_size)
    nsa_smooth = moving_average(nsa, window_size)

    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    axs[0, 0].plot(x_smooth, y_smooth, color='red')
    axs[0, 0].set_xlabel('Time (ps)')
    axs[0, 0].set_ylabel(r'KJ/mol')
    axs[0, 0].grid(True)
    axs[0, 0].set_title('Gibbs free energy')

    axs[0, 1].plot(x_smooth, sasa_smooth, color='navy')
    axs[0, 1].set_xlabel('Time (ps)')
    axs[0, 1].set_ylabel(r'Area (nm$^{2}$)')
    axs[0, 1].grid(True)
    axs[0, 1].set_title('Surface Area')

    axs[1, 0].plot(x_smooth, psa_smooth, color='green')
    axs[1, 0].set_xlabel('Time (ps)')
    axs[1, 0].set_ylabel(r'Area (nm$^{2}$)')
    axs[1, 0].grid(True)
    axs[1, 0].set_title('Polar Surface Area')

    axs[1, 1].plot(x_smooth, nsa_smooth, color='purple')
    axs[1, 1].set_xlabel('Time (ps)')
    axs[1, 1].set_ylabel(r'Area (nm$^{2}$)')
    axs[1, 1].grid(True)
    axs[1, 1].set_title('Non-polar Surface Area')
    if mer_xvg == 'merge.xvg':
        fig.savefig('TimePoint_Gibbs.png', dpi=600)
    elif mer_xvg == 'init_merge.xvg':
        fig.savefig('init_gibbs.png', dpi=600)
    elif mer_xvg == 'touch_merge.xvg':
        fig.savefig('touch_gibbs.png', dpi=600)
    elif mer_xvg == 'internalise_merge.xvg':
        fig.savefig('internalise_gibbs.png', dpi=600)


def main():
    parser = argparse.ArgumentParser(description='Draw xvg')
    parser.add_argument('gibbs_csv', type=str, help='Path to the gyration radius XVG file')
    parser.add_argument('gibbs_xvg', type=str, help='Path to the gyration radius XVG file')
    parser.add_argument('area_xvg', type=str, help='Path to the gyration radius XVG file')
    args = parser.parse_args()

    draw_gibbs_area(args.gibbs_csv, args.gibbs_xvg, args.area_xvg)


if __name__ == '__main__':
    main()