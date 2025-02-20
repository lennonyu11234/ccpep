import pandas as pd
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
import argparse


def moving_average(data, window_size):
    if len(data) < window_size:
        return data
    return [sum(data[i:i+window_size])/float(window_size) for i in range(len(data)-window_size+1)]


def draw_gibbs_area(gibbs_csv, mer_xvg):
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
        
    window_size = 30
    x_smooth = time[:-window_size] + [time[-1]]
    y_smooth = moving_average(time_gibbs, window_size)

    fig, axs = plt.subplots(figsize=(20, 10))
    axs.plot(x_smooth, y_smooth, color='red')
    axs.set_xlabel('Time (ps)')
    axs.set_ylabel(r'KJ/mol')
    axs.grid(True)
    axs.set_title('Gibbs free energy')

    if mer_xvg == 'merge.xvg':
        fig.savefig('TimePoint_Gibbs.png', dpi=600)
    elif mer_xvg == 'init_merge.xvg':
        fig.savefig('init_gibbs.png', dpi=600)
    elif mer_xvg == 'touch_merge.xvg':
        fig.savefig('touch_gibbs.png', dpi=600)
    elif mer_xvg == 'internalise_merge.xvg':
        fig.savefig('internalise_gibbs.png', dpi=600)
    elif mer_xvg == 'merge_pc.xvg':
        fig.savefig('pc_gibbs.png', dpi=600)

def main():
    parser = argparse.ArgumentParser(description='Draw xvg')
    parser.add_argument('gibbs_csv', type=str, help='Path to the gyration radius XVG file')
    parser.add_argument('gibbs_xvg', type=str, help='Path to the gyration radius XVG file')
    args = parser.parse_args()

    draw_gibbs_area(args.gibbs_csv, args.gibbs_xvg)


if __name__ == '__main__':
    main()