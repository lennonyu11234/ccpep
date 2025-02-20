import argparse


def merge_xvg(rmsd, gyrate, output_file):
    if rmsd == 'rmsd.xvg':
        with open(rmsd, 'r') as f1:
            data1 = f1.readlines()
        with open(gyrate, 'r') as f2:
            data2 = f2.readlines()
        data1 = [line.strip().split() for line in data1 if line.strip()]
        data2 = [line.strip().split() for line in data2 if line.strip()]
        data1 = data1[18:]
        data2 = data2[27:]

        merged_data = []
        for row1, row2 in zip(data1, data2):
            merged_row = row1[:2] + [row2[1]]
            merged_data.append(merged_row)

        max_widths = [max(len(str(item)) for item in column) for column in zip(*merged_data)]
        # print(merged_data)
        with open(output_file, 'w') as outfile:
            for row in merged_data:
                formatted_row = '    '.join(f"{item:<{width}}" for item, width in zip(row, max_widths))
                outfile.write(formatted_row + '\n')
    elif rmsd == 'pc1.xvg':
        with open(rmsd, 'r') as f1:
            data1 = f1.readlines()
        with open(gyrate, 'r') as f2:
            data2 = f2.readlines()
        data1 = [line.strip().split() for line in data1 if line.strip()]
        data2 = [line.strip().split() for line in data2 if line.strip()]
        data1 = data1[24:-1]
        data2 = data2[24:-1]

        merged_data = []
        for row1, row2 in zip(data1, data2):
            merged_row = row1[:2] + [row2[1]]
            merged_data.append(merged_row)

        max_widths = [max(len(str(item)) for item in column) for column in zip(*merged_data)]
        # print(merged_data)
        with open(output_file, 'w') as outfile:
            for row in merged_data:
                formatted_row = '    '.join(f"{item:<{width}}" for item, width in zip(row, max_widths))
                outfile.write(formatted_row + '\n')


def main():
    parser = argparse.ArgumentParser(description='Merge two XVG files.')
    parser.add_argument('rmsd', type=str, help='Path to the RMSD XVG file')
    parser.add_argument('gyrate', type=str, help='Path to the gyration radius XVG file')
    parser.add_argument('output_file', type=str, help='Path to the output merged XVG file')

    args = parser.parse_args()

    merge_xvg(args.rmsd, args.gyrate, args.output_file)


if __name__ == '__main__':
    main()


