cen_val = 4.879000230310884 
def process_units(units):
    units = [int(u) for u in units.split()]
    out = [u for i, u in enumerate(units) if i == 0 or u != units[i - 1]]
    return out

def main(args):
        
    for split in ['train', 'valid', 'test']:
        with open(os.path.join(args.input_dir, 'wav_16k', 'unit_{}.txt'.format(split))) as f:
            lines = f.readlines()

        with open(os.path.join(output_dir, '{}.tsv'.format(split)), 'w') as f:
            f.write('id\tsrc_audio\tsrc_n_frames\ttgt_audio\ttgt_n_frames\n')
            for line in tqdm(lines):
                id_, unit = line.strip().split('|')
                unit = process_units(unit)
                tn = len(unit)
                sem = os.path.join(output_dir, 'sem', split, id_+'.pt')
                if not os.path.isfile(sem):
                    continue
                sn = str(int(cen_val * tn))
                
                if sem == None:
                    continue
                                        
                new = "\t".join([id_, sem, sn, unit, tn]) + '\n'
                sem = None
                f.write(new)
        print(c)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=True)
    a = parser.parse_args()


    main(a)