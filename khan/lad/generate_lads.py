import argparse
import collections
import json

from schrodinger.structure import StructureReader

import lad

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="generate LAD for a dataset",
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'input_file',
        nargs='+',
        type=str,
        help="input mae files containing datset of structures used as reference lads"
    )
    parser.add_argument(
        '-output_file',
        type=str,
        default="reference_lad_data.json",
        help="output file of precomputed lad data"
    )
    parser.add_argument(
        '-key',
        type=str,
        default='r_lad_ff_charges',
        help="atom property key used as y-value"
    )

    args = parser.parse_args()

    # use default LAD params
    params = dict(lad.LAD_PARAMS)

    all_lad_data = [] 

    for fname in args.input_file:
        for st in StructureReader(fname):
            atomic_numbers = [at.atomic_number for at in st.atom]
            carts = st.getXYZ(copy=True)
            st_mcn = lad.MCN(params, atomic_numbers, carts)
            for at in st.atom:
                at_mcn = st_mcn[at.index-1]
                lad_data = lad.LADData(
                    at.atomic_number,
                    at.property[args.key],
                    at_mcn
                )
                dct = lad_data._asdict()
                if dct not in all_lad_data:
                    all_lad_data.append(dct)
                else:
                    print("skipping mcn")

    # write lad data
    with open(args.output_file, "w") as fout:
        json.dump(all_lad_data, fout, indent=4)
