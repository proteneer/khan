import argparse
import collections
import json
import os
import sys

from schrodinger.structure import StructureReader

import lad

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="LAD interpolated charges",
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'input_file',
        type=str,
        default=None,
        help="input mae file"
    )
    parser.add_argument(
        '-reference_lad_data',
        type=str,
        required=True,
        help="reference LAD data (json) used to perform interpolation"
    )
    parser.add_argument(
        '-output_file',
        type=str,
        default=None,
        required=True,
        help="output mae file for structure with interpolated charges"
    )

    args = parser.parse_args()

    # do not use the charge part of the lad 
    params = dict(lad.LAD_PARAMS)
    params["wq"] = 0.0

    reference_data = lad.read_reference_lads(args.reference_lad_data, params)

    key = 'r_lad_interpolated_ff_charges'

    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    for ist, st in enumerate(StructureReader(args.input_file)):
        atomic_numbers = [at.atomic_number for at in st.atom]
        carts = st.getXYZ(copy=True)
        charges = lad.interpolate_charges(
            atomic_numbers,
            carts,
            params,
            reference_data,
        )

        for at in st.atom:
            at.property[key] = charges[at.index - 1]

        st.append(args.output_file)

