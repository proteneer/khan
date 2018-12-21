import argparse
import os
import sys

from schrodinger.structure import StructureReader, StructureWriter
from schrodinger.infra import mm

FFLD_VERSION = 16

# Initialize mm for force field partial charges
def ff_charges(st, key, mmffld_handle):
    """
    Add ff charges as a property with key
    """
    mm.mmffld_enterMol(mmffld_handle, st.handle)
    mm.mmffld_deleteMol(mmffld_handle, st.handle)
    for atom in st.atom:
        atom.property[key] = atom.partial_charge

def initialize_mmffld():
    """
    initialize mmffld and return handle
    """
    mm.mmerr_initialize()
    error_handler = mm.MMERR_DEFAULT_HANDLER
    mm.mmlewis_initialize(error_handler)
    mm.mmffld_initialize(error_handler)

    return mm.mmffld_new(FFLD_VERSION)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Add ff charges to structures",
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'input_file',
        nargs='+',
        type=str,
        help="input mae files containing datset of structures"
    )

    args = parser.parse_args()

    mmffld_handle = initialize_mmffld()
    print("using ff: %s" % mm.mmcommon_get_ffld_name(FFLD_VERSION))
    for fname in args.input_file: 
        if fname.endswith(".mae"):

            outname = fname[:-4] + "_ff_charges.mae"
            maeout = StructureWriter(outname)
            if os.path.exists(outname):
                os.remove(outname)

            for ist, st in enumerate(StructureReader(fname)):

                if ist % 100 == 0:
                    print("on structure %d" % ist)

                try:
                    ff_charges(st, 'r_lad_ff_charges', mmffld_handle)
                except Exception as e:
                    print("skipping structure do to exception %s" % e)
                else:
                    maeout.append(st)

            maeout.close()

    print("done")
