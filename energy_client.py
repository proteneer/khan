""" Energy client reads coordinates from json file and requests an energy computation """ 

import argparse
import sys
import json
import client_server

from khan.utils.helpers import atomic_number_to_atom_id
from khan.utils.constants import KCAL_MOL_IN_HARTREE, ANGSTROM_IN_BOHR

def read_json(fname):
    """
    Read a geometry from the json file
    and exchange atom id to ANI1 atom type
    Returns json data which can be sent to the energy server
    """
    with open(fname) as fin:
        data = json.load(fin)

    X = data.get("X")
    # need to change atomic numbers to atom id
    for line in X:
        line[0] = atomic_number_to_atom_id(line[0])
        line[1] = line[1]*ANGSTROM_IN_BOHR
        line[2] = line[2]*ANGSTROM_IN_BOHR
        line[3] = line[3]*ANGSTROM_IN_BOHR

    return json.dumps({"X": X})


def parse_args(args):
    """
    Parse commandline arguments and return a namespace
    """

    parser = argparse.ArgumentParser(
        description="request ANI1 energy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "infile",
        action="store",
        default=None,
        help="Input json file"
    )

    parser.add_argument(
        "--host",
        action="store",
        default="nyc-mbp-jacobson.local",
        help="host name for energy server"
    )
    
    parser.add_argument(
        "--port",
        action="store",
        default=5000,
        type=int,
        help="port for energy server"
    )

    parser.add_argument(
       "--outfile",
       action="store",
       default="ani1_output.json",
       help="output filename"
    )
    
    args = parser.parse_args()

    return args


def main():

    args = parse_args(sys.argv)
    send_data = read_json(args.infile)

    s = client_server.connect_socket(args.host, args.port, server=False)
    client_server.send(s, send_data)

    rcv_data = client_server.recieve(s)

    with open(args.outfile, "w") as fout:
        fout.write(rcv_data)

    client_server.close_connection(s) 

if __name__ == "__main__":
    main()
