""" energy server handles json requests to compute energy and gradient """ 


from khan.training.trainer_multi_tower import TrainerMultiTower, initialize_module
from khan.data.dataset import RawDataset
import tensorflow as tf
import argparse
import os
import sys
import numpy as np
import json
import client_server

KCAL = 627.509
BOHR = 0.529
ENCODING = 'utf-8'

MAX_BYTES = 1024**2

def parse_args(args):
    """
    Parse commandline arguments and return a namespace
    """

    parser = argparse.ArgumentParser(
        description="server for energy requests",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--work-dir",
        default="~/work",
        help="location of saved NN"
    )
    parser.add_argument(
        "--ani_lib",
        default="/Users/jacobson/projects/ani1_training/khan/gpu_featurizer/ani_cpu.so", 
        help="Location of shared object"
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
        "--debug",
        action="store_true",
        default=False,
        help="print some debugging info"
    )
    
    args = parser.parse_args()

    return args


def main():

    args = parse_args(sys.argv)
    lib_path = os.path.abspath(args.ani_lib)
    initialize_module(lib_path)

    save_dir = os.path.join(args.work_dir, "save")

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        towers = ["/cpu:0"]
        trainer = TrainerMultiTower(
            sess,
            towers,
            layer_sizes=(128, 128, 64, 1)
        )

        trainer.load(save_dir)

        s = client_server.connect_socket(args.host, args.port, server=True)

        if args.debug:
            print("Server listening on port %d" % args.port)

        while True:

            if args.debug:
                print("awaiting connection...")

            conn, addr = s.accept()

            if args.debug:
                print("Connection established...")
            
            while True:

                rcv_data = client_server.recieve(conn)

                print("recieved data", rcv_data)

                if rcv_data:

                    X = json.loads(rcv_data).get('X') 
                    X_np = np.array(X, dtype=np.float32)
                    rd = RawDataset([X_np], [0.0])

                    energy = float(trainer.predict(rd)[0])
                    gradient = list(trainer.coordinate_gradients(rd))[0]
                    natoms, ndim = gradient.shape
                    gradient = [float(g) for g in gradient.reshape(natoms*ndim)]

                    send_data = json.dumps({"energy": energy, "gradient": gradient})

                    print("sending response...")

                    client_server.send(conn, send_data)

                else:
                    break

if __name__ == "__main__":
    main()

