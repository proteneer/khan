""" Exceedlingly simple (naive) implementatino of client/server model to compute ANI1 energies """
import socket

ENCODING = 'utf-8'
MAX_BYTES = 1024**2
BACKLOG = 10

def connect_socket(host, port, server=True):
    """
    startup a socket
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    if server:
        s.bind((host, port))
        s.listen(BACKLOG)
        print("connecting server")
    else:
        remote_ip = socket.gethostbyname(host)
        s.connect((remote_ip, port))
        print("connecting client")

    return s

def send(s, data):
    """
    send a string
    """
    s.sendall(bytes(data, encoding=ENCODING))

def recieve(s):
    """
    recieve a string
    """
    rcv_data = s.recv(MAX_BYTES)
    return rcv_data.decode(ENCODING)

def close_connection(s):
    s.close()
