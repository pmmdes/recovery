import numpy as np
from numpy import loadtxt
import pandas as pd

import feather

import os, psutil
from os.path import exists
import gc
import sys
import socket as st

import uuid
import random

from time import time
from math import sqrt
from datetime import datetime
from time import sleep

import pickle

SERVER = '127.0.0.1'
PORT = 5053
ADDR = (SERVER, PORT)
FORMAT = "utf-8"
HEADERSIZE = 10
DISCONNECT_MESSAGE = "!DISCONNECT"

def feather_format(path):

    if not (exists(f'{path}.feather')):
        print(f"[Converting {path} into a feather format]")
        df = pd.read_csv(f'{path}.csv', header=None)
        df.columns = df.columns.astype(str)
        df = df.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
        df.to_feather(f'{path}.feather')

    start_time = time()    
    file = pd.read_feather(f"{path}.feather").to_numpy(dtype=float)
    end_time = time()

    object_type = "MODEL" if os.path.split(path)[0][2:] == "Models" else "SIGNAL"

    #print(f'[ {object_type} LOADED IN] {end_time - start_time}s')
    #print(f'[ {object_type} LOADED]')

    return file

def calculate_signal(g):
    
    n = 64

    s = 794 if len(g) > 50000 else 436

    for c in range(n):
        for l in range(s):
            y = 100 + (1/20) * l * sqrt(l)
            g[l + c * s] = g[l + c * s] * y

    return g

def receive_message(client):
    data = b''
    while True:
        msg = client.recv(1024)
        if not msg: break
        data += msg
    return data

def pickle_format(info):
    msg = pickle.dumps(info)
    return bytes(f'{len(msg):<{HEADERSIZE}}', FORMAT) + msg

def getIP():
    hostname=st.gethostname()   
    ipaddr = st.gethostbyname(hostname) 
    return ipaddr

def generate_process_request(sessionId):

    info = {
        "size": "",
        "ip": "",
        "alg": "",
        "signal_model": "",
        "signal": "",
        "h_model": "",
        "sessionId": sessionId
    } 

    sizes = [30, 60]

    size_choosed = random.choice(sizes)

    #       s30-1 // s30-2 // s60-1 // s60-2
    signals = [ f"s{size_choosed}-1", f"s{size_choosed}-2"]

    signal_choosed = random.choice(signals)

    algorithms = ["cgne", "cgnr"]

    alg_choosed = random.choice(algorithms)

    model = 1 if size_choosed == 60 else 2

    info["size"] = size_choosed
    info["alg"] = alg_choosed
    info["signal_model"] = signal_choosed
    info["h_model"] = model
    info["ip"] = getIP()

    g = feather_format(f'./Signals/{info["signal_model"]}')

    g = calculate_signal(g)

    info["signal"] = g

    return info

def send_random_processes(amount):

    sessionId = str(uuid.uuid4())

    client = st.socket(st.AF_INET, st.SOCK_STREAM)
    client.connect(ADDR)

    for i in range(amount):

        # client = st.socket(st.AF_INET, st.SOCK_STREAM)
        # client.connect(ADDR)
        
        msg = pickle_format( generate_process_request(sessionId) )
        client.send(msg)
        print("Requisição enviada. Aguarde o processamento.")
        

        sleep(random.randint(2, 8))

    #info = {"disconnect" == DISCONNECT_MESSAGE}
    client.send(pickle_format(DISCONNECT_MESSAGE))
    client.close()

def main():

    send_random_processes(2)

    # client = st.socket(st.AF_INET, st.SOCK_STREAM)
    # client.connect(ADDR) 

    # while True:
    #     #print("Checking for messages")
    #     data = receive_message(client)
    
    #     if len(data) > 0:
    #         data = pickle.loads(data[HEADERSIZE:])

    #         print(data)

    # exit(0)

if __name__ == '__main__':
    main()