import numpy as np
import feather
import matplotlib.pyplot as plt
import pandas as pd

import os, psutil
from os.path import exists
import gc
import sys
import socket as st
import pickle

import uuid
import random

from time import time
from math import sqrt
from datetime import datetime
from numpy import loadtxt
from PIL import Image

from queue import Queue
import threading

from time import sleep

PORT = 5053
SERVER = '127.0.0.1'
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
HEADERSIZE = 10
PROCESSING_QUEUE = Queue()

server = st.socket(st.AF_INET, st.SOCK_STREAM)
server.bind(ADDR)

erro = 1e-4

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

def pickle_format(info):
    msg = pickle.dumps(info)
    return bytes(f'{len(msg):<{HEADERSIZE}}', FORMAT) + msg

def cgne(H, g, image):
    # H: matrix, f: image, g: signal vector
    # dot = scalar product of two matrices
    # zeros = fill a matrix with a given shape with zeros
    
    f_i = np.zeros((image ** 2, 1))
    r_i = g - np.dot(H, f_i)
    p_i = np.dot(np.transpose(H), r_i)
    
    for i in range(0, len(g)):
      # i variables        
      r_d = r_i
      a_i = np.dot(np.transpose(r_d), r_d) / np.dot(np.transpose(p_i), p_i)

      f_i = f_i + a_i * p_i
      h_p = np.dot(H, p_i) 
      r_i = r_i - a_i * h_p
      beta = np.dot(np.transpose(r_i), r_i) / np.dot(np.transpose(r_d), r_d)

      erro_i = abs(np.linalg.norm(r_i, ord=2) - np.linalg.norm(r_d, ord=2))
      
      if erro_i < erro:
        break

      p_i = np.dot(np.transpose(H), r_i) + beta * p_i
    #print('CPU inside cgne:', psutil.cpu_percent())
    return f_i, i

def cgnr(H, g, image):
    f_i = np.zeros((image ** 2, 1))
    r_i = g - np.dot(H, f_i)
    z_i = np.dot(np.transpose(H), r_i)
    p_i = z_i

    for i in range(0, len(g)):
        w_i = np.dot(H, p_i)
        r_d = r_i
        # i variables
        z_norm = np.linalg.norm(z_i, ord=2) ** 2
        a_i =  z_norm / np.linalg.norm(w_i, ord=2) ** 2

        f_i = f_i + a_i * p_i
        r_i = r_i - a_i * w_i
        z_i = np.dot(np.transpose(H), r_i)
        beta = np.linalg.norm(z_i, ord=2) ** 2/ z_norm

        erro_i = abs(np.linalg.norm(r_i, ord=2) - np.linalg.norm(r_d, ord=2))

        if erro_i < erro:
            break

        p_i = z_i + beta * p_i
    return f_i, i

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def handle_algortihm(info):

    if (info["h_model"] == 1):
        model = "H-1"
    else:
        model = "H-2"

    H = feather_format(f'./Server/Models/{model}')

    if (info["alg"] == "cgne"):
        figure, iterations = cgne(H, info["signal"], info["size"])
    else:
        figure, iterations = cgnr(H, info["signal"], info["size"])

    del H
    gc.collect()

    return figure, iterations 

def process_image(info):

    requested_at = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')

    # Processing the image
    start_time = time()
    figure, iterations = handle_algortihm(info)
    end_time = time()

    #print(f'[PROCESSING] Image processed {iterations + 1} times.')
    #print(f'[PROCESSING] Time spent: {end_time - start_time}')

    image_size = info["size"]
    figure = np.reshape(figure, (image_size, image_size), order='F')

    process_end = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
    
    # Creating directory of session
    ip_formated = info["ip"].replace('.', '')

    name_dir = f'./Server/Images/Session-{ip_formated}-{info["sessionId"]}'
    mkdir_p(name_dir)

    # Defining metadata of image
    metadata = {
        'requestor':f'{info["ip"]}',
        'algorithm': f'{info["alg"]}',
        'image_size': f'{image_size}px',
        'requested_at': f'{requested_at}', 
        'construction_time': f'{round(end_time - start_time, 2)}s' ,
        'iterations': f'{iterations}',
        'sessionId': f'{info["sessionId"]}'
    }

    #plt.imshow(figure, cmap='gray')
    #plt.show()

    # Saving image in session folder
    plt.imsave(f'{name_dir}/{str(uuid.uuid4())}.png', figure, metadata=metadata, cmap='gray')

    print(f'    [PROCESSING] Image from request [{info["requestNumber"]}] saved.')

    # Clear memory    
    del figure
    gc.collect()

def process_queue():
    while True:
        mem = psutil.virtual_memory()
        free_mem = (mem.available / mem.total) * 100
        #print("Free mem: ", free_mem)
        if PROCESSING_QUEUE.qsize() > 0 and free_mem > 25:
            print(f'    [PROCESSING] New process being handled. Current memory usage: {round(100 - free_mem, 2)}%')            

            process_image(PROCESSING_QUEUE.get())
            

def handle_info(info, conn):

    print(f'    [PROCESSING] New file added into the queue.')
    PROCESSING_QUEUE.put(info)

    # Just some status info
    queue_position = PROCESSING_QUEUE.qsize()
    requestNumber = info["requestNumber"]

    msg = pickle_format(f"[SERVER] Request [{requestNumber}] putted in the queue. Position: {queue_position}")

    conn.send(msg)

def handle_client(conn, addr):

    print(f"[CONNECTION] IP: {addr[0]}, Port: {addr[1]} connected.")

    full_msg = b''
    new_msg = True

    connected = True
    while connected:
        msg = conn.recv(1024)
        if msg != b'':
            if new_msg:
                msglen = int(msg[:HEADERSIZE])
                new_msg = False

            full_msg += msg

            if len(full_msg) - HEADERSIZE == msglen:
                info = pickle.loads(full_msg[HEADERSIZE:])
                new_msg = True
                full_msg = b''
                
                if (info == DISCONNECT_MESSAGE): 
                    #print("recebi um disconnect")
                    connected = False
                else: 
                    #print("tudo certo")
                    info["port"] = addr[1]
                    handle_info(info, conn)
    conn.close()
    print(f"[CONNECTION] IP: {addr[0]}, Port: {addr[1]} disconnected.")

def start_server_listener():
    server.listen(5)

    print(f"[LISTENING] Server is listening on {SERVER}")

    while True:
        conn, addr = server.accept()

        response_thread = threading.Thread(target=handle_client, name=f'Connection thread',args=(conn, addr))
        response_thread.start()

def main():

    print('[PROCESSING] Processing queue initiated.')
    queue_thread = threading.Thread(target=process_queue, name='Processing Thread')
    queue_thread.start()

    print("[START] Server is starting...")
    start_server_listener()

    exit(0)

if __name__ == '__main__':
    main()