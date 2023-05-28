import numpy as np
import feather
import matplotlib.pyplot as plt
import pandas as pd

import os, psutil
from os.path import exists
import gc
import sys
import socket

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

#from memory_profiler import profile

PROCESSING_QUEUE = Queue()

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

def calculate_signal(g):
    
    n = 64

    s = 794 if len(g) > 50000 else 436

    for c in range(n):
        for l in range(s):
            y = 100 + (1/20) * l * sqrt(l)
            g[l + c * s] = g[l + c * s] * y

    return g

def handle_algortihm(info):

    if (info["h_model"] == 1):
        model = "H-1"
    else:
        model = "H-2"

    H = feather_format(f'./Models/{model}')

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

    name_dir = f'Images/Session-{ip_formated}-{info["sessionId"]}'
    mkdir_p(name_dir)

    # Defining metadata of image
    metadata = {
        'Requestor':f'{info["ip"]}',
        'Description': f'Algorithm: {info["alg"]}',
        'Image size': f'{image_size}px',
        'Requested at:': f'{requested_at}', 
        'Construction time': f'{round(end_time - start_time, 2)}s' ,
        'Iterations': f'{iterations}',
        'Session ID': f'{info["sessionId"]}'
    }

    #plt.imshow(figure, cmap='gray')
    #plt.show()

    # Saving image in session folder
    plt.imsave(f'{name_dir}/{ip_formated}-{str(uuid.uuid4())}.png', figure, metadata=metadata, cmap='gray')

    print(f'[PROCESSING] Image saved')

    # Clear memory    
    del figure
    gc.collect()

def getIP():
    hostname=socket.gethostname()   
    ipaddr = socket.gethostbyname(hostname) 
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

    for i in range(amount):
        PROCESSING_QUEUE.put(generate_process_request(sessionId))
        sleep(random.randint(2, 8))

# if __name__ == '__main__':
    
#     #info = generate_process_request()
#     sessionId = str(uuid.uuid4())

#     info = {
#         'size': 30, 
#         'ip': '192.168.56.1', 
#         'alg': 'cgne', 
#         'signal_model': 's30-1', 
#         'signal': '', 
#         'h_model': 2,
#         'sessionId': "a90c7757-1686-4e00-8f71-cbf2d580f08d"
#     } 
    
#     g = feather_format(f'./Signals/{info["signal_model"]}')

#     g = calculate_signal(g)

#     info["signal"] = g
#     sleep(5)

#     # initial_cpu = psutil.cpu_percent(percpu=True)
#     # print('CPU initial:', initial_cpu)
    
#     process_image(info)

#     # final_cpu = psutil.cpu_percent(percpu=True)
#     # print('CPU final:', final_cpu)
#     exit(0)

    ########################################################################

def process_queue():
    while True:
        mem = psutil.virtual_memory()
        free_mem = (mem.available / mem.total) * 100
        print("Free mem: ", free_mem)
        if PROCESSING_QUEUE.qsize() > 0 and free_mem > 50:
            print(f'    [PROCESSING] Current memory usage: {round(100 - free_mem, 2)}%')
            print(f'    [PROCESSING] Signal found in queue')
            process_image(PROCESSING_QUEUE.get())

def main():

    print('[PROCESSING] Processing queue initiated.')
    queue_thread = threading.Thread(target=process_queue, name='Processing Thread')
    queue_thread.start()

    send_random_processes(2)

    exit(0)

if __name__ == '__main__':
    main()
