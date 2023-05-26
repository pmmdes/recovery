import numpy as np
import feather
import matplotlib.pyplot as plt
import pandas as pd
import gc
import os
import sys
import random

import socket
import uuid

from time import time
from math import sqrt
from datetime import datetime
from numpy import loadtxt
from PIL import Image


from os.path import exists

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

    print(f'[ {object_type} LOADED IN] {end_time - start_time}s')

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

    image_size = info["size"]

    # Process
    start_time = time()
    figure, iterations = handle_algortihm(info)
    end_time = time()

    print(f'[PROCESSING] Image processed {iterations + 1} times.')
    print(f'[PROCESSING] Time spent: {end_time - start_time}')

    figure = np.reshape(figure, (image_size, image_size), order='F')

    process_end = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')

    name_dir = f'Images/Session-{str(uuid.uuid4())}'
    mkdir_p(name_dir)

    metadata = {
        'Requestor':f'{info["ip"]}',
        'Description': f'Algorithm: {info["alg"]}',
        'Image size': f'{image_size}px',
        'Requested at:': f'{requested_at}', 
        'Construction time': f'{round(end_time - start_time, 2)}s' ,
        'Iterations': f'{iterations}'
    }

    plt.imshow(figure, cmap='gray')
    plt.show()

    ip_formated = info["ip"].replace('.', '')

    plt.imsave(f'{name_dir}/{ip_formated}-{str(uuid.uuid4())}.png', figure, metadata=metadata, cmap='gray')

    print(f'[PROCESSING] Image saved')

    # Clear memory    
    del figure
    gc.collect()

def getIP():
    hostname=socket.gethostname()   
    ipaddr = socket.gethostbyname(hostname) 
    return ipaddr

def generate_process_request():

    info = {
        "size": "",
        "ip": "",
        "alg": "",
        "signal_model": "",
        "signal": "",
        "h_model": "",
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

    return info
    
if __name__ == '__main__':

    
    info = generate_process_request()

    g = feather_format(f'./Signals/{info["signal_model"]}')

    g = calculate_signal(g)

    info["signal"] = g

    process_image(info)

    exit(0)

