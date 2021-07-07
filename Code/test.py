import csv
import os
import sys

# from GraphSAINT.test2 import do123

# os.chdir('GraphSAINT')
# sys.path.append('.')
# sys.path.append('GraphSAINT')
# import graphsaint.cython_sampler as cy

def read(path):
    with open(path, 'r', newline='') as file:
        data = list(csv.reader(file))
        return data

def abt_buy():
    abt = read('C:\\Users\\shifmal2\\Downloads\\Abt-Buy\\Abt.csv')
    abt = [row[0] for row in abt[1:]]
    buy = read('C:\\Users\\shifmal2\\Downloads\\Abt-Buy\\Buy.csv')
    buy = [row[0] for row in buy[1:]]
    matching = read('C:\\Users\\shifmal2\\Downloads\\Abt-Buy\\abt_buy_perfectMapping.csv')[1:]
    real = [1 for a in matching if a[0] in abt and a[1] in buy]
    d_abt = {}
    d_buy = {}
    for i, j in matching:
        a = d_abt.get(i, [0, []])
        d_abt[i] = [a[0] + 1, a[1] + [(i, j)]]
        b = d_buy.get(j, [0, []])
        d_buy[j] = [b[0] + 1, b[1] + [(i, j)]]
    double_abt = [a for a in d_abt.values() if a[0] >= 2]
    double_buy = [a for a in d_buy.values() if a[0] >= 2]

def cora():
    import xml.etree.ElementTree as ET
    tree = ET.parse('C:\\Users\\shifmal2\\Downloads\\cora-refs\\cora-ref\\fahl-labeled')
    root = tree.getroot()
    print()

if __name__ == '__main__':
    cora()
    print()