#!/usr/bin/env/python

import pickle


conv_layers = [70,80,90,100]
conv_size = 3

with open('datadimsrelu.pkl','rb') as f0:
    datadims0 = pickle.load(f0)

for cl in conv_layers:
    print(cl)
    datadims = datadims0.copy()
    datadims['conv_field'] = cl + conv_size // 2
    with open(f'conv_layers_{cl}/datadimsrelu.pkl', 'wb') as f:
        pickle.dump(datadims, f)


