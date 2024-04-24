import os
import sys
from collections import defaultdict
from numpy import *

name=defaultdict(list);
fin = open("dataset_own/WB_ACR/01/mos.csv","r")               # 返回一个文件对象 
line = fin.readline()               # 调用文件的 readline()方法 
while line: 
    #print(line)                   # 后面跟 ',' 将忽略换行符 
    items = line.strip().split(',')
    #print(items[1])
    name[items[0]].append(float(items[1]))
    #name[items[0]]=items[1]
    line = fin.readline() 
fin.close()
print(name)


fo = open("dataset_own/WB_ACR/01/mos_new.csv","w")
for key,value in name.items():
    name[key]=mean(value)
for key,value in name.items():
    fo.writelines([key,',', str(format(value,'.3f')),'\n'])
    #fo.write("%d\n",% value)
    #fo.write(str(value))
fo.close()
print(name)
