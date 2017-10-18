import csv
import numpy as np
import scipy
from numpy  import array
from scipy import stats
from scipy.optimize import curve_fit

class Data():

    def __init__(self, labels, _path):
	self.path=_path

	# variable name
	self.xvariable="Dose"
	self.xvar="D"
	self.xunits="Gy"
	self.yvariable="Yield"
	self.yvar="Y"
	self.yunits=""

	self.labels=labels

	self.read_data_csv(self.path)
	print "loaded data file: "+ self.path

    def read_data_csv(self,path):

	ifile  = open(path, "rb")
	dictreader = csv.DictReader(ifile, delimiter=',')
	datadict = {}

	columns_to_read = self.labels[:3]
	for row in dictreader:
	    for column, value in row.iteritems():
		datadict.setdefault(column, []).append(value)
	ifile.close()

	data=[]
	for item in columns_to_read:
	    for key in datadict:
		if item in key:
		    data.append(datadict[key])

	if len(data)==0:
	    return 0
	elif len(data)==1:
	    return 1
	elif len(data)==2:
	    return 2

	column4=[]
	column5=[]
	for i in range(0,len(data[0])):
	    if data[1][i] != 0:
		col1=float(data[1][i])
		col2=float(data[2][i])
		column4.append(col2/col1)
		column5.append(np.sqrt(col2)/col1)
	    else:
		column4.append(0.0)
		column5.append(0.0)
	data.append(column4)
	data.append(column5)

	self.table = array(data).astype(np.float)

	self.numRows=len(self.table[0])
	self.numCols=len(self.labels)
	return 3

    def get_xdata(self):
	return self.table[0]

    def get_ydata(self):
	return self.table[3]

    def get_yerr(self):
	return self.table[4]
