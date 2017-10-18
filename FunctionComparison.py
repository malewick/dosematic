
# coding: utf-8

# # Imports

# Core functionality is GTK with Matplotlib plug-in.
# Numpy and Scipy as helpers.

import csv
from datetime import datetime

import pygtk
pygtk.require('2.0')
import gtk
from gtk import gdk
import pango

import matplotlib
matplotlib.use('GTKAgg')  # or 'GTK'
import matplotlib.pyplot as plt
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
from matplotlib.backends.backend_gtkagg import NavigationToolbar2GTKAgg as NavigationToolbar

from matplotlib.figure import Figure

import numpy as np
import scipy
from numpy  import array
from scipy import stats
from scipy.optimize import curve_fit

import LogConsole


class Function():
    def __init__(self):
	self.params=[]
        self.std_err=[]
        self.p_value=[]
        self.cov_mtx=[]
        self.rss=0
        self.rmse=0
	self.dof=0
	self.func=self.quadratic

	self.mode=""

    def linear(self,x):
	"""Linear Function"""
	return self.params[0]*x + self.params[1]

    def quadratic(self,x):
	"""Quadratic Function"""
	return self.params[0]*x*x + self.params[1]*x + self.params[2]

    def get_parameters(self):
	"""Return the list of parameters"""
	return self.params, self.std_err, self.p_value, self.cov_mtx, self.rss, self.rmse, self.dof

    def set_parameters(self, params, std_err, p_value, cov_mtx, rss, rmse, dof):
	"""Set the function parameters"""
	self.params, self.std_err, self.p_value, self.cov_mtx, self.rss, self.rmse, self.dof = \
	    params, std_err, p_value, cov_mtx, rss, rmse, dof
	if len(params)==2:
	    self.func=linear
	    self.mode = "linear"
	elif len(params)==3:
	    self.func=self.quadratic
	    self.mode = "quadratic"

    def uncertainty(self,x) :
	"""Calculation of uncertainty of points"""
	if len(self.std_err)==3:
	    return self.std_err[2] + x*self.std_err[1] + x*x*self.std_err[0]
	elif len(self.std_err)==2:
	    return self.std_err[1] + x*self.std_err[0]
	else:
	    return 0

    def confidence_points(self,x) :
	"""Calculation of the confidence interval of data points"""
	return 1.96*self.uncertainty(x)

    def load_function(self, path) :
	f = open(path, 'rt')
	try:
	    reader = csv.reader(f)
	    l=list(reader)
	    print l
	    params=[float(i) for i in l[0]]
	    std_err=[float(i) for i in l[1]]
	    p_value=[float(i) for i in l[2]]
	    cov_mtx=[[float(i) for i in l[3]],[float(i) for i in l[4]],[float(i) for i in l[5]]]
	    rss=float(l[6][0])
	    rmse=float(l[6][1])
	    dof=float(l[6][2])

	    self.set_parameters(params, std_err, p_value, cov_mtx, rss, rmse, dof)

	finally:
	    f.close()

class FunctionPanel():
    def __init__(self, name, fun, plotter, plotpar, parent):

        label_Y = gtk.image_new_from_file('./equations/Yfunc1.png')
        self.entry_c = gtk.Entry()
        self.entry_c.set_width_chars(5)
        label_c = gtk.Label('c: ')
        self.entry_a = gtk.Entry()
        self.entry_a.set_width_chars(5)
        label_alpha = gtk.Label()
        label_alpha.set_use_markup(True)
        label_alpha.set_markup('&#945;: ')
        self.entry_b = gtk.Entry()
        self.entry_b.set_width_chars(5)
        label_beta = gtk.Label()
        label_beta.set_use_markup(True)
        label_beta.set_markup('&#946;: ')
        self.entry_sc = gtk.Entry()
        self.entry_sc.set_width_chars(5)
        label_sc = gtk.Label()
        label_sc.set_use_markup(True)
        label_sc.set_markup('&#963;(c): ')
        self.entry_sa = gtk.Entry()
        self.entry_sa.set_width_chars(5)
        label_salpha = gtk.Label()
        label_salpha.set_use_markup(True)
        label_salpha.set_markup('&#963;(&#945;): ')
        self.entry_sb = gtk.Entry()
        self.entry_sb.set_width_chars(5)
        label_sbeta = gtk.Label()
        label_sbeta.set_use_markup(True)
        label_sbeta.set_markup('&#963;(&#946;): ')
        table_f = gtk.Table(6,3)
        table_f.attach(label_c,0,1,0,1)
        table_f.attach(self.entry_c,1,2,0,1)
        table_f.attach(label_alpha,0,1,1,2)
        table_f.attach(self.entry_a,1,2,1,2)
        table_f.attach(label_beta,0,1,2,3)
        table_f.attach(self.entry_b,1,2,2,3)
        table_f.attach(label_sc,4,5,0,1)
        table_f.attach(self.entry_sc,5,6,0,1)
        table_f.attach(label_salpha,4,5,1,2)
        table_f.attach(self.entry_sa,5,6,1,2)
        table_f.attach(label_sbeta,4,5,2,3)
        table_f.attach(self.entry_sb,5,6,2,3)
        vruler = gtk.VSeparator()
        table_f.attach(vruler,3,4,0,3,xpadding=10)
        left_box = gtk.VBox(False,5)
        left_box.pack_start(label_Y, False, False, 8)
        left_box.pack_start(table_f, False, False)
        self.hbox = gtk.HBox()
        self.hbox.pack_start(left_box, False, False)

	# setting up the values in textfields
	if fun.mode == "quadratic" :
	    self.entry_c.set_text(     '%.3f' % (fun.params[2] ))
	    self.entry_sc.set_text(    '%.3f' % (fun.std_err[2]))
	    self.entry_a.set_text( '%.3f' % (fun.params[1] ))
	    self.entry_sa.set_text('%.3f' % (fun.std_err[1]))
	    self.entry_b.set_text(  '%.3f' % (fun.params[0] ))
	    self.entry_sb.set_text( '%.3f' % (fun.std_err[0]))

            self.entry_b. set_property("editable",True)
            self.entry_sb.set_property("editable",True)
            self.entry_b.modify_base(gtk.STATE_NORMAL, gtk.gdk.color_parse("#FFFFFF"))
            self.entry_sb.modify_base(gtk.STATE_NORMAL, gtk.gdk.color_parse("#FFFFFF"))

	elif fun.mode == "linear" :
	    self.entry_c.set_text(     '%.3f' % (fun.params[1] ))
	    self.entry_sc.set_text(    '%.3f' % (fun.std_err[1]))
	    self.entry_a.set_text( '%.3f' % (fun.params[0] ))
	    self.entry_sa.set_text('%.3f' % (fun.std_err[0]))
	    self.entry_b.set_text(  '%.3f' % (0.0 ))
	    self.entry_sb.set_text( '%.3f' % (0.0 ))

            self.entry_b. set_property("editable",False)
            self.entry_sb.set_property("editable",False)
            self.entry_b.modify_base(gtk.STATE_NORMAL, gtk.gdk.color_parse("#E1E1E1"))
            self.entry_sb.modify_base(gtk.STATE_NORMAL, gtk.gdk.color_parse("#E1E1E1"))

class ButtonsGrid() :
    def __init__(self,context,f_fun,g_fun,plotter,plotpar):

	self.context=context

	load1 = gtk.Button('Load Function f')
	load2 = gtk.Button('Load Function g')
	panel1 = FunctionPanel("f",f_fun,plotter,plotpar,self)
	panel2 = FunctionPanel("g",g_fun,plotter,plotpar,self)

	label_range = gtk.Label('Function ranges: [x1,x2]')
	label_x1 = gtk.Label('x1 = ')
	label_x2 = gtk.Label('x2 = ')
        self.entry_x1 = gtk.Entry()
        self.entry_x1.set_width_chars(5)
	self.entry_x1.set_text(str(plotter.x1))
        self.entry_x2 = gtk.Entry()
        self.entry_x2.set_width_chars(5)
	self.entry_x2.set_text(str(plotter.x2))

	hrule1 = gtk.HSeparator()
	hrule2 = gtk.HSeparator()

	grid = gtk.Table(2,11)
	grid.attach(load1,        0,2,1,2)
	grid.attach(panel1.hbox,  0,2,2,3)
	grid.attach(hrule1,       0,2,3,4)
	grid.attach(load2,        0,2,4,5)
	grid.attach(panel2.hbox,  0,2,5,6)
	grid.attach(hrule2,       0,2,6,7)
	grid.attach(label_range,  0,2,7,8)
	grid.attach(label_x1,     0,1,8,9)
	grid.attach(self.entry_x1,1,2,8,9)
	grid.attach(label_x2,     0,1,9,10)
	grid.attach(self.entry_x2,1,2,9,10)

	load1.connect('clicked',self.load_function_chooser,f_fun,plotter,plotpar)
        load2.connect('clicked',self.load_function_chooser,g_fun,plotter,plotpar)
        self.entry_x1.connect("activate",self.on_xentry_changed,plotter,plotpar)
        self.entry_x2.connect("activate",self.on_xentry_changed,plotter,plotpar)

	self.grid= grid
	self.f = f_fun
	self.g = g_fun

    def get_grid(self):
	return self.grid


    def load_function_chooser(self,button,function,plotter,plotpar) :
        """Load function from file."""
        file_chooser = gtk.FileChooserDialog("Open...", self.context, gtk.FILE_CHOOSER_ACTION_OPEN, (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_OPEN, gtk.RESPONSE_OK))
        response = file_chooser.run()
        path=''
        if response == gtk.RESPONSE_OK :
	    path = file_chooser.get_filename()
	    function.load_function(path)
	    self.context.log('Loaded curve from file:   ' + path)
            file_chooser.destroy()
	    plotter.plotting()
	    plotpar.plotting()
	    self.context.asses_function_similarity(self.f, self.g)
        else : 
            file_chooser.destroy()

    def on_xentry_changed(self,entry,plotter) :
	"""Changes function parameters when entry was changed."""
	plotter.x1 = float(self.entry_x1.get_text())
	plotter.x2 = float(self.entry_x2.get_text())
	self.context.log("Range changed: x1 = " + self.entry_x1.get_text() + ", x2 = " + self.entry_x2.get_text())
	plotter.plotting()
	plotpar.plotting()


class Plotter() :
    def __init__(self,context,f_function,g_function):
	self.context=context
	self.f_function=f_function
	self.g_function=g_function

        self.fig = Figure(figsize=(6, 4))		# create fig
        self.canvas = FigureCanvas(self.fig)		# a gtk.DrawingArea
        self.canvas.set_size_request(600,600)		# set min size
        self.markers = ['.',',','+','x','|','_','o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']
        self.colors = ['black','blue','green','red','cyan','magenta','yellow','purple','white']
        self.pstyle = ['bmh','s','6','red','0.8','black','2','black','0.3','','25','','','20','15','','','20','15']

        self.styledict = {}
        self.styledict["style"]='bmh'
        self.styledict["point_style"]='s'
        self.styledict["point_size"]='6'
        self.styledict["point_color"]='red'
        self.styledict["point_alpha"]='0.8'
        self.styledict["line_color"]='black'
        self.styledict["line_width"]='2'
        self.styledict["band_color"]='black'
        self.styledict["band_alpha"]='0.3'
        self.styledict["title_size"]='8'
        self.styledict["xtitle_size"]='8'
        self.styledict["xlabel_size"]='8'
        self.styledict["ytitle_size"]='8'
        self.styledict["ylabel_size"]='8'

        self.nselec = [1,12,5,3,-1,0,-1,0,-1,-1,-1,-1,-1,-1]
        self.plot_labels = ["", "x", "f(x)", "", "g(x)"]

        self.points_toggle=1
        self.function_toggle=1
        self.err_toggle=1
        self.ci_func_toggle=1
        self.ci_points_toggle=1
        toolbar = NavigationToolbar(self.canvas, self)
        toolbarbox = gtk.HBox()
        image = gtk.Image()
        image.set_from_stock(gtk.STOCK_PROPERTIES, gtk.ICON_SIZE_LARGE_TOOLBAR)
        options_button = gtk.Button()
        options_button.add(image)
        image2 = gtk.Image()
        image2.set_from_stock(gtk.STOCK_REFRESH, gtk.ICON_SIZE_LARGE_TOOLBAR)
        refresh_button = gtk.Button()
        refresh_button.add(image2)
        toolbarbox.pack_start(toolbar, True, True)
        toolbarbox.pack_end(options_button, False, True)
        toolbarbox.pack_end(refresh_button, False, True)
	self.vbox = gtk.VBox()
        self.vbox.pack_start(toolbarbox, False, False)
        self.vbox.pack_start(self.canvas, True, True)

	self.x1 = -0.1
	self.x2 = 20.1

        # signals
        options_button.connect('clicked',self.mpl_options)
        refresh_button.connect('clicked',self.on_refresh_clicked)

    def mpl_options(self,button) :
	"""Create GTKDialog containing options for plotting and connect signals."""
        mpl_options_dialog = MPLOptions(self.context,self)

    def on_refresh_clicked(self,button) :
	"""Refresh canvas - plot everything again"""
        self.plotting()

    def plotvline(self, **kwargs):
	self.ax1.axvline(**kwargs)

    def plothline(self, **kwargs):
	self.ax1.axhline(**kwargs)

    def replot(self):
	self.canvas.draw()

    def plotting(self):
	"""Generating matplotlib canvas"""

	plt.style.use(self.pstyle[0])
	
	self.ax1 = self.fig.add_subplot(311)
	self.ax1.clear()
	
	self.ax1.set_title("f(x) and g(x)", fontsize=self.pstyle[10])
	self.ax1.set_xlabel("", fontsize=int(self.pstyle[13]))
	self.ax1.set_ylabel("", fontsize=int(self.pstyle[17]))
	self.ax1.tick_params(axis='x', which='both', labelsize=int(self.pstyle[14]))
	self.ax1.tick_params(axis='y', which='both', labelsize=int(self.pstyle[18]))

	x = np.arange(self.x1, self.x2, 0.05)

	y = self.f_function.func(x)
	self.ax1.plot(x,y, color=self.pstyle[5], marker='.', linestyle='None', markersize=float(self.pstyle[6]))

	upper =  self.f_function.func(x) + self.f_function.confidence_points(x)
	lower =  self.f_function.func(x) - self.f_function.confidence_points(x)
	self.ax1.fill_between(x, lower, upper, facecolor='blue', alpha=float(self.pstyle[8]))

	upper =  self.f_function.func(x) + self.f_function.uncertainty(x)
	lower =  self.f_function.func(x) - self.f_function.uncertainty(x)
	self.ax1.fill_between(x, lower, upper, facecolor='green', alpha=float(self.pstyle[8]))

	y = self.g_function.func(x)
	self.ax1.plot(x,y, color=self.pstyle[5], marker='.', linestyle='None', markersize=float(self.pstyle[6]))

	upper =  self.g_function.func(x) + self.g_function.confidence_points(x)
	lower =  self.g_function.func(x) - self.g_function.confidence_points(x)
	self.ax1.fill_between(x, lower, upper, facecolor='blue', alpha=float(self.pstyle[8]))

	upper =  self.g_function.func(x) + self.g_function.uncertainty(x)
	lower =  self.g_function.func(x) - self.g_function.uncertainty(x)
	self.ax1.fill_between(x, lower, upper, facecolor='green', alpha=float(self.pstyle[8]))
	self.ax1.tick_params(axis='x',labelbottom='off')


	self.ax2 = self.fig.add_subplot(312)
	self.ax2.clear()
	
	self.ax2.set_title(self.plot_labels[0], fontsize=self.pstyle[10])
	self.ax2.set_xlabel("", fontsize=int(self.pstyle[13]))
	self.ax2.set_ylabel(self.plot_labels[2]+' / '+self.plot_labels[4], fontsize=int(self.pstyle[17]))
	self.ax2.tick_params(axis='x', which='both', labelsize=int(self.pstyle[14]))
	self.ax2.tick_params(axis='y', which='both', labelsize=int(self.pstyle[18]))
	self.ax2.tick_params(axis='x',labelbottom='off')

	x = np.arange(self.x1, self.x2, 0.05)

	ly=[] 
	for i in x:
	    if self.g_function.func(i):
		ly.append(self.f_function.func(i)/self.g_function.func(i))
	    else:
		ly.append(0.)
	y=array(ly)
	self.ax2.plot(x,y, color=self.pstyle[5], marker='.', linestyle='None', markersize=float(self.pstyle[6]))


	self.ax3 = self.fig.add_subplot(313)
	self.ax3.clear()
	
	self.ax3.set_title(self.plot_labels[0], fontsize=self.pstyle[10])
	self.ax3.set_xlabel(self.plot_labels[1], fontsize=int(self.pstyle[13]))
	self.ax3.set_ylabel(self.plot_labels[2]+' - '+self.plot_labels[4], fontsize=int(self.pstyle[17]))
	self.ax3.tick_params(axis='x', which='both', labelsize=int(self.pstyle[14]))
	self.ax3.tick_params(axis='y', which='both', labelsize=int(self.pstyle[18]))

	x = np.arange(self.x1, self.x2, 0.05)

	ly=[] 
	for i in x:
	    ly.append(self.f_function.func(i) - self.g_function.func(i))
	y=array(ly)
	self.ax3.plot(x,y, color=self.pstyle[5], marker='.', linestyle='None', markersize=float(self.pstyle[6]))


	self.fig.subplots_adjust(left=0.12, right=0.97, top=0.94, bottom=0.11, hspace=0.17)

	self.canvas.draw()




class Plotpar() :
    def __init__(self,context,f_function,g_function):
	self.context=context
	self.f_function=f_function
	self.g_function=g_function

        self.fig = Figure(figsize=(6, 4))		# create fig
        self.canvas = FigureCanvas(self.fig)		# a gtk.DrawingArea
        self.canvas.set_size_request(600,600)		# set min size
        self.markers = ['.',',','+','x','|','_','o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']
        self.colors = ['black','blue','green','red','cyan','magenta','yellow','purple','white']
        self.pstyle = ['bmh','s','6','red','0.8','black','2','black','0.3','','25','','','20','15','','','20','15']

        self.styledict = {}
        self.styledict["style"]='bmh'
        self.styledict["point_style"]='s'
        self.styledict["point_size"]='6'
        self.styledict["point_color"]='red'
        self.styledict["point_alpha"]='0.8'
        self.styledict["line_color"]='black'
        self.styledict["line_width"]='2'
        self.styledict["band_color"]='black'
        self.styledict["band_alpha"]='0.3'
        self.styledict["title_size"]='8'
        self.styledict["xtitle_size"]='8'
        self.styledict["xlabel_size"]='8'
        self.styledict["ytitle_size"]='8'
        self.styledict["ylabel_size"]='8'

        self.nselec = [1,12,5,3,-1,0,-1,0,-1,-1,-1,-1,-1,-1]
        self.plot_labels = ["", "x", "f(x)", "", "g(x)"]

        toolbar = NavigationToolbar(self.canvas, self)
        toolbarbox = gtk.HBox()
        image = gtk.Image()
        image.set_from_stock(gtk.STOCK_PROPERTIES, gtk.ICON_SIZE_LARGE_TOOLBAR)
        options_button = gtk.Button()
        options_button.add(image)
        image2 = gtk.Image()
        image2.set_from_stock(gtk.STOCK_REFRESH, gtk.ICON_SIZE_LARGE_TOOLBAR)
        refresh_button = gtk.Button()
        refresh_button.add(image2)
        toolbarbox.pack_start(toolbar, True, True)
        toolbarbox.pack_end(options_button, False, True)
        toolbarbox.pack_end(refresh_button, False, True)
	self.vbox = gtk.VBox()
        self.vbox.pack_start(toolbarbox, False, False)
        self.vbox.pack_start(self.canvas, True, True)

	self.x1 = -0.1
	self.x2 = 20.1

        # signals
        options_button.connect('clicked',self.mpl_options)
        refresh_button.connect('clicked',self.on_refresh_clicked)

    def mpl_options(self,button) :
	"""Create GTKDialog containing options for plotting and connect signals."""
        mpl_options_dialog = MPLOptions(self.context,self)

    def on_refresh_clicked(self,button) :
	"""Refresh canvas - plot everything again"""
        self.plotting()

    def plotvline(self, **kwargs):
	self.ax1.axvline(**kwargs)

    def plothline(self, **kwargs):
	self.ax1.axhline(**kwargs)

    def replot(self):
	self.canvas.draw()

    def plotting(self):
	"""Generating matplotlib canvas"""

	plt.style.use(self.pstyle[0])

	f_kwargs = {
		'color' : 'black',
		'fmt' : 's',
		'ecolor' : 'black',
		'elinewidth' : 1.5,
		'capsize' : 4.5,
		'capthick' : 1.0,
		'markersize' : 10
		}

	g_kwargs = {
		'color' : 'red',
		'fmt' : 's',
		'ecolor' : 'red',
		'elinewidth' : 1.5,
		'capsize' : 4.5,
		'capthick' : 1.0,
		'markersize' : 10
		}
	
	if self.f_function.mode=="quadratic" or self.g_function.mode=="quadratic":

	    self.ax1 = self.fig.add_subplot(131)
	    self.ax1.clear()
	    self.ax2 = self.fig.add_subplot(132)
	    self.ax2.clear()
	    self.ax3 = self.fig.add_subplot(133)
	    self.ax3.clear()

	    if self.f_function.mode=="quadratic":
		fc = self.f_function.params[2] 
		fa = self.f_function.params[1] 
		fb = self.f_function.params[0] 
		sfc =self.f_function.std_err[2] 
		sfa =self.f_function.std_err[1] 
		sfb =self.f_function.std_err[0] 
	    else:
		fc = self.f_function.params[1] 
		fa = self.f_function.params[0] 
		fb = 0.0 
		sfc =self.f_function.std_err[1] 
		sfa =self.f_function.std_err[0] 
		sfb =0.0

	    if self.g_function.mode=="quadratic":
		gc = self.g_function.params[2] 
		ga = self.g_function.params[1] 
		gb = self.g_function.params[0] 
		sgc =self.g_function.std_err[2] 
		sga =self.g_function.std_err[1] 
		sgb =self.g_function.std_err[0] 
	    else:
		gc = self.g_function.params[1] 
		ga = self.g_function.params[0] 
		gb = 0.0 
		sgc =self.g_function.std_err[1] 
		sga =self.g_function.std_err[0] 
		sgb =0.0
 
	    fx = [0]
	    fy = [fc]
	    fe = [sfc]
	    self.ax1.errorbar(fx,fy,fe, **f_kwargs)
	    gx = [1]
	    gy = [gc]
	    ge = [sgc]
	    self.ax1.errorbar(gx,gy,ge, **g_kwargs)

	    fx = [0]
	    fy = [fa]
	    fe = [sfa]
	    self.ax2.errorbar(fx,fy,fe, **f_kwargs)
	    gx = [1]
	    gy = [ga]
	    ge = [sga]
	    self.ax2.errorbar(gx,gy,ge, **g_kwargs)

	    fx = [0]
	    fy = [fb]
	    fe = [sfb]
	    self.ax3.errorbar(fx,fy,fe, **f_kwargs)
	    gx = [1]
	    gy = [gb]
	    ge = [sgb]
	    self.ax3.errorbar(gx,gy,ge, **g_kwargs)

	    plt.setp([self.ax1,self.ax2,self.ax3], xticks=[it for it in range(2)], xticklabels=['f', 'g'])

	    self.ax1.set_xlim([-0.5,1.5])
	    self.ax2.set_xlim([-0.5,1.5])
	    self.ax3.set_xlim([-0.5,1.5])

	elif self.f_function.mode=="linear" and self.g_function.mode=="linear":

	    self.ax1 = self.fig.add_subplot(121)
	    self.ax1.clear()
	    self.ax2 = self.fig.add_subplot(122)
	    self.ax2.clear()

	    fc = self.f_function.params[1] 
	    fa = self.f_function.params[0] 
	    sfc =self.f_function.std_err[1] 
	    sfa =self.f_function.std_err[0] 

	    gc = self.g_function.params[1] 
	    ga = self.g_function.params[0] 
	    sgc =self.g_function.std_err[1] 
	    sga =self.g_function.std_err[0] 
 
	    fx = [0]
	    fy = [fc]
	    fe = [sfc]
	    self.ax1.errorbar(fx,fy,fe, **f_kwargs)
	    gx = [1]
	    gy = [gc]
	    ge = [sgc]
	    self.ax1.errorbar(gx,gy,ge, **g_kwargs)

	    fx = [0]
	    fy = [fa]
	    fe = [sfa]
	    self.ax2.errorbar(fx,fy,fe, **f_kwargs)
	    gx = [1]
	    gy = [ga]
	    ge = [sga]
	    self.ax2.errorbar(fx,fy,fe, **g_kwargs)

	    plt.setp([self.ax1,self.ax2,self.ax3], xticks=[it for it in range(2)], xticklabels=['f', 'g'])

	    self.ax1.set_xlim([-0.5,1.5])
	    self.ax2.set_xlim([-0.5,1.5])

	self.fig.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.1, wspace=0.30)
	self.canvas.draw()







class UserInterface(gtk.Window):
        
    def __init__(self):
	"""Init function with the whole GUI declaration and signals"""
        
        gtk.Window.__init__(self)
        self.set_default_size(800, 800)
        self.connect('destroy', lambda win: gtk.main_quit())
        
        self.set_title('DOSEMATIC v1.0 -- Function Comparison')

        # main layout container
        main_eb = gtk.EventBox()

        # horizontal box
        hbox = gtk.HBox(False, 8)
        # vertical box
        VBOX = gtk.VBox(False, 0)

        main_eb.add(VBOX)
        #main_eb.modify_bg(gtk.STATE_NORMAL, gtk.gdk.Color(red=60000,green=60000,blue=60000))
        self.add(main_eb)
        self.vbox1 = gtk.VBox(False,8)
        self.vbox2 = gtk.VBox(False,8)
        hbox.pack_start(self.vbox1, True, True)
        hbox.pack_start(self.vbox2, True, True)

        top_band = gtk.HBox()
        bottom_band = gtk.HBox()
        top_eb = gtk.EventBox()
        bottom_eb = gtk.EventBox()
        top_eb.add(top_band)
        bottom_eb.add(bottom_band)
        top_eb.modify_bg(gtk.STATE_NORMAL, gtk.gdk.Color(0,0,0))
        bottom_eb.modify_bg(gtk.STATE_NORMAL, gtk.gdk.Color(0,0,0))
        l1 = gtk.Label('DOSEMATIC v1.0 --- beta testing --- dummy module')
        l2 = gtk.Label('author: Maciej Lewicki                                       mlewicki@ift.uni.wroc.pl,   malewick@cern.ch')
        top_band.add(l1)
        bottom_band.add(l2)
        hruler = gtk.HSeparator()
        hruler2 = gtk.HSeparator()
        VBOX.pack_start(top_eb,False,False)
        VBOX.pack_start(hruler,False,True,5)
        VBOX.pack_start(hbox,True,True)
        VBOX.pack_start(hruler2,False,True,5)
        VBOX.pack_end(bottom_eb,False,False)   

        # TEXT SCREEN______________________________________________________
        self.scroll_text = LogConsole.LogConsole(self)
        text_view_box = gtk.HBox(False,5)
        text_view_box.pack_start(self.scroll_text.scrolled_window,True,True,10)
	text_view_box.set_size_request(0,160)
        self.vbox1.pack_end(text_view_box,True,False,5)

	# Functions to compare_____________________________________________
	self.f_function = Function()
	self.f_function.load_function("data/function_nice.csv")
	self.log("Loaded f function from file: " + "data/function_nice.csv")
	self.g_function = Function()
	self.g_function.load_function("data/function_nice_2.csv")
	self.log("Loaded g function from file: " + "data/function_nice.csv")
        #__________________________________________________________________

	# MAT-PLOT-LIB_____________________________________________________

        self.plotter = Plotter(self,self.f_function,self.g_function)
        plotter_hbox = gtk.HBox(False,5)
        self.plotter.plotting()

        self.plotpar = Plotpar(self,self.f_function,self.g_function)
        plotpar_hbox = gtk.HBox(False,5)
        self.plotpar.plotting()

	self.function_grid = ButtonsGrid(self, self.f_function, self.g_function, self.plotter, self.plotpar)

	notebook = gtk.Notebook()
	notebook.append_page(self.plotter.vbox, gtk.Label("Functions"))
	notebook.append_page(self.plotpar.vbox, gtk.Label("Parameters"))
        plotter_hbox.pack_start(notebook,True,True,10)
	plotter_hbox.pack_start(self.function_grid.get_grid(), False, True, 5)

        self.vbox1.pack_start(plotter_hbox,True,True,10)

	self.asses_function_similarity(self.f_function, self.g_function)
        #__________________________________________________________________

    def log(self,txt):
	"""Logging into main log console"""
	self.scroll_text.log(txt)

    def load_function_chooser(self,button,function) :
        """Load function from file."""
        file_chooser = gtk.FileChooserDialog("Open...", self, gtk.FILE_CHOOSER_ACTION_OPEN, (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_OPEN, gtk.RESPONSE_OK))
        response = file_chooser.run()
        path=''
        if response == gtk.RESPONSE_OK :
	    path = file_chooser.get_filename()
	    function.load_function(path)
	    self.log('Loaded curve from file:   ' + path)
            file_chooser.destroy()
        else : 
            file_chooser.destroy()	    

    def asses_function_similarity(self, f,g) :

	if f.mode=="quadratic":
	    fc = f.params[2] 
	    fa = f.params[1] 
	    fb = f.params[0] 
	    sfc =f.std_err[2] 
	    sfa =f.std_err[1] 
	    sfb =f.std_err[0] 
	else:
	    fc = f.params[1] 
	    fa = f.params[0] 
	    fb = 0.0 
	    sfc =f.std_err[1] 
	    sfa =f.std_err[0] 
	    sfb =0.0

	if g.mode=="quadratic":
	    gc = g.params[2] 
	    ga = g.params[1] 
	    gb = g.params[0] 
	    sgc =g.std_err[2] 
	    sga =g.std_err[1] 
	    sgb =g.std_err[0] 
	else:
	    gc = g.params[1] 
	    ga = g.params[0] 
	    gb = 0.0 
	    sgc =g.std_err[1] 
	    sga =g.std_err[0] 
	    sgb =0.0

	c_abs_diff = abs(fc - gc)
	c_agrees = False
	if abs(fc - gc) <= sfc or abs(fc - gc) <= sgc :
	    c_agrees = True
	a_abs_diff = abs(fa - ga)
	a_agrees = False
	if abs(fa - ga) <= sfa or abs(fa - ga) <= sga :
	    a_agrees = True
	b_abs_diff = abs(fb - gb)
	b_agrees = False
	if abs(fb - gb) <= sfb or abs(fb - gb) <= sgb :
	    b_agrees = True
 
	if c_agrees and a_agrees and b_agrees :
	    self.log("Functions f and g don't differ signifficantly in terms of fitted parameters.")
	else :
	    self.log("Functions f and g differ signifficantly.")



#______________MAIN______________#

manager = UserInterface()
manager.show_all()
gtk.main()
