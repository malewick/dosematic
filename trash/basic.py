#!/usr/bin/env python

#from pyexcel_xls import get_data
import csv

import pygtk
pygtk.require('2.0')
import gtk
from gtk import gdk

#import gi
#gi.require_version('Gtk', '3.0')
#from gi.repository import Gtk, GObject

import matplotlib
matplotlib.use('GTKAgg')  # or 'GTK'
import matplotlib.pyplot as plt
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
from matplotlib.backends.backend_gtkagg import NavigationToolbar2GTKAgg as NavigationToolbar

from matplotlib.figure import Figure

from numpy  import array
from scipy import stats
import numpy as np
from scipy.optimize import curve_fit

import uncer

def isfloat(s):
    try:
        float(s)
        return True
    except:
        return False

# --- CLASSES ---
class DataManager(gtk.Window):

        # global variables needed to share among classes
	global labels

        ###########################################################################################
	def __init__(self):
                # init gtk::Window
		gtk.Window.__init__(self)
		self.set_default_size(600, 800)
		self.connect('destroy', lambda win: gtk.main_quit())

		self.set_title('DOSEMATIC v0.1')

                # variable name -- TODO
		self.xvariable="Dose"
		self.xvar="D"
		self.xunits="Gy"
		self.yvariable="Yield"
		self.yvar="Y"
		self.yunits=""

                # main layout container
		main_eb = gtk.EventBox()

                # horizontal box
		hbox = gtk.HBox(False, 8)
                # vertical box
		VBOX = gtk.VBox(False, 0)

		main_eb.add(VBOX)
		#main_eb.modify_bg(gtk.STATE_NORMAL, gtk.gdk.Color(red=60000,green=60000,blue=60000))
		self.add(main_eb)
		vbox1 = gtk.VBox(False,8)
		hbox.pack_start(vbox1, True, True)

		top_band = gtk.HBox()
		bottom_band = gtk.HBox()
		top_eb = gtk.EventBox()
		bottom_eb = gtk.EventBox()
		top_eb.add(top_band)
		bottom_eb.add(bottom_band)
		top_eb.modify_bg(gtk.STATE_NORMAL, gtk.gdk.Color(0,0,0))
		bottom_eb.modify_bg(gtk.STATE_NORMAL, gtk.gdk.Color(0,0,0))
		l1 = gtk.Label('DOSEMATIC v1.0 --- beta testing --- module 1, basic view')
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
		self.text = gtk.TextView()				# TEXT VIEW
		self.text.set_wrap_mode(gtk.WRAP_WORD)		# wrap words
		self.scroll_text = gtk.ScrolledWindow()		# into scrollable env
		self.scroll_text.set_shadow_type(gtk.SHADOW_ETCHED_IN)
		self.scroll_text.set_policy(gtk.POLICY_AUTOMATIC,gtk.POLICY_AUTOMATIC)
		self.scroll_text.add(self.text)
		text_view_box = gtk.VBox(False,5)
		text_view_box.pack_start(self.scroll_text,True,True)
		#__________________________________________________________________

		# ESTIMATOR________________________________________________________
		estimator_box = gtk.HBox(False,5)
		self.estxt = gtk.TextView()
		self.estxt.set_wrap_mode(gtk.WRAP_WORD)
		self.scroll_estxt = gtk.ScrolledWindow()
		self.scroll_estxt.set_shadow_type(gtk.SHADOW_ETCHED_IN)
		self.scroll_estxt.set_policy(gtk.POLICY_AUTOMATIC,gtk.POLICY_AUTOMATIC)
		self.scroll_estxt.add(self.estxt)
		label = gtk.Label(self.yvariable+' = ')
		entry = gtk.Entry()
		entry.set_text("0.00")
		button = gtk.Button('Estimate '+self.xvariable)
		button.connect('clicked',self.y_estimate,entry)
		combo = gtk.combo_box_new_text()
		combo.append_text("Method A")
		combo.append_text("Method B")
		combo.append_text("Method C-original")
		combo.append_text("Method C-simplified")
		self.method="Method C-simplified"
		combo.set_active(3)
		combo.connect('changed', self.on_method_changed)
		ruler = gtk.HSeparator()
		grid = gtk.Table(2,4)
		grid.attach(label, 0,1,0,1)
		grid.attach(entry, 1,2,0,1)
		grid.attach(button, 0,2,1,2)
		grid.attach(ruler,0,2,2,3)
		grid.attach(combo,0,2,3,4)
		estimator_box.pack_start(grid,False,False)
		estimator_box.pack_start(self.scroll_estxt,True,True)
		#__________________________________________________________________

		# FUNCTION TAB_____________________________________________________
		function_box = gtk.HBox(False,5)
		self.ftxt = gtk.TextView()
		self.ftxt.set_wrap_mode(gtk.WRAP_WORD)
		self.scroll_ftxt = gtk.ScrolledWindow()
		self.scroll_ftxt.set_shadow_type(gtk.SHADOW_ETCHED_IN)
		self.scroll_ftxt.set_policy(gtk.POLICY_AUTOMATIC,gtk.POLICY_AUTOMATIC)
		self.scroll_ftxt.add(self.ftxt)
		label_Y = gtk.Label()
		label_Y.set_use_markup(True)
		label_Y.set_markup('Y = c + &#945;D + &#946;D<sup>2</sup>') 
		self.entry_c = gtk.Entry()
		self.entry_c.set_width_chars(5)
		label_c = gtk.Label('c: ')
		self.entry_alpha = gtk.Entry()
		self.entry_alpha.set_width_chars(5)
		label_alpha = gtk.Label()
		label_alpha.set_use_markup(True)
		label_alpha.set_markup('&#945;: ') 
		self.entry_beta = gtk.Entry()
		self.entry_beta.set_width_chars(5)
		label_beta = gtk.Label()
		label_beta.set_use_markup(True)
		label_beta.set_markup('&#946;: ') 
		self.entry_sc = gtk.Entry()
		self.entry_sc.set_width_chars(5)
		label_sc = gtk.Label()
		label_sc.set_use_markup(True)
		label_sc.set_markup('&#963;(c): ') 
		self.entry_salpha = gtk.Entry()
		self.entry_salpha.set_width_chars(5)
		label_salpha = gtk.Label()
		label_salpha.set_use_markup(True)
		label_salpha.set_markup('&#963;(&#945;): ') 
		self.entry_sbeta = gtk.Entry()
		self.entry_sbeta.set_width_chars(5)
		label_sbeta = gtk.Label()
		label_sbeta.set_use_markup(True)
		label_sbeta.set_markup('&#963;(&#946;): ') 
		table_f = gtk.Table(6,3)
		#table_f.attach(label_Y, False, False)
		table_f.attach(label_c,0,1,0,1)
		table_f.attach(self.entry_c,1,2,0,1)
		table_f.attach(label_alpha,0,1,1,2)
		table_f.attach(self.entry_alpha,1,2,1,2)
		table_f.attach(label_beta,0,1,2,3)
		table_f.attach(self.entry_beta,1,2,2,3)
		table_f.attach(label_sc,4,5,0,1)
		table_f.attach(self.entry_sc,5,6,0,1)
		table_f.attach(label_salpha,4,5,1,2)
		table_f.attach(self.entry_salpha,5,6,1,2)
		table_f.attach(label_sbeta,4,5,2,3)
		table_f.attach(self.entry_sbeta,5,6,2,3)
		vruler = gtk.VSeparator()
		table_f.attach(vruler,3,4,0,3,xpadding=10)
		check_function = gtk.CheckButton("Plot function")
		check_points = gtk.CheckButton("Plot data points")
		check_err = gtk.CheckButton("Plot uncertainty band")
		check_ci_curve = gtk.CheckButton("Plot CI95% band (curve)")
		check_ci_points = gtk.CheckButton("Plot CI95% band (points)")
		check_function.set_active(True)
		check_points.set_active(True)
		check_err.set_active(True)
		check_ci_curve.set_active(True)
		check_ci_points.set_active(True)
		vbox_checks = gtk.VBox(False, 5)
		vbox_checks.pack_start(check_function, False, False)
		vbox_checks.pack_start(check_points, False, False)
		vbox_checks.pack_start(check_err, False, False)
		vbox_checks.pack_start(check_ci_curve, False, False)
		vbox_checks.pack_start(check_ci_points, False, False)
		check_function.connect('toggled',self.on_toggled, 'function')
		check_points.connect('toggled',self.on_toggled, 'points')
		check_err.connect('toggled',self.on_toggled, 'err')
		check_ci_curve.connect('toggled',self.on_toggled, 'ci_curve')
		check_ci_points.connect('toggled',self.on_toggled, 'ci_points')
		hbox_buttons = gtk.HBox(True,5)
		button_save_f = gtk.Button("Save Funtion")
		button_load_f = gtk.Button("Load Funtion")
		hbox_buttons.pack_start(button_save_f,True,True)
		hbox_buttons.pack_start(button_load_f,True,True)
		button_save_f.connect('clicked',self.save_function)
		button_load_f.connect('clicked',self.load_function)
		left_box = gtk.VBox(False,5)
		ruler_f1 = gtk.HSeparator()
		ruler_f2 = gtk.HSeparator()
		left_box.pack_start(label_Y, False, False)
		left_box.pack_start(table_f, False, False)
		left_box.pack_start(ruler_f1, False, True, 5)
		left_box.pack_start(vbox_checks, False, False)
		left_box.pack_start(ruler_f2, False, True, 5)
		left_box.pack_start(hbox_buttons, False, True)
		function_box.pack_start(left_box, False, False)
		function_box.pack_start(self.scroll_ftxt, True, True)
		#__________________________________________________________________

		# NOTEBOOK WRAP____________________________________________________
		self.notebook = gtk.Notebook()
		self.notebook.append_page(text_view_box, gtk.Label('Log console'))
		self.notebook.append_page(estimator_box, gtk.Label('Estimator'))
		self.notebook.append_page(function_box, gtk.Label('Calibration function'))
		vbox1.pack_end(self.notebook,True,True)
		#__________________________________________________________________

		# MAT-PLOT-LIB_____________________________________________________
		self.fig = Figure(figsize=(6, 4))		# create fig
		self.canvas = FigureCanvas(self.fig)		# a gtk.DrawingArea
		self.canvas.set_size_request(600,400)		# set min size
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
		self.styledict["title_size"]='25'
		self.styledict["xtitle_size"]='20'
		self.styledict["xlabel_size"]='15'
		self.styledict["ytitle_size"]='20'
		self.styledict["ylabel_size"]='15'

		self.nselec = [1,12,5,3,-1,0,-1,0,-1,-1,-1,-1,-1,-1]
		self.plot_labels = ["Foci per cell vs Dose", "Dose", "Foci per cell", " [Gy]", " []"]
		#print plt.style.available
		self.mode='quadratic'
		self.function = None
		if self.mode=='linear' :
			self.function = self.linear
		elif self.mode=='quadratic' :
			self.function = self.quadratic
		self.fit_toggle='active'
		self.points_toggle=1
		self.function_toggle=1
		self.err_toggle=1
		self.ci_func_toggle=1
		self.ci_points_toggle=1
		self.plotting()					# --- CORE plotting function ---
		toolbar = NavigationToolbar(self.canvas, self)
		toolbarbox = gtk.HBox()
		image = gtk.Image()
		image.set_from_stock(gtk.STOCK_PROPERTIES, gtk.ICON_SIZE_LARGE_TOOLBAR)
		options_button = gtk.Button()
		options_button.add(image)
		options_button.connect('clicked',self.mpl_options)
		image2 = gtk.Image()
		image2.set_from_stock(gtk.STOCK_REFRESH, gtk.ICON_SIZE_LARGE_TOOLBAR)
		refresh_button = gtk.Button()
		refresh_button.add(image2)
		refresh_button.connect('clicked',self.on_refresh_clicked)
		toolbarbox.pack_start(toolbar, True, True)
		toolbarbox.pack_end(options_button, False, True)
		toolbarbox.pack_end(refresh_button, False, True)
		vbox1.pack_start(toolbarbox, False, False)
		vbox1.pack_start(self.canvas, True, True)	# into box layout
		#__________________________________________________________________

		
	def plotting(self):
		plt.style.use(self.pstyle[0])

		self.ax1 = self.fig.add_subplot(111)
		self.ax1.clear()

		self.ax1.set_title(self.plot_labels[0], fontsize=self.pstyle[10])
		self.ax1.set_xlabel(self.plot_labels[1]+self.plot_labels[3], fontsize=int(self.pstyle[13]))
		self.ax1.set_ylabel(self.plot_labels[2]+self.plot_labels[4], fontsize=int(self.pstyle[17]))
		self.ax1.tick_params(axis='x', which='both', labelsize=int(self.pstyle[14]))
		self.ax1.tick_params(axis='y', which='both', labelsize=int(self.pstyle[18]))

		x = np.arange(-0.1, max(20,200)*1.1, 0.05)

                xdata=array([100,110,120,130,140,150,160,170,180,190,1000]);
                ydata=array([100,110,120,130,140,130,160,170,180,190,1000]);

		if (self.fit_toggle=='active'):
			self.params, self.rmse, self.p_value, self.std_err, self.dof, self.rss, self.cov_mtx = self.fit_function(xdata,ydata)
			self.function_changed()

		if self.function_toggle==1:
			y = self.function(x,self.params)
			self.ax1.plot(x,y, color=self.pstyle[5], marker='.', linestyle='None', markersize=float(self.pstyle[6]))

		if self.ci_func_toggle==1 and self.fit_toggle=='active':
			conf = self.confidence(x,xdata,len(x),np.mean(xdata),self.dof,self.rmse)
			upper =  self.function(x,self.params) + conf
			lower =  self.function(x,self.params) - conf
			self.ax1.fill_between(x, lower, upper, facecolor=self.pstyle[7], alpha=float(self.pstyle[8]))

		if self.ci_points_toggle==1:
			upper =  self.function(x,self.params) + self.confidence_points(x,self.std_err)
			lower =  self.function(x,self.params) - self.confidence_points(x,self.std_err)
			self.ax1.fill_between(x, lower, upper, facecolor='blue', alpha=float(self.pstyle[8]))

		if self.err_toggle==1:
			upper =  self.function(x,self.params) + self.uncertainty(x,self.std_err)
			lower =  self.function(x,self.params) - self.uncertainty(x,self.std_err)
			self.ax1.fill_between(x, lower, upper, facecolor='green', alpha=float(self.pstyle[8]))

		self.canvas.draw()

	def on_refresh_clicked(self,button) :
		self.plotting()

	def log(self,txt):
		end_iter = self.text.get_buffer().get_end_iter()
		self.text.get_buffer().insert(end_iter, txt+'\n')
		adj = self.scroll_text.get_vadjustment()
		adj.set_value( adj.upper - adj.page_size )
		self.notebook.set_current_page(0)

	def loges(self,txt):
		end_iter = self.estxt.get_buffer().get_end_iter()
		self.estxt.get_buffer().insert(end_iter, txt+'\n')
		adj = self.scroll_estxt.get_vadjustment()
		adj.set_value( adj.upper - adj.page_size )
		self.notebook.set_current_page(1)

	def logf(self,txt):
		end_iter = self.ftxt.get_buffer().get_end_iter()
		self.ftxt.get_buffer().insert(end_iter, txt+'\n')
		adj = self.scroll_ftxt.get_vadjustment()
		adj.set_value( adj.upper - adj.page_size )
		self.notebook.set_current_page(2)

	def linear(self, x, params):
		return params[0]*x + params[1]

	def quadratic(self, x, params):
		return params[0]*x*x + params[1]*x + params[2]

	def fit_linear(self, x, a, b):
		return a*x + b

	def fit_quadratic(self, x, a, b, c):
		return a*x*x + b*x + c

	def confidence(self, x, xdata, n, mean_x, dof, RMSE):
		alpha=0.05
		t = stats.t.isf(alpha/2., df=dof)
		#conf = t * np.sqrt((RSS/(n-2))*(1.0/n + ( (x-mean_x)**2 / ((np.sum(x**2)) - n*(mean_x**2)))))
		Sxx = np.sum(xdata**2) - np.sum(xdata)**2/n
		se_a = RMSE / np.sqrt(Sxx)
		se_b = RMSE * np.sqrt(np.sum(xdata**2)/(n*Sxx))
		
		conf = t * RMSE * np.sqrt(  1./n + (x-mean_x)**2/Sxx)
		#pred = t * RMSE * np.sqrt(1+1./n + (x-mean_x)**2/Sxx)
		return conf

	def uncertainty(self, x, std_err) :
		return std_err[2] + x*std_err[1] + x*x*std_err[0]

	def confidence_points(self, x, std_err) :
		return 1.96*self.uncertainty(x, std_err)

	def fit_function(self,x,y):
		# fit the model
		if self.mode=='linear' :
			popt, pcov = curve_fit(self.fit_linear, x, y)
		elif self.mode=='quadratic' :
			popt, pcov = curve_fit(self.fit_quadratic, x, y)
		# parameters standard error
		std_err = np.sqrt(np.diag(pcov))
		# degrees of freedom
		ndata = len(y)
		npar = len(popt)
		dof = max(0, ndata - npar)
		# root mean squared error
		residuals = y - self.function(x,popt)
		RSS = sum(residuals**2)
		MSE = RSS/dof
		RMSE = np.sqrt(MSE)
		# t-value
		t_value = popt/std_err
		# p-value P(>|t|)
		p_value=(1 - stats.t.cdf( abs(t_value), dof))*2

		return popt, RMSE, p_value, std_err, dof, RSS, pcov

	def function_changed(self):
		if self.mode=='quadratic' :
			self.entry_c.set_text('%.3f' % self.params[2])
			self.entry_alpha.set_text('%.3f' % self.params[1])
			self.entry_beta.set_text('%.3f' % self.params[0])
			self.entry_sc.set_text('%.3f' % self.std_err[2])
			self.entry_salpha.set_text('%.3f' % self.std_err[1])
			self.entry_sbeta.set_text('%.3f' % self.std_err[0])

			self.logf("params:\t[beta\talpha\tc ]")
			self.logf("values\t\t" + str(self.params))
			self.logf("std_err\t" + str(self.std_err))
			self.logf("p-value\t" + str(self.p_value))
			self.logf("RSS\t" + str(self.rss))
			self.logf("RMSE\t" + str(self.rmse))
			self.logf("---------------------------------------------------------------------------")

	def y_estimate(self, button, entry):
		if not isfloat(entry.get_text()):
			self.loges("___Not a number!___")
			return
		Y = float(entry.get_text())
		plist = self.get_fit_params()
		u = uncer.UCER(Y=Y,par_list=plist)
		D = u.D
		if self.method=="Method A":
			DL, DU = u.method_a()
		elif self.method=="Method B":
			DL, DU = u.method_b()
		elif self.method=="Method C-original":
			DL, DU = u.method_c1()
		elif self.method=="Method C-simplified":
			DL, DU = u.method_c2()

		xlab=self.xvar
		ylab=self.yvar
		self.loges( xlab + " estimation for   " + ylab + " = " + str(Y) + " using " + self.method + ":")
		self.loges( "D = " + str(D) + ";   DL = " + str(DL) + ";   DU = " + str(DU))
		self.loges("-----------------------------------------------------------------")

		self.ax1.axhline(y=Y,linewidth=1,linestyle='-',color='red')
		self.ax1.axvline(x=D,linewidth=1,linestyle='-',color='blue')
		self.ax1.axvline(x=DL,linewidth=1,linestyle='--',color='green')
		self.ax1.axvline(x=DU,linewidth=1,linestyle='--',color='green')
		self.canvas.draw()

	def mpl_options(self,button) :
		dialog = gtk.Dialog("My Dialog",self,0,(gtk.STOCK_OK, gtk.RESPONSE_OK))
		box = dialog.get_content_area()
		table = gtk.Table(2,18)
		table.set_row_spacings(5)
		table.set_col_spacings(5)
		l=[]
		l.append(gtk.Label("Canvas Style"))
		l.append(gtk.Label("Marker Style"))
		l.append(gtk.Label("Marker Size"))
		l.append(gtk.Label("Marker Color"))
		l.append(gtk.Label("Marker Alpha"))
		l.append(gtk.Label("Line Color"))
		l.append(gtk.Label("Line Width"))
		l.append(gtk.Label("CI Band Color"))
		l.append(gtk.Label("CI Band Alpha"))
		l.append(gtk.Label("Title"))
		l.append(gtk.Label("Title size"))
		l.append(gtk.Label("X-axis title"))
		l.append(gtk.Label("X-axis unit"))
		l.append(gtk.Label("X-axis title size"))
		l.append(gtk.Label("X-axis labels size"))
		l.append(gtk.Label("Y-axis title"))
		l.append(gtk.Label("Y-axis unit"))
		l.append(gtk.Label("Y-axis title size"))
		l.append(gtk.Label("Y-axis labels size"))
		hbox=[]
		hlines=[]
		for i in range(0,len(l)) :
			l[i].set_alignment(xalign=0,yalign=0.5) 
			hbox.append(gtk.HBox(False,5))
			hlines.append(gtk.HSeparator())
			table.attach(l[i],0,1,2*i,2*i+1)
			table.attach(hbox[i],1,2,2*i,2*i+1)
			table.attach(hlines[i],0,2,2*i+1,2*i+2)
		
		combo_cs = self.create_combobox(plt.style.available,hbox,0)
		combo_mst = self.create_combobox(self.markers,hbox,1)
		spin_msz = self.create_spinbutton(hbox,float(self.pstyle[2]), 1.0,20.0,1.0,2, 2)
		combo_mc = self.create_combobox(self.colors,hbox,3)
		spin_ma = self.create_spinbutton(hbox,float(self.pstyle[4]), 0.0,1.0,0.05,2, 4)
		combo_lc = self.create_combobox(self.colors,hbox,5)
		spin_lw = self.create_spinbutton(hbox,float(self.pstyle[6]), 0.0,10.0,0.5,2, 6)
		combo_bc = self.create_combobox(self.colors,hbox,7)
		spin_ba = self.create_spinbutton(hbox,float(self.pstyle[8]), 0.0,1.0,0.05,2, 8)

		entry_title = self.create_entry(hbox,0, 9)
		entry_xaxis = self.create_entry(hbox,1, 11)
		entry_xunit = self.create_entry(hbox,3, 12)
		entry_yaxis = self.create_entry(hbox,2, 15)
		entry_yunit = self.create_entry(hbox,4, 16)

		spin_title_size = self.create_spinbutton(hbox,float(self.pstyle[10]), 10.0,40.0,1.0,1 , 10)
		spin_xtile_size = self.create_spinbutton(hbox,float(self.pstyle[13]), 10.0,40.0,1.0,1 , 13)
		spin_xlabels_size = self.create_spinbutton(hbox,float(self.pstyle[14]), 10.0,40.0,1.0,1 , 14)
		spin_ytile_size = self.create_spinbutton(hbox,float(self.pstyle[17]), 10.0,40.0,1.0,1 , 17)
		spin_ylabels_size = self.create_spinbutton(hbox,float(self.pstyle[18]), 10.0,40.0,1.0,1 , 18)

		box.add(table)
		dialog.show_all()
		response = dialog.run()
		if response == gtk.RESPONSE_OK :
			dialog.destroy()
		else :
			dialog.destroy()

	def create_combobox(self,slist,whereto,n) :
		combo = gtk.combo_box_new_text()
		whereto[n].pack_start(combo)
		for style in slist :
			combo.append_text(str(style))
		combo.set_active(self.nselec[n])
		combo.connect('changed', self.on_combo_changed, n)

	def create_spinbutton(self,whereto,val,mini,maxi,step,digits,n) :
		adj = gtk.Adjustment(val,mini,maxi,step,0.5,0.0)
		spin = gtk.SpinButton(adj,step,digits)
		whereto[n].pack_start(spin)
		spin.connect('changed',self.on_spin_changed,n)

	def create_entry(self,whereto,m,n) :
		entry_title = gtk.Entry()
		entry_title.set_text(self.plot_labels[m])
		whereto[n].pack_start(entry_title)
		entry_title.connect("activate",self.on_entry_changed,m)

	def on_combo_changed(self,cb,n):
		model = cb.get_model()
		index = cb.get_active()
		cb.set_active(index)
		self.pstyle[n] = model[index][0]
		self.nselec[n]=index
		self.plotting()

	def on_spin_changed(self,spin,n) :
		self.pstyle[n] = spin.get_value()
		self.plotting()

	def on_entry_changed(self,entry,n) :
		self.plot_labels[n] = entry.get_text()
		self.plotting()

	def on_toggled(self,button,s) :
		if(s=='ci_points'): self.ci_points_toggle*=-1
		elif(s=='ci_curve'): self.ci_func_toggle*=-1
		elif(s=='function'): self.function_toggle*=-1
		elif(s=='points'): self.points_toggle*=-1
		elif(s=='err'): self.err_toggle*=-1
		self.plotting()

	def save_function(self,button) : 

		file_chooser = gtk.FileChooserDialog("Open...", self, gtk.FILE_CHOOSER_ACTION_SAVE, (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_SAVE, gtk.RESPONSE_OK))
		response = file_chooser.run()
		path=''
		if response == gtk.RESPONSE_OK :
			path = file_chooser.get_filename()
			self.logf('Curve saved in file:   ' + path)
			self.logf("---------------------------------------------------------------------------")
			if ".csv" not in path:
				path = path + '.csv'
			file_chooser.destroy()

			ofile = open(path,"wb")
			writer = csv.writer(ofile, delimiter=',')
			writer.writerow(self.params)
			writer.writerow(self.std_err)
			writer.writerow(self.p_value)
			writer.writerow(self.cov_mtx[0])
			writer.writerow(self.cov_mtx[1])
			writer.writerow(self.cov_mtx[2])
			writer.writerow((self.rss, self.rmse, 0.0))
			ofile.close()
		else :
			file_chooser.destroy()

	def get_fit_params(self):
		l=[self.params,self.std_err,self.p_value,self.cov_mtx[0],self.cov_mtx[1],self.cov_mtx[2],[self.rss,self.rmse,0.0]]
		return l

	def load_function(self,button) : 
		file_chooser = gtk.FileChooserDialog("Open...", self, gtk.FILE_CHOOSER_ACTION_OPEN, (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_OPEN, gtk.RESPONSE_OK))
		response = file_chooser.run()
		path=''
		if response == gtk.RESPONSE_OK :
			path = file_chooser.get_filename()
			self.logf('Loaded curve from file:   ' + path)
			self.logf("---------------------------------------------------------------------------")
			f = open(path, 'rt')
			try:
				reader = csv.reader(f)
				l=list(reader)
				print l
				self.params=[float(i) for i in l[0]]
				self.std_err=[float(i) for i in l[1]]
				self.p_value=[float(i) for i in l[2]]
				self.cov_mtx=[[float(i) for i in l[3]],[float(i) for i in l[4]],[float(i) for i in l[5]]]
				self.rss=float(l[6][0])
				self.rmse=float(l[6][1])
				self.function_changed()
				self.fit_toggle='inactive'
				self.points_toggle=False
				self.plotting()
			finally:
				f.close()
			#self.plotting()
			file_chooser.destroy()
		else : 
			file_chooser.destroy()

	def on_method_changed(self,cb):
		model = cb.get_model()
		index = cb.get_active()
		cb.set_active(index)
		self.method = model[index][0]
		#self.plotting()

#________________________________MAIN_____________________________________________________________#

manager = DataManager()
manager.show_all()
gtk.main()
