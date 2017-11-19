
# coding: utf-8

# # Imports

# Core functionality is GTK with Matplotlib plug-in.
# Numpy and Scipy as helpers.

import csv
from datetime import datetime

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

import Data
import uncer
import LogConsole

# ## Install requirements:
# GTK v2.24.30<br>
# pyGTK v2.24.0<br>
# matplotlib v1.5.1<br>
# numpy v1.12.1<br>
# scipy v0.17.0<br>
# <br>
# Check:

print "GTK v" + '.'.join(str(i) for i in gtk.gtk_version)
print "matplotlib v" + matplotlib.__version__
print "numpy v" + np.__version__
print "scipy v" + scipy.__version__

# # UserInterface
# It is a class containing GUI build and connected signals.
# In principle could be divided into smaller parts, but ain't nobody got time for that.

def linear(x, params):
    """Linear Function"""
    return params[0]*x + params[1]

def quadratic(x, params):
    """Quadratic Function"""
    return params[0]*x*x + params[1]*x + params[2]

def fit_linear(x, a, b):
    """Linear Function"""
    return a*x + b

def fit_quadratic(x, a, b, c):
    """Quadratic Function"""
    return a*x*x + b*x + c

def confidence(x, xdata, n, mean_x, dof, RMSE):
    """Calculation of the confidence interval of the fitting function"""
    alpha=0.05
    t = stats.t.isf(alpha/2., df=dof)
    #conf = t * np.sqrt((RSS/(n-2))*(1.0/n + ( (x-mean_x)**2 / ((np.sum(x**2)) - n*(mean_x**2)))))
    Sxx = np.sum(xdata**2) - np.sum(xdata)**2/n
    se_a = RMSE / np.sqrt(Sxx) if Sxx != 0  else 0.0
    se_b = RMSE * np.sqrt(np.sum(xdata**2)/(n*Sxx)) if Sxx != 0  else 0.0

    conf = t * RMSE * np.sqrt(  1./n + (x-mean_x)**2/Sxx) if Sxx != 0  else 0.0
    #pred = t * RMSE * np.sqrt(1+1./n + (x-mean_x)**2/Sxx)
    return conf

def uncertainty(x, std_err) :
    """Calculation of uncertainty of points"""
    if len(std_err)==3:
	return std_err[2] + x*std_err[1] + x*x*std_err[0]
    elif len(std_err)==2:
	return std_err[1] + x*std_err[0]
    else:
	return 0

def confidence_points(x, std_err) :
    """Calculation of the confidence interval of data points"""
    return 1.96*uncertainty(x, std_err)

class FitFunction():
    def __init__(self,title,mode_input):
        
        self.title=title
        self.params=[]
        self.std_err=[]
        self.p_value=[]
        self.chi_sq=[]
        self.cov_mtx=[]
        self.rss=0
        self.rmse=0
	self.dof=0
        
        #self.mode='quadratic'
        self.mode=mode_input
        self.func = None
        if self.mode=='linear' :
            self.func = linear
            self.fit = fit_linear
        elif self.mode=='quadratic' :
            self.func = quadratic
            self.fit = fit_quadratic
            
    def get_fit_params(self):
        """Return the list of fit parameters"""
	return self.params, self.std_err, self.p_value, self.cov_mtx, self.rss, self.rmse, self.dof

    def fit_function(self,x,y,yerr):
	"""fitting the model"""
	print " --- x", x
	print " --- y", y
	popt, pcov = curve_fit(self.fit, x, y, sigma=yerr)
	
	std_err = np.sqrt(np.diag(pcov)) # parameters standard error
	ndata = len(y) # degrees of freedom
	npar = len(popt)
	dof = max(0, ndata - npar)
	residuals = y - self.func(x,popt)
	RSS = sum(residuals**2)
	MSE = RSS/dof
	RMSE = np.sqrt(MSE)
	chi_sq = sum((y - self.func(x,popt))*(y - self.func(x,popt))/self.func(x,popt))
	chi_sq_wg = chi_sq/dof
	t_value = popt/std_err
	p_value = stats.chi2.cdf(chi_sq, dof)
	Z = scipy.stats.norm.ppf(1-p_value)

	self.params=popt
	self.rmse=RMSE
	self.p_value=p_value
	self.chi_sq = chi_sq
	self.std_err=std_err
	self.dof=dof
	self.rss=RSS
	self.cov_mtx=pcov

	return popt, RMSE, p_value, std_err, chi_sq, dof, RSS, pcov


class MyTreeView():

    def __init__(self, context, data):

        self.context=context # that's the main window
        self.data=data # and this is the Data instance

        # Create model and whole treeview
        self.model = self.create_model()		# MODEL
        self.treeview = gtk.TreeView(self.model)	# TREE VIEW
        self.treeview.set_rules_hint(True)
        self.add_columns()				# FILL COLUMNS

        # Add buttons
        button_add1 = gtk.Button('Add row')		# ADD 1 ROW
        button_clear = gtk.Button('Clear data')		# ADD 10 ROWS
        button_load = gtk.Button("Load data")		# LOAD FILE
        button_save = gtk.Button("Save data")		# LOAD FILE

        # Pack everything
        hbox_buttons = gtk.HBox(False,5)		# layout packaging
        hbox_buttons2 = gtk.HBox(False,5)		# layout packaging
        hbox_buttons.pack_start(button_add1, True, True)
        hbox_buttons.pack_start(button_clear, True, True)
        hbox_buttons2.pack_start(button_load, True, True)
        hbox_buttons2.pack_start(button_save, True, True)
        context.vbox2.pack_start(hbox_buttons2, False, False)
        context.vbox2.pack_end(hbox_buttons, False, False)

        button_add1.connect('clicked',self.add_rows,1)	# SIGNALS HANDLING
        button_clear.connect('clicked',self.clear_rows)
        button_load.connect('clicked',self.on_load_file)
        button_save.connect('clicked',self.on_save_file)

	self.scrolled_window = gtk.ScrolledWindow()
        self.scrolled_window.set_shadow_type(gtk.SHADOW_ETCHED_IN)
        self.scrolled_window.set_policy(gtk.POLICY_NEVER,gtk.POLICY_AUTOMATIC)
        self.scrolled_window.add(self.treeview)

        #__________________________________________________________________

    def add_columns(self):
        for i in range(self.data.numCols):
            renderer = gtk.CellRendererText()
            renderer.props.wrap_width = 100

            if i==0 or i==1 or i==2:
                renderer.set_property('editable', True)
                renderer.connect('edited',self.edited_cb, i)
            else :
                renderer.set_property('editable', False)
                renderer.set_property('background-gdk', gtk.gdk.Color(red=65535-11000, green=65535-7000, blue=65535-11000))
            column = gtk.TreeViewColumn(self.data.labels[i], renderer, text=i)
            column.set_resizable(True)
            column.set_sizing(gtk.TREE_VIEW_COLUMN_FIXED)
            column.set_min_width(50)
            column.set_fixed_width(80)
            column.set_expand(False)
            self.treeview.append_column(column)
    
	self.treeview.connect("key-release-event", self.on_navigate_key)

    def on_navigate_key(self, treeview, event):
        keyname = gdk.keyval_name(event.keyval)
        path, col = treeview.get_cursor()
        columns = [c for c in treeview.get_columns()] 
        colnum = columns.index(col)        
	nrows = len(treeview.get_model())

        if keyname == 'Tab':
	    newcol = treeview.get_column((colnum+1)%3)
	    newpath=path
	    treeview.set_cursor(newpath, newcol, True)

        elif keyname == 'Return':
	    newcol = treeview.get_column(colnum)
	    if path[0]+1 < nrows :
		newpath=(path[0]+1,)
		treeview.set_cursor(newpath, newcol, True)

        else:
            pass

    def add_rows(self,button,n):
        self.context.log('n of rows to add: ' + str(n))
        for i in range(0,n) :
            self.model.append([0,0,0,0,0])
            self.data.table = np.insert( self.data.table, len(self.data.get_xdata()), values=0, axis=1 )
        adj = self.scrolled_window.get_vadjustment()
        adj.set_value( adj.upper - adj.page_size )

    def clear_rows(self,button):
        self.context.log('data table cleared')
	#self.model = [[0,0,0,0,0]]
	self.model.clear()
	self.model.append([0,0,0,0,0])
	print self.data.table
	self.data.table = np.array([[0],[0],[0],[0],[0]], dtype='f')
	print
	print self.data.table
        adj = self.scrolled_window.get_vadjustment()
        adj.set_value( adj.upper - adj.page_size )

    def edited_cb(self, cell, path, new_content, user_data):
        column = user_data
        liststore=self.model
        print 'edited_cb', len(self.model), path
	print new_content
        new_content=new_content.replace(",",".")
        if isfloat(new_content) and float(new_content)>=0.0 :
	    if column > 0 and column < 3:
		liststore[path][column] = '%.0f' % float(new_content)
	    else :
		liststore[path][column] = '%.3f' % float(new_content)

            self.data.table[int(column)][int(path)] = float(new_content)
            if float(liststore[path][1]) != 0:
                liststore[path][3] = '%.3f' % (float(liststore[path][2]) / float(liststore[path][1]))
                liststore[path][4] = '%.3f' % (np.sqrt(float(liststore[path][2])) / float(liststore[path][1]))
                self.data.table[3][int(path)] = float(self.data.table[2][int(path)]) / float(self.data.table[1][int(path)])
                self.data.table[4][int(path)] = np.sqrt(self.data.table[2][int(path)]) / self.data.table[1][int(path)]
            else:
                liststore[path][3]='%.3f' % (0.0)
                self.data.table[3][int(path)]=0.0
                liststore[path][4]='%.3f' % (0.0)
                self.data.table[4][int(path)]=0.0
            print "data[", column, "][", path, "]  = ", self.data.table[int(column)][int(path)]
            self.context.plotter.plotting()
        else :
            self.context.log("___Wrong input format!___\n" + new_content + " is not a floating point number!")

    def create_model(self):
        types = [str]*self.data.numCols
        store = None
        store = gtk.ListStore(*types)

        temp=zip(*self.data.table)
        for row in temp:
	    srow = []
	    for i in range(0,len(row)) :
		if i>0 and i<3 :
		    srow.append('%.0f' % row[i])
		else :
		    srow.append('%.3f' % row[i])
            store.append(srow)
        return store

    def on_load_file(self, button) :

	fc = MyFileChooser()
	file_chooser = fc.get_filechooser()

	response = file_chooser.run()
	if response == gtk.RESPONSE_OK:
	    path = file_chooser.get_filename()
	    if ".csv" in path:
		read_data_flag = self.data.read_data_csv(path)
		if read_data_flag == 0:
		    self.context.log("error while loading file: "+path)
		    return
		elif read_data_flag <= 2:
		    self.context.log("error while reading columns in file: "+path+". File not loaded! Data table left unmodified.")
		    return
		self.context.log("Loaded data file: "+path)
		self.model = self.create_model()
		self.treeview.set_model(self.model)
		# TODO : fix this!
		self.context.plotter.fit_toggle = 'active'
		self.context.plotter.points_toggle=True
		self.context.plotter.plotting()
	    else : 
		self.context.log("___Wrong file format!___")
	    file_chooser.destroy()
	elif response == gtk.RESPONSE_CANCEL:
	    file_chooser.destroy()

    def on_save_file(self, button) :

	fc = MyFileChooser()
	file_chooser = fc.get_filechooser()

        response = file_chooser.run()
        path=''
        if response == gtk.RESPONSE_OK :
            path = file_chooser.get_filename()
            self.context.log('Data saved in file:   ' + path)
            if ".csv" not in path:
                path = path + '.csv'
            ofile = open(path,"wb")
            writer = csv.writer(ofile, delimiter=',')
            writer.writerow(self.data.labels)
            for row in zip(*self.data.table):
                print row
                writer.writerow(row)
            ofile.close()
            file_chooser.destroy()
        else :
            file_chooser.destroy()

class MyFileChooser() :

    def __init__(self):
	file_chooser = gtk.FileChooserDialog("Open..",
		None,
		gtk.FILE_CHOOSER_ACTION_OPEN,
		(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
		    gtk.STOCK_OPEN, gtk.RESPONSE_OK))
	file_chooser.set_default_response(gtk.RESPONSE_OK)

	filtr = gtk.FileFilter()
	filtr.set_name("All files")
	filtr.add_pattern("*")
	file_chooser.add_filter(filtr)

	filtr = gtk.FileFilter()
	filtr.set_name("Images")
	filtr.add_mime_type("image/png")
	filtr.add_mime_type("image/jpeg")
	filtr.add_mime_type("image/gif")
	filtr.add_pattern("*.png")
	filtr.add_pattern("*.jpg")
	filtr.add_pattern("*.gif")
	filtr.add_pattern("*.tif")
	filtr.add_pattern("*.xpm")
	file_chooser.add_filter(filtr)

	filtr = gtk.FileFilter()
	filtr.set_name("Data")
	filtr.add_mime_type("image/csv")
	filtr.add_pattern("*.csv")
	filtr.add_pattern("*.dat")
	file_chooser.add_filter(filtr)
	self.fc = file_chooser

    def get_filechooser(self):
	return self.fc

class Estimator():
    def __init__(self, context, data, fitfunction):

	self.context=context
	self.data=data
	self.fitfunction=fitfunction

        label = gtk.Label(data.yvariable+' = ')
        entry = gtk.Entry()
        entry.set_text("0.00")
        entry.connect("activate",self.y_estimate,entry)
        button = gtk.Button('Estimate '+data.xvariable)
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

	self.hbox = gtk.HBox()
        self.hbox.pack_start(grid,False,False)

    def on_method_changed(self,cb):
        """When estimation method is changed make appropriate changes"""
        model = cb.get_model()
        index = cb.get_active()
        cb.set_active(index)
        self.method = model[index][0]

    def y_estimate(self, button, entry):
	"""Calculation of reverse function and relevant confidence intervals"""
	if not isfloat(entry.get_text()):
	    self.context.log("___Not a number!___")
	    return
	Y = float(entry.get_text())
	plist = self.fitfunction.get_fit_params()
	u=None
	if self.fitfunction.mode=="linear":
	    u = uncer.UncerLin(Y=Y,par_list=plist)
	elif self.fitfunction.mode=="quadratic":
	    u = uncer.UncerQuad(Y=Y,par_list=plist)
	D = u.D
	if self.method=="Method A":
	    DL, DU = u.method_a()
	elif self.method=="Method B":
	    DL, DU = u.method_b()
	elif self.method=="Method C-original":
	    DL, DU = u.method_c1()
	elif self.method=="Method C-simplified":
	    DL, DU = u.method_c2()

	xlab=self.data.xvar
	ylab=self.data.yvar
	est=""
	est += ( "\n\t" + xlab + " estimation for   " + ylab + " = " + str(Y) + " using " + self.method + ":")
	est += ( "\n\t" + "D = " + str(D) + ";   DL = " + str(DL) + ";   DU = " + str(DU))

	self.context.log(est)

	self.context.plotter.plothline(y=Y, linewidth=1,linestyle='-',color='red')
	self.context.plotter.plotvline(x=D, linewidth=1,linestyle='-',color='blue')
	self.context.plotter.plotvline(x=DL,linewidth=1,linestyle='--',color='green')
	self.context.plotter.plotvline(x=DU,linewidth=1,linestyle='--',color='green')
	self.context.plotter.replot()

class FunctionTab():
    def __init__(self, context, fitfunction):
	self.context=context
	self.fitfunction=fitfunction

        #label_Y = gtk.Label()
        #label_Y.set_use_markup(True)
        #label_Y.set_markup('Y = c + &#945;D + &#946;D<sup>2</sup>') 
	label_Y = gtk.image_new_from_file('./equations/Yfunc1.png')
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
        hbox_buttons = gtk.HBox(True,5)
        button_save_f = gtk.Button("Save Funtion")
        button_load_f = gtk.Button("Load Funtion")
        hbox_buttons.pack_start(button_save_f,True,True)
        hbox_buttons.pack_start(button_load_f,True,True)
        left_box = gtk.VBox(False,5)
        ruler_f1 = gtk.HSeparator()
        ruler_f2 = gtk.HSeparator()
        left_box.pack_start(label_Y, False, False, 3)
        left_box.pack_start(table_f, False, False)
        left_box.pack_start(ruler_f1, False, True, 5)
        left_box.pack_start(vbox_checks, False, False)
        left_box.pack_start(ruler_f2, False, True, 5)
        left_box.pack_start(hbox_buttons, False, True)
	self.hbox = gtk.HBox()
        self.hbox.pack_start(left_box, False, False)

        # signals
        check_function.connect('toggled',self.on_toggled, 'function')
        check_points.connect('toggled',self.on_toggled, 'points')
        check_err.connect('toggled',self.on_toggled, 'err')
        check_ci_curve.connect('toggled',self.on_toggled, 'ci_curve')
        check_ci_points.connect('toggled',self.on_toggled, 'ci_points')
        button_save_f.connect('clicked',context.save_function)
        button_load_f.connect('clicked',context.load_function)

        self.entry_c.connect("activate",self.on_entry_changed)
        self.entry_alpha.connect("activate",self.on_entry_changed)
        self.entry_beta.connect("activate",self.on_entry_changed)
        self.entry_sc.connect("activate",self.on_entry_changed)
        self.entry_salpha.connect("activate",self.on_entry_changed)
        self.entry_sbeta.connect("activate",self.on_entry_changed)

    def on_entry_changed(self,entry) :
        """Changes function parameters when entry was changed."""
	c = float(self.entry_c.get_text()) if isfloat(self.entry_c.get_text()) else 0.0
	alpha = float(self.entry_alpha.get_text()) if isfloat(self.entry_alpha.get_text()) else 0.0
	beta = float(self.entry_beta.get_text()) if isfloat(self.entry_beta.get_text()) else 0.0
	sc = float(self.entry_sc.get_text()) if isfloat(self.entry_sc.get_text()) else 0.0
	salpha = float(self.entry_salpha.get_text()) if isfloat(self.entry_salpha.get_text()) else 0.0
	sbeta = float(self.entry_sbeta.get_text()) if isfloat(self.entry_sbeta.get_text()) else 0.0

	if self.fitfunction.mode=="quadratic":
	    self.fitfunction.params=[beta,alpha,c]
	    self.fitfunction.std_err=[sbeat,salpha,sc]
	elif self.fitfunction.mode=="linear":
	    self.fitfunction.params=[alpha,c]
	    self.fitfunction.std_err=[salpha,sc]

	self.context.plotter.fit_toggle='inactive'
        self.context.plotter.plotting()

    def on_toggled(self,button,s) :
        """Toggle plotting one of the given features"""
        if(s=='ci_points'): self.context.plotter.ci_points_toggle*=-1
        elif(s=='ci_curve'): self.context.plotter.ci_func_toggle*=-1
        elif(s=='function'): self.context.plotter.function_toggle*=-1
        elif(s=='points'): self.context.plotter.points_toggle*=-1
        elif(s=='err'): self.context.plotter.err_toggle*=-1
        self.context.plotter.plotting()

    def function_changed(self):
        """Choose relevant function: linear/quadratic. TODO: Add linear."""
        if self.fitfunction.mode=='quadratic' :
            self.entry_c.set_text('%.3f' % self.fitfunction.params[2])
            self.entry_alpha.set_text('%.3f' % self.fitfunction.params[1])
            self.entry_beta.set_text('%.3f' % self.fitfunction.params[0])
            self.entry_sc.set_text('%.3f' % self.fitfunction.std_err[2])
            self.entry_salpha.set_text('%.3f' % self.fitfunction.std_err[1])
            self.entry_sbeta.set_text('%.3f' % self.fitfunction.std_err[0])

        elif self.fitfunction.mode=='linear' :
            self.entry_c.set_text('%.3f' % self.fitfunction.params[1])
            self.entry_alpha.set_text('%.3f' % self.fitfunction.params[0])
            self.entry_beta.set_text('0')
            self.entry_sc.set_text('%.3f' % self.fitfunction.std_err[1])
            self.entry_salpha.set_text('%.3f' % self.fitfunction.std_err[0])
            self.entry_sbeta.set_text('0')

	    self.entry_beta. set_property("editable",False)
	    self.entry_sbeta.set_property("editable",False)
	    self.entry_beta.modify_base(gtk.STATE_NORMAL, gtk.gdk.color_parse("#E1E1E1"))
	    self.entry_sbeta.modify_base(gtk.STATE_NORMAL, gtk.gdk.color_parse("#E1E1E1"))


	p=self.fitfunction.p_value
	flag=""
	if p<2.9e-7 :
	    flag="perfect"
	elif p<0.00134990 :
	    flag="highly significant"
	elif p<0.0227501 :
	    flag="significant"
	elif p<0.158655 :
	    flag="marginally significant"
	else :
	    flag="not significant"


	self.context.log("Function was fitted to data:\n\tparams:\t[alpha\tc ]" + "\n" + \
	    "\tvalues\t\t" + str(self.fitfunction.params) + "\n" + \
	    "\tstd_err\t" + str(self.fitfunction.std_err) + "\n" + \
	    "\tchi-sq\t" + '%.3f' % (self.fitfunction.chi_sq) + ",\tdof:\t" + str(self.fitfunction.dof) + "\n" + \
	    "\tp-value\t" + '%.3f' % (self.fitfunction.p_value) + "\t -> \t" + flag + "\n" + \
	    "\tRSS\t" + '%.3f' % (self.fitfunction.rss) + "\n" + \
	    "\tRMSE\t" + '%.3f' % (self.fitfunction.rmse) + "\n")


class Plotter() :
    def __init__(self,context,data,fitfunction):
	self.context=context
	self.data=data
	self.fitfunction=fitfunction

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
	self.plot_labels = []
	self.plot_labels.append(data.labels[3]+" vs "+data.labels[0]) # plot title
	self.plot_labels.append(data.labels[0]) # x-axis title
	self.plot_labels.append(data.labels[3]) # y-axis title
	self.plot_labels.append("[Gy]") # x-axis unit
	self.plot_labels.append(" ") # y-axis unit
        #print plt.style.available

        self.fit_toggle='active'
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

        # signals
        self.canvas.mpl_connect('pick_event', self.on_pick)
        options_button.connect('clicked',self.mpl_options)
        refresh_button.connect('clicked',self.on_refresh_clicked)

    def on_pick(self,event):
        artist = event.artist
        xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
        x, y = artist.get_xdata(), artist.get_ydata()
        ind = event.ind
        print 'Artist picked:', event.artist
        print '{} vertices picked'.format(len(ind))
        print 'Pick between vertices {} and {}'.format(min(ind), max(ind))
        print 'x, y of mouse: {:.2f},{:.2f}'.format(xmouse, ymouse)
        print 'Data point:', x[ind[0]], y[ind[0]]
        print
        self.context.log('Data point:\t  ' + str(x[ind[0]]) + '\t' + str(y[ind[0]]))
        self.context.treeview.treeview.set_cursor(min(ind))
        self.context.treeview.treeview.grab_focus()
        
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
	
	self.ax1 = self.fig.add_subplot(111)
	self.ax1.clear()
	
	if self.points_toggle==1:
	    self.ax1.errorbar(self.data.get_xdata(),self.data.get_ydata(),self.data.get_yerr(), fmt='none', ecolor='black',elinewidth=0.5,capsize=0.5,capthick=0.5)
	    self.ax1.plot(self.data.get_xdata(),self.data.get_ydata(), color=self.pstyle[3], label=self.data.labels[3], marker=self.pstyle[1], alpha=float(self.pstyle[4]), linestyle='None', markersize=float(self.pstyle[2]), picker=float(self.pstyle[2]))

	self.ax1.set_title(self.plot_labels[0], fontsize=self.pstyle[10])
	self.ax1.set_xlabel(self.plot_labels[1]+self.plot_labels[3], fontsize=int(self.pstyle[13]))
	self.ax1.set_ylabel(self.plot_labels[2]+self.plot_labels[4], fontsize=int(self.pstyle[17]))
	self.ax1.tick_params(axis='x', which='both', labelsize=int(self.pstyle[14]))
	self.ax1.tick_params(axis='y', which='both', labelsize=int(self.pstyle[18]))

	x = np.arange(-0.1, max(20,max(self.data.get_xdata()))*1.1, 0.05)

	if (self.fit_toggle=='active'):
	    if len(self.data.get_xdata()) >= 3 :
		print "before fit", self.fitfunction.params
		print "before fit", self.fitfunction.params
		print "xdata", self.data.get_xdata()
		print "ydata", self.data.get_ydata()
		self.fitfunction.fit_function(self.data.get_xdata(),self.data.get_ydata(),self.data.get_yerr())
		print "after fit", self.fitfunction.params
		print "after fit", self.fitfunction.params
		self.context.functiontab.function_changed()
	    else :
		self.context.log("Too few data to fit the function!")

	if self.function_toggle==1:
	    y = self.fitfunction.func(x,self.fitfunction.params)
	    self.ax1.plot(x,y, color=self.pstyle[5], marker='.', linestyle='None', markersize=float(self.pstyle[6]))

	if self.ci_func_toggle==1 and self.fit_toggle=='active':
	    conf = confidence(x,self.data.get_xdata(),len(x),np.mean(self.data.get_xdata()),self.fitfunction.dof,self.fitfunction.rmse)
	    upper =  self.fitfunction.func(x,self.fitfunction.params) + conf
	    lower =  self.fitfunction.func(x,self.fitfunction.params) - conf
	    self.ax1.fill_between(x, lower, upper, facecolor=self.pstyle[7], alpha=float(self.pstyle[8]))

	if self.ci_points_toggle==1:
	    upper =  self.fitfunction.func(x,self.fitfunction.params) + confidence_points(x,self.fitfunction.std_err)
	    lower =  self.fitfunction.func(x,self.fitfunction.params) - confidence_points(x,self.fitfunction.std_err)
	    self.ax1.fill_between(x, lower, upper, facecolor='blue', alpha=float(self.pstyle[8]))

	if self.err_toggle==1:
	    upper =  self.fitfunction.func(x,self.fitfunction.params) + uncertainty(x,self.fitfunction.std_err)
	    lower =  self.fitfunction.func(x,self.fitfunction.params) - uncertainty(x,self.fitfunction.std_err)
	    self.ax1.fill_between(x, lower, upper, facecolor='green', alpha=float(self.pstyle[8]))

	self.fig.subplots_adjust(left=0.13, right=0.96, top=0.91, bottom=0.13, hspace=0.04)
	self.canvas.draw()

	print self.fitfunction.params
	print self.fitfunction.std_err


class MPLOptions() :
    def __init__(self,context,plotter):
        self.context=context
        self.plotter=plotter
        dialog = gtk.Dialog("My Dialog",context,0,(gtk.STOCK_OK, gtk.RESPONSE_OK))
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
        combo_mst = self.create_combobox(self.plotter.markers,hbox,1)
        spin_msz = self.create_spinbutton(hbox,float(self.plotter.pstyle[2]), 1.0,20.0,1.0,2, 2)
        combo_mc = self.create_combobox(self.plotter.colors,hbox,3)
        spin_ma = self.create_spinbutton(hbox,float(self.plotter.pstyle[4]), 0.0,1.0,0.05,2, 4)
        combo_lc = self.create_combobox(self.plotter.colors,hbox,5)
        spin_lw = self.create_spinbutton(hbox,float(self.plotter.pstyle[6]), 0.0,10.0,0.5,2, 6)
        combo_bc = self.create_combobox(self.plotter.colors,hbox,7)
        spin_ba = self.create_spinbutton(hbox,float(self.plotter.pstyle[8]), 0.0,1.0,0.05,2, 8)

        entry_title = self.create_entry(hbox,0, 9)
        entry_xaxis = self.create_entry(hbox,1, 11)
        entry_xunit = self.create_entry(hbox,3, 12)
        entry_yaxis = self.create_entry(hbox,2, 15)
        entry_yunit = self.create_entry(hbox,4, 16)

        spin_title_size = self.create_spinbutton(hbox,float(self.plotter.pstyle[10]), 10.0,40.0,1.0,1 , 10)
        spin_xtile_size = self.create_spinbutton(hbox,float(self.plotter.pstyle[13]), 10.0,40.0,1.0,1 , 13)
        spin_xlabels_size = self.create_spinbutton(hbox,float(self.plotter.pstyle[14]), 10.0,40.0,1.0,1 , 14)
        spin_ytile_size = self.create_spinbutton(hbox,float(self.plotter.pstyle[17]), 10.0,40.0,1.0,1 , 17)
        spin_ylabels_size = self.create_spinbutton(hbox,float(self.plotter.pstyle[18]), 10.0,40.0,1.0,1 , 18)

        box.add(table)
        dialog.show_all()
        response = dialog.run()
        if response == gtk.RESPONSE_OK :
            dialog.destroy()
        else :
            dialog.destroy()

    def create_combobox(self,slist,whereto,n) :
        """Create combobox from list of strings"""
        combo = gtk.combo_box_new_text()
        whereto[n].pack_start(combo)
        for style in slist :
            combo.append_text(str(style))
        combo.set_active(self.plotter.nselec[n])
        combo.connect('changed', self.on_combo_changed, n)

    def create_spinbutton(self,whereto,val,mini,maxi,step,digits,n) :
        """Create spinbox for given parameters"""
        adj = gtk.Adjustment(val,mini,maxi,step,0.5,0.0)
        spin = gtk.SpinButton(adj,step,digits)
        whereto[n].pack_start(spin)
        spin.connect('changed',self.on_spin_changed,n)

    def create_entry(self,whereto,m,n) :
        """Create text entry with predefined labels"""
        entry_title = gtk.Entry()
        entry_title.set_text(self.plotter.plot_labels[m])
        whereto[n].pack_start(entry_title)
        entry_title.connect("activate",self.on_entry_changed,m)

    def on_combo_changed(self,cb,n):
        """For a given style combo change plotting attributes according to what was set in options."""
        model = cb.get_model()
        index = cb.get_active()
        cb.set_active(index)
        self.plotter.pstyle[n] = model[index][0]
        self.plotter.nselec[n]=index
        self.context.plotter.plotting()

    def on_spin_changed(self,spin,n) :
        """For a given style spin change plotting attributes according to what was set in options."""
        self.plotter.pstyle[n] = spin.get_value()
        self.context.plotter.plotting()

    def on_entry_changed(self,entry,n) :
        """For a given plot label change plotting attributes according to what was set in options."""
        self.plotter.plot_labels[n] = entry.get_text()
        self.context.plotter.plotting()

    def on_toggled(self,button,s) :
        """Toggle plotting one of the given features"""
        if(s=='ci_points'): self.ci_points_toggle*=-1
        elif(s=='ci_curve'): self.ci_func_toggle*=-1
        elif(s=='function'): self.function_toggle*=-1
        elif(s=='points'): self.points_toggle*=-1
        elif(s=='err'): self.err_toggle*=-1
        self.context.plotter.plotting()
        

class UserInterface(gtk.Window):
    
    def __init__(self,module_name,mode_input,labels):
        """Init function with the whole GUI declaration and signals"""

        self.data = Data.Data(labels,'data/samples/'+module_name+'.csv')
	self.fitfunction = FitFunction("fitfunction",mode_input)

        gtk.Window.__init__(self)
        self.set_default_size(600, 800)
        #self.connect('destroy', lambda win: gtk.main_quit())
        
        self.set_title('DOSEMATIC v1.0 -- Acute Module -- ' + module_name)

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
        hbox.pack_start(self.vbox1, True, True,5)
        hbox.pack_start(self.vbox2, True, True,5)

        top_band = gtk.HBox()
        bottom_band = gtk.HBox()
        top_eb = gtk.EventBox()
        bottom_eb = gtk.EventBox()
        top_eb.add(top_band)
        bottom_eb.add(bottom_band)
        top_eb.modify_bg(gtk.STATE_NORMAL, gtk.gdk.Color(0,0,0))
        bottom_eb.modify_bg(gtk.STATE_NORMAL, gtk.gdk.Color(0,0,0))
        l1 = gtk.Label('DOSEMATIC v1.0 --- beta testing --- Acute --- '+module_name)
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

        # TREE VIEW________________________________________________________
        self.treeview = MyTreeView(self, self.data)
        self.vbox2.pack_start(self.treeview.scrolled_window, True, True)

        # TEXT SCREEN______________________________________________________
        self.scroll_text = LogConsole.LogConsole(self)
        text_view_box = gtk.VBox(False,5)
        text_view_box.pack_start(self.scroll_text.scrolled_window,True,True)
	text_view_box.set_size_request(0,160)

        # ESTIMATOR________________________________________________________
        estimator = Estimator(self,self.data,self.fitfunction)

        # FUNCTION TAB_____________________________________________________
        self.functiontab = FunctionTab(self,self.fitfunction)

	# WRAPPER__________________________________________________________
	hwrap = gtk.HBox()
	vsepr = gtk.VSeparator()
	hwrap.pack_start(self.functiontab.hbox,True,True,5)
	hwrap.pack_start(vsepr,False,False,5)
	hwrap.pack_end(estimator.hbox,True,True,5)
        self.vbox1.pack_end(hwrap,True,True)
        VBOX.pack_end(text_view_box,True,True)

        # MAT-PLOT-LIB_____________________________________________________
        self.plotter = Plotter(self,self.data,self.fitfunction)
        self.vbox1.pack_start(self.plotter.vbox,True,True)
        self.plotter.plotting()
        #__________________________________________________________________

    def log(self,txt):
        """Logging into main log console"""
	self.scroll_text.log(txt)
    
    def save_function(self,button) :
        """Saving function parameters, errors, covariance matrix, rss and rmse."""

	fc = MyFileChooser()
	file_chooser = fc.get_filechooser()

        response = file_chooser.run()
        path=''
        if response == gtk.RESPONSE_OK :
            path = file_chooser.get_filename()
            self.log("\n\tCurve saved in file:\t"+path)
            if ".csv" not in path:
                path = path + '.csv'
            file_chooser.destroy()

            ofile = open(path,"wb")
            writer = csv.writer(ofile, delimiter=',')
            writer.writerow(self.fitfunction.params)
            writer.writerow(self.fitfunction.std_err)
            writer.writerow(self.fitfunction.p_value)
            writer.writerow(self.fitfunction.cov_mtx[0])
            writer.writerow(self.fitfunction.cov_mtx[1])
            writer.writerow(self.fitfunction.cov_mtx[2])
            writer.writerow((self.fitfunction.rss, self.fitfunction.rmse, 0.0))
            ofile.close()
        else :
            file_chooser.destroy()

    def load_function(self,button) :
        """Load function from file."""

	fc = MyFileChooser()
	file_chooser = fc.get_filechooser()

        response = file_chooser.run()
        path=''
        if response == gtk.RESPONSE_OK :
            path = file_chooser.get_filename()
            self.log('\n\tLoaded curve from file:\t' + path)
            f = open(path, 'rt')
            try:
                reader = csv.reader(f)
                l=list(reader)
                print l
                self.fitfunction.params=[float(i) for i in l[0]]
                self.fitfunction.std_err=[float(i) for i in l[1]]
                self.fitfunction.p_value=[float(i) for i in l[2]]
                self.fitfunction.cov_mtx=[[float(i) for i in l[3]],[float(i) for i in l[4]],[float(i) for i in l[5]]]
                self.fitfunction.rss=float(l[6][0])
                self.fitfunction.rmse=float(l[6][1])
                self.functiontab.function_changed()
                self.plotter.fit_toggle='inactive'
                self.plotter.points_toggle=False
		self.plotter.plotting()
            finally:
                f.close()
            file_chooser.destroy()
        else : 
            file_chooser.destroy()
            
def isfloat(s):
    """Why is this not a thing?"""
    try:
        float(s)
        return True
    except:
        return False

#______________MAIN______________#

#manager = UserInterface("dic","linear",[ "Dose", "Cells counted", "Dicentrics counted", "Dicentrics per cell", "SE" ])
#manager.show_all()
#gtk.main()

