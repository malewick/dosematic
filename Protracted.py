# coding utf-8

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

import uncer
import LogConsole

# Install requirements:
print "GTK v" + '.'.join(str(i) for i in gtk.gtk_version)
print "matplotlib v" + matplotlib.__version__
print "numpy v" + np.__version__
print "scipy v" + scipy.__version__

def isfloat(s):
    """Why is this not a thing?"""
    try:
        float(s)
        return True
    except:
        return False

def quadratic(x, t, t0, params):
    """Quadratic Function"""
    return params[0]*x*x*G_Function(t,t0) + params[1]*x + params[2]

def uncertainty(x, std_err) :
    """Calculation of uncertainty of points"""
    return std_err[2] + x*std_err[1] + x*x*std_err[0]

def confidence_points(x, std_err) :
    """Calculation of the confidence interval of data points"""
    return 1.96*uncertainty(x, std_err)

def G_Function(t,t0):
    x=t/t0
    return 2./(x*x)*(x-1+np.exp(-x))


class FitFunction():
    def __init__(self,title):
        
        self.title=title
        self.params=[]
        self.std_err=[]
        self.p_value=[]
        self.cov_mtx=[]
        self.rss=0
        self.rmse=0
        self.dof=0
	self.t0=2
	self.t=20
        
	self.func = quadratic
            
    def get_fit_params(self):
        """Return the list of fit parameters"""
	params=self.params[:]
	params[0]*G_Function(self.t,self.t0)
        return params, self.std_err, self.p_value, self.cov_mtx, self.rss, self.rmse, self.dof


class Plot():
    def __init__(self, title, function, labels):
        
        self.title=title
        
        # MAT-PLOT-LIB_____________________________________________________
        self.fig = Figure(figsize=(6, 4))		# create fig
        self.canvas = FigureCanvas(self.fig)		# a gtk.DrawingArea
        self.canvas.set_size_request(500,300)		# set min size
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
        self.plot_labels.append(labels[3]+" vs "+labels[0]) # plot title
        self.plot_labels.append(labels[0]) # x-axis title
        self.plot_labels.append(labels[3]) # y-axis title
        self.plot_labels.append("[Gy]") # x-axis unit
        self.plot_labels.append(" ") # y-axis unit
        #print plt.style.available
        self.fit_toggle='inactive'
        self.function_toggle=1
        self.err_toggle=1
        self.ci_func_toggle=1
        self.ci_points_toggle=1
        #self.plotting(function)					# --- CORE plotting function ---

        
    def plotting(self, function):
        """Generating matplotlib canvas"""
        plt.style.use(self.pstyle[0])

        self.ax1 = self.fig.add_subplot(111)
        self.ax1.clear()

        self.ax1.set_title(self.plot_labels[0], fontsize=self.pstyle[10])
        self.ax1.set_xlabel(self.plot_labels[1]+self.plot_labels[3], fontsize=int(self.pstyle[13]))
        self.ax1.set_ylabel(self.plot_labels[2]+self.plot_labels[4], fontsize=int(self.pstyle[17]))
        self.ax1.tick_params(axis='x', which='both', labelsize=int(self.pstyle[14]))
        self.ax1.tick_params(axis='y', which='both', labelsize=int(self.pstyle[18]))

        x = np.arange(-0.1, max(20,200)*1.1, 0.05)

        if self.function_toggle==1:
            y = function.func(x,function.t,function.t0,function.params)
            self.ax1.plot(x,y, color=self.pstyle[5], marker='.', linestyle='None', markersize=float(self.pstyle[6]))

        if self.ci_points_toggle==1:
            upper =  function.func(x,function.t,function.t0,function.params)+confidence_points(x,function.std_err)
            lower =  function.func(x,function.t,function.t0,function.params)-confidence_points(x,function.std_err)
            self.ax1.fill_between(x, lower, upper, facecolor='blue', alpha=float(self.pstyle[8]))

        if self.err_toggle==1:
            upper =  function.func(x,function.t,function.t0,function.params) + uncertainty(x,function.std_err)
            lower =  function.func(x,function.t,function.t0,function.params) - uncertainty(x,function.std_err)
            self.ax1.fill_between(x, lower, upper, facecolor='green', alpha=float(self.pstyle[8]))

        self.fig.tight_layout()
        self.canvas.draw()

    def plotvline(self, **kwargs):
        self.ax1.axvline(**kwargs)

    def plothline(self, **kwargs):
        self.ax1.axhline(**kwargs)

    def replot(self):
        self.canvas.draw()

        
class Estimator():
    def __init__(self, context, function):

        self.context=context
        self.function=function

	self.yvariable="Yield"
	self.xvariable="Dose"

        label = gtk.Label(self.yvariable+' = ')
        entry = gtk.Entry()
        entry.set_text("0.00")
	entry.connect("activate",self.y_estimate,entry)
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

        self.hbox = gtk.HBox()
        self.hbox.pack_start(grid,False,False,10)

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
        plist = self.function.get_fit_params()
	plist[0][0] *= G_Function(self.function.t,self.function.t0)
        u=None
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

        xlab=self.xvariable
        ylab=self.yvariable
	self.context.log( xlab + " estimation for   " + ylab + " = " + str(Y) + " using " + self.method + ":\n D = " + str(D) + ";   DL = " + str(DL) + ";   DU = " + str(DU)+"\n -----------------------------------------------------------------")

        self.context.plot.plothline(y=Y, linewidth=1,linestyle='-',color='red')
        self.context.plot.plotvline(x=D, linewidth=1,linestyle='-',color='blue')
        self.context.plot.plotvline(x=DL,linewidth=1,linestyle='--',color='green')
        self.context.plot.plotvline(x=DU,linewidth=1,linestyle='--',color='green')
        self.context.plot.replot()

class G_Plot():
    def __init__(self, title, function):
        
        self.title=title
        
        # MAT-PLOT-LIB_____________________________________________________
        self.fig = Figure(figsize=(6, 4))		# create fig
        self.canvas = FigureCanvas(self.fig)		# a gtk.DrawingArea
        self.canvas.set_size_request(400,300)		# set min size
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

        self.function = function
        
        self.nselec = [1,12,5,3,-1,0,-1,0,-1,-1,-1,-1,-1,-1]
        self.plot_labels = ["Time Exposure", "t", "G(x)", " [h]", " []"]
        #print plt.style.available
        self.fit_toggle='inactive'
        self.function_toggle=1
        self.err_toggle=1
        self.ci_func_toggle=1
        self.ci_points_toggle=1
        #self.plotting(function)

        
    def plotting(self,t,t0):
        """Generating matplotlib canvas"""
        plt.style.use(self.pstyle[0])

        self.ax1 = self.fig.add_subplot(111)
        self.ax1.clear()

        self.ax1.set_title(self.plot_labels[0], fontsize=self.pstyle[10])
        self.ax1.set_xlabel(self.plot_labels[1]+self.plot_labels[3], fontsize=int(self.pstyle[13]))
        self.ax1.set_ylabel(self.plot_labels[2]+self.plot_labels[4], fontsize=int(self.pstyle[17]))
        self.ax1.tick_params(axis='x', which='both', labelsize=int(self.pstyle[14]))
        self.ax1.tick_params(axis='y', which='both', labelsize=int(self.pstyle[18]))

        x = np.arange(0.000001, 1.5*max(t,t0), 0.01)

        y = self.function(x,t0)
        self.ax1.plot(x,y, color=self.pstyle[5], marker='.', linestyle='None', markersize=float(self.pstyle[6]))

        self.ax1.axvline(x=t,linewidth=1,linestyle='-',color='red')
        self.ax1.axhline(y=self.function(t,t0),linewidth=1,linestyle='--',color='red')
        
        self.fig.tight_layout()
        self.canvas.draw()


# # UserInterface
# It is a class containing GUI build and connected signals.
# In principle could be divided into smaller parts, but ain't nobody got time for that.

# In[35]:

class UserInterface(gtk.Window):

    def __init__(self,module_name,labels):
        """Init function with the whole GUI declaration and signals"""
        
        gtk.Window.__init__(self)
        self.set_default_size(600, 800)
        #self.connect('destroy', lambda win: gtk.main_quit())

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
        hbox.pack_start(vbox1, True, True, 5)

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
        
        #_____PLOTTING Y-FUNCTION______________________________________________________
        self.function = FitFunction("YFunction")
        self.plot = Plot("YPlot",self.function,labels)
        self.load_function("data/function_nice.csv",self.function,self.plot)
        toolbar = NavigationToolbar(self.plot.canvas, self)
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
        Y_vbox = gtk.VBox()
        Y_vbox.pack_start(toolbarbox, False, False)
        Y_vbox.pack_start(self.plot.canvas, True, True)
        
        #_____PLOTTING G-FUNCTION______________________________________________________
        self.G_plot = G_Plot("GPlot",G_Function)
        self.G_plot.plotting(self.function.t,self.function.t0)
        G_toolbar = NavigationToolbar(self.G_plot.canvas, self)
        G_toolbarbox = gtk.HBox()
        G_image = gtk.Image()
        G_image.set_from_stock(gtk.STOCK_PROPERTIES, gtk.ICON_SIZE_LARGE_TOOLBAR)
        G_options_button = gtk.Button()
        G_options_button.add(G_image)
        #G_options_button.connect('clicked',self.mpl_options)
        G_image2 = gtk.Image()
        G_image2.set_from_stock(gtk.STOCK_REFRESH, gtk.ICON_SIZE_LARGE_TOOLBAR)
        G_refresh_button = gtk.Button()
        G_refresh_button.add(G_image2)
        #G_refresh_button.connect('clicked',self.on_refresh_clicked)
        G_toolbarbox.pack_start(G_toolbar, True, True)
        G_toolbarbox.pack_end(G_options_button, False, True)
        G_toolbarbox.pack_end(G_refresh_button, False, True)
        G_vbox = gtk.VBox()
        G_vbox.pack_start(G_toolbarbox, False, False)
        G_vbox.pack_start(self.G_plot.canvas, True, True)      
        #__________________________________________________________________

        # TEXT SCREEN______________________________________________________
        self.scroll_text = LogConsole.LogConsole(self)
        text_view_box = gtk.VBox(False,5)
        text_view_box.pack_start(self.scroll_text.scrolled_window,True,True)
        text_view_box.set_size_request(0,120)
        vbox1.pack_end(text_view_box,True,False)

        # ESTIMATOR________________________________________________________
	estimator_box = gtk.VBox()
	estimator = Estimator(self,self.function)
        estimator_box.pack_start(estimator.hbox,True,True,20)
        #estimator_box.pack_start(self.scroll_estxt,True,True)
        #__________________________________________________________________
        
        # Y FUNCTION TAB_____________________________________________________
        function_box = gtk.HBox(False,5)
        self.ftxt = gtk.TextView()
        self.ftxt.set_wrap_mode(gtk.WRAP_WORD)
        self.scroll_ftxt = gtk.ScrolledWindow()
        self.scroll_ftxt.set_shadow_type(gtk.SHADOW_ETCHED_IN)
        self.scroll_ftxt.set_policy(gtk.POLICY_AUTOMATIC,gtk.POLICY_AUTOMATIC)
        self.scroll_ftxt.add(self.ftxt)
        #label_Y = gtk.Label()
        #label_Y.set_use_markup(True)
        #label_Y.set_markup('Y = c + &#945;D + &#946;D<sup>2</sup>')
        label_Y = gtk.image_new_from_file('./equations/YfuncG1.png')
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
        check_err = gtk.CheckButton("Plot uncertainty band")
        check_ci_curve = gtk.CheckButton("Plot CI95% band (curve)")
        check_ci_points = gtk.CheckButton("Plot CI95% band (points)")
        check_function.set_active(True)
        check_err.set_active(True)
        check_ci_curve.set_active(True)
        check_ci_points.set_active(True)
        vbox_checks = gtk.VBox(False, 5)
        vbox_checks.pack_start(check_function, False, False)
        vbox_checks.pack_start(check_err, False, False)
        vbox_checks.pack_start(check_ci_curve, False, False)
        vbox_checks.pack_start(check_ci_points, False, False)
        check_function.connect('toggled',self.on_toggled, 'function')
        check_err.connect('toggled',self.on_toggled, 'err')
        check_ci_curve.connect('toggled',self.on_toggled, 'ci_curve')
        check_ci_points.connect('toggled',self.on_toggled, 'ci_points')
        hbox_buttons = gtk.HBox(True,5)
        button_save_f = gtk.Button("Save Funtion")
        button_load_f = gtk.Button("Load Funtion")
        hbox_buttons.pack_start(button_save_f,True,True)
        hbox_buttons.pack_start(button_load_f,True,True)
        button_save_f.connect('clicked',self.save_function)
        button_load_f.connect('clicked',self.load_function_filechooser)
        left_box = gtk.VBox(False,5)
        ruler_f1 = gtk.HSeparator()
        ruler_f2 = gtk.HSeparator()
        left_box.pack_start(label_Y, False, False,5)
        left_box.pack_start(table_f, False, False)
        left_box.pack_start(ruler_f1, False, True, 5)
        left_box.pack_start(vbox_checks, False, False)
        left_box.pack_start(ruler_f2, False, True, 5)
        left_box.pack_start(hbox_buttons, False, True)
        function_box.pack_start(left_box, False, False, 5)
        ruler_f3 = gtk.VSeparator()
        function_box.pack_start(ruler_f3, True, True, 5)
        function_box.pack_end(estimator_box, True, True, 5)
        #__________________________________________________________________

        # G FUNCTION TAB_____________________________________________________
        G_function_box = gtk.HBox(False,5)
        label_G = gtk.image_new_from_file('./equations/Gfunc1.png')
        self.entry_t = gtk.Entry()
        self.entry_t.set_width_chars(5)
	self.entry_t.set_text('%.3f' % self.function.t)
	self.entry_t.connect("activate",self.t_changed)
        label_t = gtk.Label('t: ')
        self.entry_t0 = gtk.Entry()
        self.entry_t0.set_width_chars(5)
	self.entry_t0.set_text('%.3f' % self.function.t0)
	self.entry_t0.connect("activate",self.t0_changed)
        label_t0 = gtk.Label('t0: ')
        table_g = gtk.Table(6,3)
        table_g.attach(label_t,0,1,0,1)
        table_g.attach(self.entry_t,1,2,0,1)
        table_g.attach(label_t0,3,4,0,1)
        table_g.attach(self.entry_t0,4,5,0,1)
        G_vruler = gtk.VSeparator()
        table_g.attach(G_vruler,2,3,0,3,xpadding=10)
        G_left_box = gtk.VBox(False,5)
        ruler_g1 = gtk.HSeparator()
        ruler_g2 = gtk.HSeparator()
        G_left_box.pack_start(label_G, False, False, 5)
        G_left_box.pack_start(table_g, False, False)
        G_left_box.pack_start(ruler_g1, False, True, 5)
        G_function_box.pack_start(G_left_box, True, False, 5)
        #__________________________________________________________________

        # NOTEBOOK WRAP____________________________________________________
        self.notebook = gtk.Notebook()
        self.notebook.append_page(function_box, gtk.Label('Calibration function'))
        #self.notebook.append_page(estimator_box, gtk.Label('Estimator'))
        #__________________________________________________________________
        
        # 2nd NOTEBOOK WRAP____________________________________________________
        self.notebook2 = gtk.Notebook()
        self.notebook2.append_page(G_function_box, gtk.Label('Exposure function'))
        #__________________________________________________________________
        
        table_plots_and_notebooks = gtk.Table(2,2)
        table_plots_and_notebooks.attach(Y_vbox,0,1,0,1)
        table_plots_and_notebooks.attach(G_vbox,1,2,0,1)
        table_plots_and_notebooks.attach(self.notebook,0,1,1,2)
        table_plots_and_notebooks.attach(self.notebook2,1,2,1,2)
        table_plots_and_notebooks.set_row_spacings(5)
        table_plots_and_notebooks.set_col_spacings(5)
        vbox1.pack_end(table_plots_and_notebooks,True,True)
        
        self.log("Protracted Exposure -- Program Started -- Welcome!")
	self.function_changed()

	self.plot.plotting(self.function)
	self.G_plot.plotting(self.function.t,self.function.t0)

    def log(self,txt):
	"""Logging into main log console"""
        self.scroll_text.log(txt)

    def on_refresh_clicked(self,button) :
        """Refresh canvas - plot everything again"""
        self.plot.plotting(self.function)
        
    def function_changed(self):
	self.entry_c.set_text('%.3f' % self.function.params[2])
	self.entry_alpha.set_text('%.3f' % self.function.params[1])
	self.entry_beta.set_text('%.3f' % self.function.params[0])
	self.entry_sc.set_text('%.3f' % self.function.std_err[2])
	self.entry_salpha.set_text('%.3f' % self.function.std_err[1])
	self.entry_sbeta.set_text('%.3f' % self.function.std_err[0])

	self.log("\n\tparams:\t[beta\talpha\tc ]" + \
"\n\tvalues\t\t" + str(self.function.params) + \
"\n\tstd_err\t" + str(self.function.std_err) + \
"\n\tp-value\t" + str(self.function.p_value) + \
"\n\tRSS\t" + str(self.function.rss) + \
"\n\tRMSE\t" + str(self.function.rmse) + \
"\n\t---------------------------------------------------------------------------")

    def t_changed(self,entry):
	self.function.t=float(self.entry_t.get_text())
	self.G_plot.plotting(self.function.t,self.function.t0)
	self.plot.plotting(self.function)
            
    def t0_changed(self,entry):
	self.function.t0=float(self.entry_t0.get_text())
	self.G_plot.plotting(self.function.t,self.function.t0)
	self.plot.plotting(self.function)
            
    def on_combo_changed(self,cb,n):
        """For a given style combo change plotting attributes according to what was set in options."""
        model = cb.get_model()
        index = cb.get_active()
        cb.set_active(index)
        self.pstyle[n] = model[index][0]
        self.plot.nselec[n]=index
        self.plot.plotting(self.function)

    def on_spin_changed(self,spin,n) :
        """For a given style spin change plotting attributes according to what was set in options."""
        self.pstyle[n] = spin.get_value()
        self.plot.plotting(self.function)

    def on_entry_changed(self,entry,n) :
        """For a given plot label change plotting attributes according to what was set in options."""
        self.plot_labels[n] = entry.get_text()
        self.plot.plotting(self.function)

    def on_toggled(self,button,s) :
        """Toggle plotting one of the given features"""
        if(s=='ci_points'): self.plot.ci_points_toggle*=-1
        elif(s=='ci_curve'): self.plot.ci_func_toggle*=-1
        elif(s=='function'): self.plot.function_toggle*=-1
        elif(s=='err'): self.plot.err_toggle*=-1
        self.plot.plotting(self.function)
        
    def on_method_changed(self,cb):
        """When estimation method is changed make appropriate changes in DM"""
        model = cb.get_model()
        index = cb.get_active()
        cb.set_active(index)
        self.method = model[index][0]
        
    def mpl_options(self,button) :
        """Create GTKDialog containing options for plotting and connect signals."""
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
            
    def save_function(self,button) :
        """Saving function parameters, errors, covariance matrix, rss and rmse."""

        file_chooser = gtk.FileChooserDialog("Open...", self, gtk.FILE_CHOOSER_ACTION_SAVE, (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_SAVE, gtk.RESPONSE_OK))
        response = file_chooser.run()
        path=''
        if response == gtk.RESPONSE_OK :
            path = file_chooser.get_filename()
            self.log('Curve saved in file:   ' + path)
            self.log("---------------------------------------------------------------------------")
            if ".csv" not in path:
                path = path + '.csv'
            file_chooser.destroy()

            ofile = open(path,"wb")
            writer = csv.writer(ofile, delimiter=',')
            writer.writerow(self.function.params)
            writer.writerow(self.function.std_err)
            writer.writerow(self.function.p_value)
            writer.writerow(self.function.cov_mtx[0])
            writer.writerow(self.function.cov_mtx[1])
            writer.writerow(self.function.cov_mtx[2])
            writer.writerow((self.function.rss, self.function.rmse, 0.0))
            ofile.close()
        else :
            file_chooser.destroy()

    def load_function_filechooser(self,button) :
        """Load function from file."""
        file_chooser = gtk.FileChooserDialog("Open...", self, gtk.FILE_CHOOSER_ACTION_OPEN, (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_OPEN, gtk.RESPONSE_OK))
        response = file_chooser.run()
        path=''
        if response == gtk.RESPONSE_OK :
            path = file_chooser.get_filename()
            self.load_function(path,self.function,self.plot)
            self.log('Loaded curve from file:   ' + path)
            self.log("---------------------------------------------------------------------------")
            
    def load_function(self,path,function,plot) :
        """Load function from file."""
        f = open(path, 'rt')
        try:
            reader = csv.reader(f)
            l=list(reader)
            print l
            function.params=[float(i) for i in l[0]]
            function.std_err=[float(i) for i in l[1]]
            function.p_value=[float(i) for i in l[2]]
            function.cov_mtx=[[float(i) for i in l[3]],[float(i) for i in l[4]],[float(i) for i in l[5]]]
            function.rss=float(l[6][0])
            function.rmse=float(l[6][1])
            plot.fit_toggle='inactive'
            plot.plotting(function)
        finally:
            f.close()
            
    def create_combobox(self,slist,whereto,n) :
        """Create combobox from list of strings"""
        combo = gtk.combo_box_new_text()
        whereto[n].pack_start(combo)
        for style in slist :
            combo.append_text(str(style))
        combo.set_active(self.plot.nselec[n])
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
        entry_title.set_text(self.plot_labels[m])
        whereto[n].pack_start(entry_title)
        entry_title.connect("activate",self.on_entry_changed,m)


# Main

#______________MAIN______________#

#manager = UserInterface("",[" "," "," "," "," "])
#manager.show_all()
#gtk.main()

