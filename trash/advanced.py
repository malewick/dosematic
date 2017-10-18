#!/usr/bin/env python

#from pyexcel_xls import get_data
import csv

#import pygtk
#pygtk.require('2.0')
import gtk
from gtk import gdk

import matplotlib
matplotlib.use('GTKAgg')  # or 'GTK'
import matplotlib.pyplot as plt
from matplotlib.backends.backend_gtk import FigureCanvasGTK as FigureCanvas
from matplotlib.backends.backend_gtkagg import NavigationToolbar2GTKAgg as NavigationToolbar

from matplotlib.figure import Figure

from numpy  import array
from scipy import stats
import numpy as np
from scipy.optimize import curve_fit

#_________________________________________________________________________________________________#

# --- GLOBAL VARIABLES ---
labels=[]
data=[]
# --- GLOBAL FUNCTIONS ---
###################################################################################################
def isfloat(s):
        try:
                float(s)
                return True
        except:
                return False
###################################################################################################
def read_data_csv():

    ifile  = open('adv.csv', "rb")
    reader = csv.reader(ifile, delimiter=',')

    global labels
    global data
    # data: n of cells | 0|.|.|.|10 and more|
    temp_data=[]
    temp_row=[]

    rownum = 0
    for row in reader:
        if rownum == 0:
            labels = row
        else:
            for x in row :
                if isfloat(x) : temp_row.append(float(x))
                else : temp_row.append(0.0)
            temp_data.append(list(temp_row))
        temp_row[:]=[]
        rownum += 1
    ifile.close()
    data = zip(*temp_data)
    data = [list(x) for x in data]
###################################################################################################

# --- CLASSES ---
class DataManager(gtk.Window):
    read_data_csv()
    global labels
    global data
    data=array(data)
    numRows=len(data[0])
    numCols=len(labels)

    def __init__(self):

        gtk.Window.__init__(self)
        self.set_default_size(600, 800)
        self.connect('destroy', lambda win: gtk.main_quit())

        self.data=data

        self.set_title('Advanced mode')
        self.set_border_width(8)

        hbox = gtk.HBox(False, 8)
        self.add(hbox)
        vbox1 = gtk.VBox(False,8)
        vbox2 = gtk.VBox(False,8)
        hbox.pack_start(vbox1, True, True)
        hbox.pack_start(vbox2, True, True)

        label = gtk.Label('Advanced mode')
        vbox2.pack_start(label, False, False)

        # TREE VIEW________________________________________________________
        self.model = self.create_model()                # MODEL
        self.treeview = gtk.TreeView(self.model)        # TREE VIEW
        self.treeview.set_rules_hint(True)
        self.add_columns()                              # FILL COLUMNS
        # -> TREE VIEW BUTTONS
        button_add1 = gtk.Button('Add row')             # ADD 1 ROW
        button_add10 = gtk.Button('Add 10 rows')        # ADD 10 ROWS
        hbox_buttons = gtk.HBox(False,5)                # layout packaging
        hbox_buttons.pack_start(button_add1, True, True)
        hbox_buttons.pack_start(button_add10, True, True)
        vbox2.pack_end(hbox_buttons, False, False)
        button_add1.connect('clicked',self.add_rows,1)  # SIGNALS HANDLING
        button_add10.connect('clicked',self.add_rows,10)
        # -> INTO SCROLABLE WINDOW
        self.sw = gtk.ScrolledWindow()
        self.sw.set_shadow_type(gtk.SHADOW_ETCHED_IN)
        self.sw.set_policy(gtk.POLICY_NEVER,gtk.POLICY_AUTOMATIC)
        self.sw.add(self.treeview)                      # ADD TO SW
        vbox2.pack_start(self.sw, True, True)
        #__________________________________________________________________

        # TEXT SCREEN______________________________________________________
        self.text = gtk.TextView()                              # TEXT VIEW
        self.text.set_wrap_mode(gtk.WRAP_WORD)          # wrap words
        self.scroll_text = gtk.ScrolledWindow()         # into scrollable env
        self.scroll_text.set_shadow_type(gtk.SHADOW_ETCHED_IN)
        self.scroll_text.set_policy(gtk.POLICY_AUTOMATIC,gtk.POLICY_AUTOMATIC)
        self.scroll_text.add(self.text)
        text_label = gtk.Label('Log console')
        text_view_box = gtk.VBox(False,5)
        text_view_box.pack_start(text_label, False, False)
        text_view_box.pack_start(self.scroll_text,True,True)
        vbox1.pack_end(text_view_box,True,True)
        #__________________________________________________________________

    def log(self,txt):
        end_iter = self.text.get_buffer().get_end_iter()
        self.text.get_buffer().insert(end_iter, txt+'\n')
        adj = self.scroll_text.get_vadjustment()
        adj.set_value( adj.upper - adj.page_size )


    def add_columns(self):
        for i in range(self.numCols):
            renderer = gtk.CellRendererText()
            renderer.props.wrap_width = 100

            if i>0:
                renderer.set_property('editable', True)
                renderer.connect('edited',self.edited_cb, (self.model, i))
            else :
                renderer.set_property('editable', False)
            renderer.props.wrap_mode = "PANGO_WRAP_WORD"
            renderer.props.wrap_width = 70
            column = gtk.TreeViewColumn(labels[i], renderer, text=i)
            column.set_resizable(True)
            column.set_sizing(gtk.TREE_VIEW_COLUMN_FIXED)
            column.set_min_width(30)
            column.set_fixed_width(70)
            column.set_expand(False)
            self.treeview.append_column(column)

    def edited_cb(self, cell, path, new_content, user_data):
        liststore, column = user_data
        if isfloat(new_content) and float(new_content)>=0.0 :
            liststore[path][column] = float(new_content)
            self.data[int(column)][int(path)] = float(new_content)
            print "data[", column, "][", path, "]  = ", float(new_content)

    def add_rows(self,button,n):
        self.log('n of rows to add: ' + str(n))
        for i in range(0,n) :
            self.model.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            self.data = np.insert( self.data, len(self.data[0]), values=0, axis=1 )
        adj = self.sw.get_vadjustment()
        adj.set_value( adj.upper - adj.page_size )

    def create_model(self):
        types = [float]*self.numCols
        store = gtk.ListStore(*types)

        temp=zip(*self.data)
        for row in temp:
            store.append(row)
        return store


#________________________________MAIN_____________________________________________________________#

manager = DataManager()
manager.show_all()
gtk.main()
