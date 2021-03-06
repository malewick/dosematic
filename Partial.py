
# coding: utf-8

import csv
import gtk
from gtk import gdk
import pango
import numpy as np
from numpy  import array
from datetime import datetime

import scipy
import scipy.special

import LogConsole

print "GTK v" + '.'.join(str(i) for i in gtk.gtk_version)
print "numpy v" + np.__version__

def isfloat(s):
    try:
            float(s)
            return True
    except:
            return False

class Data():
    def __init__(self, labels):

        self.labels_input=[]
        
        self.read_data_init()

        self.numRows_input=len(self.table[0])
        self.numCols_input=len(self.labels_input)

    def read_data_init(self):

        ifile  = open('data/partial_dic.csv', "rb")
        reader = csv.reader(ifile, delimiter=',')
        rownum = 0
        data=[]
        for row in reader:
            if rownum == 0:
                self.labels_input = row
                rownum += 1
            else:
                row =  [float(x) for x in row]
                data.append(list(row))
        ifile.close()
        data = zip(*data)
        data = [list(x) for x in data]
        self.table = array(data)

        ifile  = open('data/partial_rings.csv', "rb")
        reader = csv.reader(ifile, delimiter=',')
        data=[]
        iterreader = iter(reader)
        next(iterreader)
        for row in iterreader:
            row =  [float(x) for x in row]
            data.append(list(row))
        ifile.close()
        data = zip(*data)
        data = [list(x) for x in data]
        self.table_rings = array(data)

        ifile  = open('data/partial_acen.csv', "rb")
        reader = csv.reader(ifile, delimiter=',')
        data=[]
        iterreader = iter(reader)
        next(iterreader)
        for row in iterreader:
            row =  [float(x) for x in row]
            data.append(list(row))
        ifile.close()
        data = zip(*data)
        data = [list(x) for x in data]
        self.table_acentrics = array(data)

    def read_data_csv(self, filename, tag):

        ifile  = open(filename, "rb")
        reader = csv.reader(ifile, delimiter=',')
        data=[]
	rownum=0
        for row in reader:
            if rownum == 0:
                self.labels_input = row
                rownum += 1
            else:
                row =  [float(x) for x in row]
                data.append(list(row))
        ifile.close()
        data = zip(*data)
        data = [list(x) for x in data]
        if 'dicentrics' in tag :
            self.table = array(data)
        elif 'rings' in tag :
            self.table_rings = array(data)
        elif 'acentrics' in tag :
            self.table_acentrics = array(data)

        self.numRows_input=max([len(self.table[0]),len(self.table_rings[0]),len(self.table_acentrics[0])])

	if len(data)==0:
	    return 0
        elif len(data)==1:
	    return 1
        elif len(data)==2:
	    return 2

	return 3

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


class Results():

    def __init__(self):

        self.labels = ["N","Xd","Xr","Xa","Xd/N","SE","dix","u-test","Y Dolphin","Y1","Y2","Qdr","Y Qdr"]
        self.dscrpt = ["number of scored cells","number of dicentrics observed","number of rings observed","number of acentrics observed","number of dicentrics per cell","standard error","dispertion index","u-test value","yield according to Dolphin method","yield of dicentrics plus rings","yield of acentrics","yield of dicentrics and rings among damaged cells","yield according to Qdr method"]

        self.numRows_output=0
        self.numCols_output=len(self.labels)

        # logic 
        self.N  = 0.0   
        self.Nu = 0.0 
        self.n0 = 0.0 
        self.Xd = 0.0  
        self.Xr = 0.0 
        self.Xa = 0.0 
        self.dix= 0.0
        self.utst=0.0
        self.Ydol=0.0
        self.Y1 = 1.0
        self.Y2 = 1.0
        self.Qdr= 0.0
        self.Yqdr=0.0

    def yieldDolphin(self,N,X,n0):
        """
        input:
        N - total number of scored cells
        X - total number of observed dicentrics
        n0 - number of cells free of dicentrics
        return:
        Y - yield calculated with "Dolphin method"
        f - fraction of irradiated body calculated with "Dolphin method"
        """
        if N == 0.0 or (n0-N) == 0:
            return -1,-1
        Y = scipy.special.lambertw( X*np.exp(X/(n0-N)) / (n0-N) ) - X/(n0-N)
        if Y==0 or Y.imag!=0:
            return 0,-1
        f = X/Y/N
        return Y.real, f.real

    def yieldQdr(self,Nu,X,Y1,Y2):
        """
        input:
        Nu - total number of damged cells
        X - total number of dicentrics and rings
        Y1 - yield of dicentrics plus rings
        Y2 - yield of excess acentrics
        return:
        Qdr - yield of dicentrics and rings among damaged cells
        Y - yield calculated with "Qdr method"
        """
        if Nu == 0.0 :
            return -1,-1
        Qdr = X/Nu
        Y = Qdr * (1-np.exp(-Y1-Y2))
        return Y, Qdr

# --- CLASSES ---
class UserInterface(gtk.Window):

    def __init__(self,module_name,labels):
        """Init gtk window and build the GUI."""
        
        gtk.Window.__init__(self)
        self.set_default_size(600, 800)
        #self.connect('destroy', lambda win: gtk.main_quit())

        self.data=Data(labels)
        self.results = Results()

        self.set_title('Partial Dose')
        self.set_border_width(8)

        vbox = gtk.VBox(False, 8)
        self.add(vbox)
        hbox = gtk.HBox(False, 8)
        vbox1 = gtk.VBox(False,8)
        vbox2 = gtk.VBox(False,8)
        vbox3 = gtk.VBox(False,8)
        vbox.pack_start(hbox, True, True)
        vbox.pack_start(vbox1, False, False)
        hbox.pack_start(vbox2, True, True)
        hbox.pack_start(vbox3, True, True)

        label2 = gtk.Label('Input')
        vbox2.pack_start(label2, False, False)

        # INPUT TREE VIEW__________________________________________________
        self.input_model = self.create_input_model("dicentrics")
        self.rings_model = self.create_input_model("rings")
        self.acen_model = self.create_input_model("acentrics")

        self.input_treeview = gtk.TreeView(self.input_model)
        self.input_treeview.set_rules_hint(True)
        self.input_treeview.set_size_request(504,0)
        self.add_input_colums(self.input_treeview,self.input_model)
        self.input_sw = gtk.ScrolledWindow()
        self.input_sw.set_shadow_type(gtk.SHADOW_ETCHED_IN)
        self.input_sw.set_policy(gtk.POLICY_AUTOMATIC,gtk.POLICY_AUTOMATIC)
        self.input_sw.add(self.input_treeview)

        self.rings_treeview = gtk.TreeView(self.rings_model)
        self.rings_treeview.set_rules_hint(True)
        self.rings_treeview.set_size_request(504,0)
        self.add_input_colums(self.rings_treeview,self.rings_model) 
        self.rings_sw = gtk.ScrolledWindow()
        self.rings_sw.set_shadow_type(gtk.SHADOW_ETCHED_IN)
        self.rings_sw.set_policy(gtk.POLICY_AUTOMATIC,gtk.POLICY_AUTOMATIC)
        self.rings_sw.add(self.rings_treeview)

        self.acen_treeview = gtk.TreeView(self.acen_model)
        self.acen_treeview.set_rules_hint(True)
        self.acen_treeview.set_size_request(504,0)
        self.add_input_colums(self.acen_treeview,self.acen_model)
        self.acen_sw = gtk.ScrolledWindow()
        self.acen_sw.set_shadow_type(gtk.SHADOW_ETCHED_IN)
        self.acen_sw.set_policy(gtk.POLICY_AUTOMATIC,gtk.POLICY_AUTOMATIC)
        self.acen_sw.add(self.acen_treeview)

        self.notebook = gtk.Notebook()
        self.notebook.append_page(self.input_sw, gtk.Label('Dicentrics'))
        self.notebook.append_page(self.rings_sw, gtk.Label('Rings'))
        self.notebook.append_page(self.acen_sw, gtk.Label('Acentrics'))
        vbox2.pack_start(self.notebook, True, True)

        # -> TREE VIEW BUTTONS
        button_add1 = gtk.Button('Add row')    
        button_clear = gtk.Button('Clear data')
        button_save = gtk.Button('Export to csv')
        button_load = gtk.Button('Load csv')

	grid = gtk.Table(2,2)
	grid.attach(button_add1, 0,1,0,1)
	grid.attach(button_clear, 1,2,0,1)
	grid.attach(button_save, 0,1,1,2)
	grid.attach(button_load, 1,2,1,2)

        vbox2.pack_end(grid, False, False)
        button_add1.connect('clicked',self.add_rows,1)  # SIGNALS HANDLING
        button_clear.connect('clicked',self.clear_all)
        button_save.connect('clicked',self.save_input,self.notebook)
        button_load.connect('clicked',self.load_input,self.notebook)
        #__________________________________________________________________

        label3 = gtk.Label('Output')
        vbox3.pack_start(label3, False, False,14)

        # OUTPUT TREE VIEW__________________________________________________
        self.output_model = self.create_output_model()    # output_model
        self.output_treeview = gtk.TreeView(self.output_model)  # TREE VIEW
        self.output_treeview.set_rules_hint(True)
        self.output_treeview.set_size_request(780,0)
        self.add_output_colums()                              # FILL COLUMNS
        # -> INTO SCROLABLE WINDOW
        self.output_sw = gtk.ScrolledWindow()
        self.output_sw.set_shadow_type(gtk.SHADOW_ETCHED_IN)
        self.output_sw.set_policy(gtk.POLICY_AUTOMATIC,gtk.POLICY_AUTOMATIC)
        self.output_sw.add(self.output_treeview)                      # ADD TO output_sw

        vbox3.pack_start(self.output_sw, True, True)
        #__________________________________________________________________

        button_save_out = gtk.Button('Export to csv')

	grid_out = gtk.Table(2,1)
	grid_out.attach(button_save_out, 0,1,0,1)

        vbox3.pack_end(grid_out, False, False)

        button_save_out.connect('clicked',self.save_output)
 
        # TEXT SCREEN______________________________________________________
        self.text = LogConsole.LogConsole(self)
        text_view_box = gtk.VBox(False,5)
        text_view_box.pack_start(self.text.scrolled_window,True,True)
        text_view_box.set_size_request(0,160)
        vbox1.pack_end(text_view_box,True,False)
        
        self.calculate_output()

    def log(self,txt):
        """Logging into main log console"""
        self.text.log(txt)

    def clear_rows(self,button,notebook):

	idx = notebook.get_current_page()
	if idx == 0 :
	    model = self.input_treeview.get_model()
	    table = self.data.table
	elif idx == 1 :
	    model = self.rings_treeview.get_model()
	    table = self.data
	elif idx == 2 :
	    model = self.acen_treeview.get_model()
	    table = self.data.table_acentrics
        model.clear()
        model.append([0,0,0,0,0,0,0,0,0,0,0,0])
        table = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]], dtype='f')

    def clear_all(self,button):

	model1 = self.input_treeview.get_model()
	table1 = self.data.table
	model2 = self.rings_treeview.get_model()
	table2 = self.data.table_rings
	model3 = self.acen_treeview.get_model()
	table3 = self.data.table_acentrics
        model1.clear()
        model2.clear()
        model3.clear()
        model1.append([0,0,0,0,0,0,0,0,0,0,0,0])
        model2.append([0,0,0,0,0,0,0,0,0,0,0,0])
        model3.append([0,0,0,0,0,0,0,0,0,0,0,0])
        table1 = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]], dtype='f')
        table2 = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]], dtype='f')
        table3 = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]], dtype='f')

	self.output_model.clear()
	self.output_model.append(13*[0])
	self.results.numRows_output=1

    def save_input(self,button,notebook) :
	self.text.log("saving input")
	self.text.log(str(notebook.get_current_page()))

    def load_input(self,button,notebook) :
	self.text.log("loading input")

	tag=""
	if notebook.get_current_page()==0 :
	    tag='dicentrics'
	    model=self.input_model
	    treeview=self.input_treeview
	elif notebook.get_current_page()==1 :
	    tag='rings'
	    model=self.rings_model
	    treeview=self.rings_treeview
	elif notebook.get_current_page()==2 :
	    tag='acentrics'
	    model=self.acen_model
	    treeview=self.acen_treeview

	self.text.log(str(notebook.get_current_page())+"\t"+tag)

        fc = MyFileChooser()
        file_chooser = fc.get_filechooser()

        response = file_chooser.run()
        if response == gtk.RESPONSE_OK:
	    path = file_chooser.get_filename()
            if ".csv" in path:

		read_data_flag = self.data.read_data_csv(path,tag)
                if read_data_flag == 0:
		    self.context.log("error while loading file: "+path)
                    return
                elif read_data_flag <= 2:
		    self.context.log("error while reading columns in file: "+path+". File not loaded! Data table left unmodified.")
                    return
                self.text.log("Loaded data file: "+path)

		model = self.create_input_model(tag)
		treeview.set_model(model)

            else :
		self.text.log("___Wrong file format!___")
            file_chooser.destroy()
        elif response == gtk.RESPONSE_CANCEL:
	    file_chooser.destroy()


    def save_output(self,button) :
	self.text.log("saving output")

    def add_input_colums(self,treeview,model):
        """cell renderer"""
        for i in range(self.data.numCols_input):
            renderer = gtk.CellRendererText()
            renderer.set_property('editable', True)
            renderer.set_property('xalign', 1.0)
            renderer.connect('edited',self.edited_cb, (model, i))
            renderer.props.wrap_mode = "PANGO_WRAP_WORD"
            renderer.props.wrap_width = 50
            column = gtk.TreeViewColumn(self.data.labels_input[i], renderer, text=i)
            column.set_resizable(True)
            column.set_sizing(gtk.TREE_VIEW_COLUMN_FIXED)
            column.set_fixed_width(42)
            column.set_expand(False)
            treeview.append_column(column)



            # TODO -- something's fucked up!
#	    treeview.connect("key-release-event", self.on_navigate_key)
#
#    def on_navigate_key(self, treeview, event):
#        keyname = gdk.keyval_name(event.keyval)
#        path, col = treeview.get_cursor()
#        columns = [c for c in treeview.get_columns()] 
#        colnum = columns.index(col)        
#        nrows = len(treeview.get_model())
#
#        if keyname == 'Tab':
#            newcol = treeview.get_column((colnum+1)%3)
#            newpath=path
#            treeview.set_cursor(newpath, newcol, True)
#
#        elif keyname == 'Return':
#            newcol = treeview.get_column(colnum)
#            if path[0]+1 < nrows :
#                newpath=(path[0]+1,)
#                treeview.set_cursor(newpath, newcol, True)
#
#        else:
#            pass
#



            
    def cell_coloring(self, column, renderer, model, itr, data) :
        utest_value = float(model.get_value(itr, 7))
        if abs(utest_value)>1.96 and column==self.output_treeview.get_column(7) :
            renderer.set_property('background-gdk', gtk.gdk.Color(red=65535-1000, green=65535-11000, blue=65535-11000))
        else :
            renderer.set_property('background-gdk', gtk.gdk.Color(red=65535, green=65535, blue=65535))


    def add_output_colums(self):
        """cell renderer"""
        for i in range(self.results.numCols_output):
            renderer = gtk.CellRendererText()
            renderer.props.wrap_width = 100
            renderer.set_property('editable', False)
            renderer.set_property('xalign', 1.0)
            renderer.props.wrap_mode = "PANGO_WRAP_WORD"
            renderer.props.wrap_width = 60
            column = gtk.TreeViewColumn(self.results.labels[i], renderer, text=i)
            column.set_resizable(True)
            column.set_sizing(gtk.TREE_VIEW_COLUMN_FIXED)
            column.set_fixed_width(60)
            column.set_expand(False)
            self.output_treeview.append_column(column)

            column.set_cell_data_func(renderer, self.cell_coloring, None)

    def edited_cb(self, cell, path, new_content, user_data):
        """handling signal of edited cell"""
        liststore, column = user_data

        if isfloat(new_content) and float(new_content)>=0.0 :

            liststore[path][column] = float(new_content)
            self.data.table[int(column)][int(path)] = float(new_content)

            print "data[", column, "][", path, "]  = ", self.data.table[int(column)][int(path)]
            print "model[", column, "][", path, "]  = ", liststore[path][column]

        self.calculate_output()
            
    def calculate_output(self) :
        """Calculating the output basing on the input treeview"""

        self.results.numRows_output=self.data.numRows_input
        nRows=self.data.numRows_input
        nCols=self.data.numCols_input
        
        number_of_scored_cells= np.array([0.0]*nRows, dtype=np.float)
        number_of_damaged_cells=np.array([0.0]*nRows, dtype=np.float)
        number_of_zero_cells= np.array([0.0]*nRows, dtype=np.float)
        number_of_dicentrics =  np.array([0.0]*nRows, dtype=np.float)
        number_of_rings      =  np.array([0.0]*nRows, dtype=np.float)
        number_of_acentrics  =  np.array([0.0]*nRows, dtype=np.float)
        mean_dic_per_cell=nRows*[0.0]
        standard_deviation=nRows*[0.0]
        standard_error=nRows*[0.0]

        nr=0
        for row in self.input_model:
            for n in range(0,nCols):
                number_of_scored_cells[nr] += float(row[n])
                if n == 0 :
                    number_of_zero_cells[nr] += float(row[n])
                else :
                    number_of_damaged_cells[nr]  +=  float(row[n])

                number_of_dicentrics[nr] += n*float(row[n])
            nr += 1

        nr=0
        for row in self.input_model:
            mean_dic_per_cell[nr] = number_of_dicentrics[nr]/number_of_scored_cells[nr]
            stddev = 0.0
            for n in range(0,nCols):
                brackets = float(row[n]) - mean_dic_per_cell[nr]
                stddev += brackets*brackets
            standard_deviation[nr] = np.sqrt(stddev/(number_of_scored_cells[nr]-1))
            standard_error[nr] = standard_deviation[nr]/np.sqrt(number_of_scored_cells[nr])
            nr += 1

        nr=0
        for row in self.rings_model:
            for n in range(0,nCols):
                number_of_rings[nr] += n*float(row[n])
            nr += 1

        nr=0
        for row in self.acen_model:
            for n in range(0,nCols):
                number_of_acentrics[nr] += n*float(row[n])
            nr += 1

        dispertion_index=nRows*[0.0]
        utest=nRows*[0.0]
        yieldDol=nRows*[0.0]
        yieldDol_f=nRows*[0.0]
        yieldDicRings=nRows*[0.0]
        yieldAcen=nRows*[0.0]
        Qdr=nRows*[0.0]
        yieldQdr=nRows*[0.0]

        for n in range(0,nRows):
            dispertion_index[n] = pow(standard_deviation[n],2) / mean_dic_per_cell[n]  if mean_dic_per_cell[n] else 0.0
            utest[n] = (dispertion_index[n] - 1.) * np.sqrt(2*(1-1./mean_dic_per_cell[n])) if mean_dic_per_cell[n] else 0.0
            yieldDol[n], yieldDol_f[n] = self.results.yieldDolphin(number_of_scored_cells[n],number_of_dicentrics[n],number_of_zero_cells[n])

            yieldDicRings[n]=(number_of_dicentrics[n]+number_of_rings[n])/number_of_scored_cells[n] if number_of_scored_cells[n] else 0.0
            yieldAcen[n]=(number_of_acentrics[n])/number_of_scored_cells[n] if number_of_scored_cells[n] else 0.0
            yieldQdr[n], Qdr[n] = self.results.yieldQdr(number_of_damaged_cells[n],number_of_dicentrics[n]+number_of_rings[n],yieldDicRings[n],yieldAcen[n])

        for n in range(0,nRows) :
            self.output_model[n][0]='%.0f' % (number_of_scored_cells[n])
            self.output_model[n][1]='%.0f' % (number_of_dicentrics[n])
            self.output_model[n][2]='%.0f' % (number_of_rings[n])
            self.output_model[n][3]='%.0f' % (number_of_acentrics[n])
            self.output_model[n][4]='%.2f' % (mean_dic_per_cell[n])
            self.output_model[n][5]='%.2f' % (standard_error[n])
            self.output_model[n][6]='%.2f' % (dispertion_index[n])
            self.output_model[n][7]='%.2f' % (utest[n])
            self.output_model[n][8]='%.2f' % (yieldDol[n])
            self.output_model[n][9]='%.2f' % (yieldDicRings[n])
            self.output_model[n][10]='%.2f' % (yieldAcen[n])
            self.output_model[n][11]='%.2f' % (Qdr[n])
            self.output_model[n][12]='%.2f' % (yieldQdr[n])

    def create_input_model(self,tag):
        types = [str]*self.data.numCols_input
        store = gtk.ListStore(*types)

        temp=[]
        if 'dicentrics' in tag :
            temp=zip(*self.data.table)
        elif 'rings' in tag :
            temp=zip(*self.data.table_rings)
        elif 'acentrics' in tag :
            temp=zip(*self.data.table_acentrics)

        for row in temp:
            srow = ['%.0f' % (i) for i in row]
            store.append(srow)
        return store

    def create_output_model(self):
        types = [str]*self.results.numCols_output
        store = gtk.ListStore(*types)

        for i in range(0,self.data.numRows_input):
            row=["0.0"]*self.results.numCols_output
            row[0]=str(i)
            store.append(row)
        return store
    
    def add_rows(self,button,n):
        self.log('n of rows to add: ' + str(n))

        for i in range(0,n) :

            self.input_model.append(self.data.numCols_input*["0"])
            self.data.table = np.insert( self.data.table, len(self.data.table[0]), values=0, axis=1 )

            self.rings_model.append(self.data.numCols_input*["0"])
            self.data.table_rings = np.insert( self.data.table_rings, len(self.data.table_rings[0]), values=0, axis=1 )

            self.acen_model.append(self.data.numCols_input*["0"])
            self.data.table_acentrics = np.insert( self.data.table_acentrics, len(self.data.table_acentrics[0]), values=0, axis=1 )

            self.output_model.append(self.results.numCols_output*["0"])

            self.results.numRows_output += 1
            self.data.numRows_input += 1

        adj = self.input_sw.get_vadjustment()
        adj.set_value( adj.upper - adj.page_size )
        self.calculate_output()

#________________________________MAIN_____________________________________________________________#

#manager = UserInterface("dupa",["labels"])
#manager.show_all()
#gtk.main()
