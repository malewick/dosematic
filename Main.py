# coding: utf-8

import csv

import gtk
from gtk import gdk

from collections import OrderedDict

from PIL import Image

import time
import thread

import SplashScreen
import Acute
import Partial
import Protracted
import FunctionComparison


print "GTK v" + '.'.join(str(i) for i in gtk.gtk_version)

# Names all availbale modules in the current version
# Keys below are used through this module
module_titles = {
    'dic': """Dicentric Assay""",
    'foci': """DNA Repair Foci""",
    'micro': """Micronucleus Assay""",
    'fish': """FISH Translocation""",
    'pccch': """PCC Chemically Induced""",
    'pccf': """PCC Fusion""",
    'other': """Other Modules"""
}

# Names of submodules for each cathegory
desc=OrderedDict()
desc["dic"]=  [ "Acute", "Partial", "Protracted" ]
desc["foci"]= [ "Acute", "Partial" ]
desc["micro"]=[ "Acute", "Partial", "Protracted" ]
desc["fish"]= [ "Acute", "Partial", "Protracted" ]
desc["pccch"]=[ "Acute" ]
desc["pccf"]= [ "Acute", "Partial"]
desc["other"]= [ "Function Comparison"]

# Images directories for each module
images=OrderedDict()
images["dic"]=["dic_acute.png","dic_partial.png","dic_protracted.png"]
images["foci"]=["foci_acute.png","foci_partial.png"]
images["micro"]=["micro_acute.png","micro_partial.png","micro_protracted.png"]
images["fish"]=["fish_acute.png","fish_partial.png","fish_protracted.png"]
images["pccch"]=["pccch_acute.png"]
images["pccf"]=["pccf_acute.png","pccf_partial.png"]
images["other"]=["func_comp.jpg"]

# keeps list of recently opened modules: {key:module_number}
module_titles_recent = {}

# general description of DoseMatic
general_description = """General Description"""

# These are full descritptions of each of the modules
full_description = {
    'dic': ["""Dicentric Assay 1""",
            """Dicentric Assay 2""",
            """Dicentric Assay 3"""],
    'foci': ["""DNA Repair Foci 1""",
             """DNA Repair Foci 3""",
             """DNA Repair Foci 3"""],
    'micro': ["""Micronucleus Assay 1""",
              """Micronucleus Assay 2""",
              """Micronucleus Assay 3"""],
    'fish': ["""FISH Translocation 1""",
             """FISH Translocation 2""",
             """FISH Translocation 3"""],
    'pccch': ["""PCC Chemically Induced 1""",
              """PCC Chemically Induced 2""",
              """PCC Chemically Induced 3"""],
    'pccf': ["""PCC Fusion 1""",
             """PCC Fusion 2""",
             """PCC Fusion 3"""],
    'other': ["""Function Comparison"""]
}

# type of the calibration function used
mode={}
mode["dic"]=  "quadratic"
mode["foci"]= "linear"
mode["micro"]="quadratic"
mode["fish"]= "quadratic"
mode["pccch"]="linear"
mode["pccf"]= "linear"
mode["other"]=""

# disctinctive labels for each of the cathegories
labels={}
labels["dic"]=  [ "Dose", "Cells counted", "Dicentrics counted", "Dicentrics per cell", "SE" ]
labels["foci"]= [ "Dose", "Cells counted", "Foci scored", "Foci per cell", "SE" ]
labels["micro"]=[ "Dose", "Cells counted", "Mn scored", "Mn per cell", "SE" ]
labels["fish"]= [ "Dose", "Cells counted", "Translocations scored", "Translocations per cell", "SE" ]
labels["pccch"]=[ "Dose", "Cells counted", "PCC rings", " ", " " ]
labels["pccf"]= [ "Dose", "Cells counted", "PCC", " ", " " ]
labels["other"]=[ "f label", "g label", "f xlabel", "g xlabel", "f ylabel", "g ylabel" ]


# # DataManager
# It is a class containing main GUI and connected signals opening desired modules.

class MainWindow(gtk.Window):

    def __init__(self):
        """Init function with the whole GUI declaration and signals"""
        
        gtk.Window.__init__(self)
        self.set_default_size(900, 800)
        self.connect('destroy', lambda win: gtk.main_quit())

        self.set_title('DOSEMATIC v0.2')
        
        # Add a VBox
        vbox = gtk.VBox()
        self.add(vbox)

        # Setup Scrolled Window
        scrolled_win = gtk.ScrolledWindow()
        scrolled_win.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
        
        # Setup ListStore to contain images and description
        model={}
        DEFAULT_IMAGE_WIDTH=230
        view={}
        frame={}
        frames = gtk.VBox()
        for key in desc:
            model[key] = (gtk.ListStore(gtk.gdk.Pixbuf, str))
            for im, dsc in zip(images[key],desc[key]):
                try:
                    pixbuf = gtk.gdk.pixbuf_new_from_file("images/"+im)
                    pix_w = pixbuf.get_width()
                    pix_h = pixbuf.get_height()
                    new_h = (pix_h * DEFAULT_IMAGE_WIDTH) / pix_w # Calculate the scaled height before resizing image
                    scaled_pix = pixbuf.scale_simple(DEFAULT_IMAGE_WIDTH, new_h, gtk.gdk.INTERP_TILES)
                    model[key].append((scaled_pix, dsc))
                    i+=1
                except:
                    pass
            # Setup GtkIconView
            view[key] = gtk.IconView(model[key]) # Pass the model stored in a ListStore to the GtkIconView
            view[key].set_pixbuf_column(0)
            view[key].set_text_column(1)
            view[key].set_selection_mode(gtk.SELECTION_SINGLE)
            view[key].set_columns(0)
            view[key].set_item_width(265)
            # connect signals to IconView
            view[key].connect('selection-changed', self.on_selection_changed, key)
            view[key].connect('item-activated', self.on_item_activated, key)
        
            frame[key] = gtk.Frame(module_titles[key])
            frame[key].set_border_width(10)
            frame[key].add(view[key])
                
        eb = gtk.EventBox()
        eb.modify_bg(gtk.STATE_NORMAL, gtk.gdk.color_parse("white"))
        eb.add(frames)
        
        self.list_recent=[]
        self.read_recent(self.list_recent)
        # Recently used modules:
        list_recent=self.list_recent
        model_recent = gtk.ListStore(gtk.gdk.Pixbuf, str)
        images_recent=[]
        images_recent.append(images[list_recent[0][0]][int(list_recent[0][1])])
        images_recent.append(images[list_recent[1][0]][int(list_recent[1][1])])
        images_recent.append(images[list_recent[2][0]][int(list_recent[2][1])])
        desc_recent=[]
        desc_recent.append(module_titles[list_recent[0][0]] + ": " + desc[list_recent[0][0]][int(list_recent[0][1])])
        desc_recent.append(module_titles[list_recent[1][0]] + ": " + desc[list_recent[1][0]][int(list_recent[1][1])])
        desc_recent.append(module_titles[list_recent[2][0]] + ": " + desc[list_recent[2][0]][int(list_recent[2][1])])
        for im, dsc in zip(images_recent,desc_recent):
            try:
                pixbuf = gtk.gdk.pixbuf_new_from_file("images/"+im)
                pix_w = pixbuf.get_width()
                pix_h = pixbuf.get_height()
                new_h = (pix_h * DEFAULT_IMAGE_WIDTH) / pix_w # Calculate the scaled height before resizing image
                scaled_pix = pixbuf.scale_simple(DEFAULT_IMAGE_WIDTH, new_h, gtk.gdk.INTERP_TILES)
                model_recent.append((scaled_pix, dsc))
            except:
                pass
        # Setup GtkIconView
        view_recent = gtk.IconView(model_recent) # Pass the model stored in a ListStore to the GtkIconView
        view_recent.set_pixbuf_column(0)
        view_recent.set_text_column(1)
        view_recent.set_selection_mode(gtk.SELECTION_SINGLE)
        view_recent.set_columns(0)
        view_recent.set_item_width(265)
        # connect signals to IconView
        view_recent.connect('selection-changed', self.on_selection_changed_recent, list_recent)
        view_recent.connect('item-activated', self.on_item_activated_recent, list_recent)

        frame_recent = gtk.Frame("Recently Used")
        frame_recent.set_border_width(15)
        frame_recent.add(view_recent)

        # Pack objects
        frames.pack_start(frame_recent,True,True)
        hsep = gtk.HSeparator()
        frames.pack_start(hsep,False,False,5)
        for key in desc:
            frames.pack_start(frame[key],True,True)
        
        scrolled_win.add_with_viewport(eb)
        scrolled_win.set_size_request(900,610)
        scrolled_win.set_border_width(0)
        
        vbox.pack_start(scrolled_win)

        # Add TextView to show info about modules
        self.text = gtk.TextView()
        self.text.set_editable(False)
        self.text.set_left_margin(10)
        self.text.set_right_margin(10)
        self.text.set_pixels_above_lines(5)
        self.text.set_wrap_mode(gtk.WRAP_WORD)		# wrap words
        scroll_text = gtk.ScrolledWindow()		# into scrollable env
        scroll_text.set_shadow_type(gtk.SHADOW_ETCHED_IN)
        scroll_text.set_policy(gtk.POLICY_AUTOMATIC,gtk.POLICY_AUTOMATIC)
        scroll_text.add(self.text)
        scroll_text.set_border_width(10)
        
        # Buttons: help->[guide,credits], launch->[basic,advanced], options?
        #i_help = gtk.Image()
        #i_help.set_from_stock(gtk.STOCK_HELP, gtk.ICON_SIZE_LARGE_TOOLBAR)
        #help_button = gtk.Button()
        #help_button.add(i_help)
        #help_button.set_tooltip_text("Help");
        
        # Buttons: help->[guide,credits], launch->[basic,advanced], options?
        i_info = gtk.Image()
        i_info.set_from_stock(gtk.STOCK_INFO, gtk.ICON_SIZE_LARGE_TOOLBAR)
        info_button = gtk.LinkButton("http://malewick.web.cern.ch/malewick/dosematic/TRS405_scr.pdf")
        info_button.add(i_info)
        info_button.set_tooltip_text("Information");
        
        i_index = gtk.Image()
        i_index.set_from_stock(gtk.STOCK_INDEX, gtk.ICON_SIZE_LARGE_TOOLBAR)
        index_button = gtk.LinkButton("http://malewick.web.cern.ch/malewick/dosematic/uncertainty.pdf")
        index_button.add(i_index)
        index_button.set_tooltip_text("Handbook and Documentation");
        
        # pack buttons and scrollable text view together
        buttons_vbox = gtk.VBox()
        hruler = gtk.HSeparator()
        #buttons_vbox.pack_start(hruler, False, False, 20)
        buttons_vbox.pack_start(info_button, False, False, 5)
        #buttons_vbox.pack_start(help_button, False, False, 5)
        buttons_vbox.pack_start(index_button, False, False, 5)

        hbox = gtk.HBox()
        hbox.pack_start(scroll_text, True, True, 0)
        hbox.pack_start(buttons_vbox, False, False, 5)
        
        vbox.pack_start(hbox,True,True,0)
        self.send_to_textview(general_description)
        self.module_chosen=False
        
    def send_to_textview(self,txt):
        """Print text to the TextView"""
        textbuffer = self.text.get_buffer()
        textbuffer.set_text(txt)
        self.text.set_buffer(textbuffer)
        
    def on_selection_changed(self, icon_view, key):
        if icon_view.get_selected_items():
            mode_num = icon_view.get_selected_items()[0][0]
            self.send_to_textview(full_description[key][mode_num])
        
    def on_item_activated(self, icon_view, path, key):
        mode_num = icon_view.get_selected_items()[0][0]
        self.send_to_textview('selected: '+full_description[key][mode_num])
        self.list_recent.append([key,mode_num])
        self.write_recent(self.list_recent)
        self.open_module(key,mode_num)
        
    def on_selection_changed_recent(self, icon_view, listr):
        if icon_view.get_selected_items():
            mode_num = icon_view.get_selected_items()[0][0]
            self.send_to_textview(full_description[listr[mode_num][0]][int(listr[mode_num][1])])
        
    def on_item_activated_recent(self, icon_view, path, listr):
        mode_num = icon_view.get_selected_items()[0][0]
        self.send_to_textview('selected: '+full_description[listr[mode_num][0]][int(listr[mode_num][1])])
        self.list_recent.append([listr[mode_num][0],listr[mode_num][1]])
        self.write_recent(self.list_recent)
        
    def write_recent(self, list_recent) :
        writer = csv.writer(open("recent.csv", 'w'), delimiter=',')
        if len(list_recent) >= 3:
            for row in list_recent[-3:]:
                writer.writerow(row)
        # if something went wrong have some dummy values
        else:
            self.write_recent([("dic",0),("foci",0),("pccf",1)])
        
    def read_recent(self, list_recent) :
        reader = csv.reader(open("data/recent.csv", 'rb'), delimiter=',')
        for row in reader:
            list_recent.append(row)
            
    def open_module(self, key, num):
        """something's fucked up with elifs"""
        print key, desc[key][num]
        if "Acute" in desc[key][num]:
            print "here"
            manager = Acute.UserInterface(key, mode[key], labels[key])
            manager.show_all()
        elif "Partial" in desc[key][num]:
            print "there"
            manager = Partial.UserInterface(key, labels[key])
            manager.show_all()
        elif "Protracted" in desc[key][num]:
            print "where eagles dare"
            manager = Protracted.UserInterface(key, labels[key])
            manager.show_all()
        elif "Function Comparison" in desc[key][num]:
            print "and where they don't"
            manager = FunctionComparison.UserInterface()
            manager.show_all()



#______________MAIN______________#

m = SplashScreen.pngtranswin()
m.show_window()

mw = MainWindow()
mw.show_all()
gtk.main()



