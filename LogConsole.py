from datetime import datetime

import pygtk
pygtk.require('2.0')
import gtk
from gtk import gdk
import pango


class LogConsole():

    def __init__(self, context):
        self.text = gtk.TextView()                      # TEXT VIEW
        self.text.set_wrap_mode(gtk.WRAP_WORD)          # wrap words
        self.scrolled_window = gtk.ScrolledWindow()
        self.scrolled_window.add(self.text)
        self.scrolled_window.set_shadow_type(gtk.SHADOW_ETCHED_IN)
        self.scrolled_window.set_policy(gtk.POLICY_AUTOMATIC,gtk.POLICY_AUTOMATIC)

    def init_tags(self,buff):
        """Available text tags: http://pygtk.org/pygtk2tutorial/sec-TextTagsAndTextTagTables.html"""
        tag = buff.create_tag("normal",
                                foreground_gdk=gtk.gdk.Color(red=20000,green=20000,blue=20000),
                                background='white',
                                size_points=12.0)
        tag = buff.create_tag("fg_blue",
                                foreground_gdk=gtk.gdk.Color(red=10000,green=10000,blue=60000),
                                background='yellow',
                                size_points=24.0)
        tag = buff.create_tag("fg_gray",
                                foreground_gdk=gtk.gdk.Color(red=40000,green=40000,blue=40000))
        tag = buff.create_tag("bg_green",
                                background_gdk=gtk.gdk.Color(red=10000,green=60000,blue=10000),
                                size_points=10.0)
        tag = buff.create_tag("strikethrough",
                                strikethrough=True)
        tag = buff.create_tag("underline",
                                underline=pango.UNDERLINE_SINGLE)
        tag = buff.create_tag("centered",
                                justification=gtk.JUSTIFY_CENTER)
        tag = buff.create_tag("rtl_quote",
                                wrap_mode=gtk.WRAP_WORD,
                                direction=gtk.TEXT_DIR_RTL,
                                indent=30,
                                left_margin=20,
                                right_margin=20)
        tag = buff.create_tag("negative_indent",indent=-25)

    def log(self,txt):
        """log stuff to log console (TextView)"""
        timestamp = "[" + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "]:  "
        self.add_to_text_buffer(self.text.get_buffer(),timestamp,"fg_gray")
        self.add_to_text_buffer(self.text.get_buffer(),txt+"\n","normal")
        adj = self.scrolled_window.get_vadjustment()
        adj.set_value( adj.upper - adj.page_size )

    def add_to_text_buffer(self,buff,txt,tag):
        """add text to a given buffer, with a proper tag"""
        tagtable = buff.get_tag_table()
        if not tagtable.lookup("fg_blue"):
            self.init_tags(buff)
        end_iter = buff.get_end_iter()
        buff.insert_with_tags_by_name(end_iter, txt, tag)
