#!/usr/bin/env python

from h5xplorer.h5xplorer import h5xplorer
import h5x_menu

app = h5xplorer(h5x_menu.context_menu,baseimport='baseimport.py',extended_selection=False)