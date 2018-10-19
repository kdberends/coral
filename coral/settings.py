#
# This file: Preferences for plotting
# 
# Author: Koen Berends
# Contact: k.d.berends@utwente.nl 
# Copyright (c) 2017 University of Twente
#
# Permission is hereby granted, free of charge, to any person obtaining a 
# copy of this software and associated documentation files (the "Software"), 
# to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the Software 
# is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all 
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

# =============================================================================
# Imports
# =============================================================================

import seaborn as sns
import matplotlib as mpl 
# =============================================================================
# Functions
# =============================================================================

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
flatui_r = list(reversed(flatui))

def set_plotstyle(style='', palette=flatui_r, scale=1.2):
    """
    Arguments:
    
    style: 
          'paper': for use in journal
          'digital': for use on-screen

    palette:
          name of colorpalette, e.g.:
          - flatui
          - paired
          - deep, muted, pastel, bright, dark, colorblind
    """
    sns.set_context("paper")

    if style.lower() == 'paper':    
        # Set the font to be serif, rather than sans
        sns.set(font='serif',
                palette=palette,
                font_scale=2,
                style='whitegrid')
    elif style.lower() == 'digital':
        sns.set(font='serif',
                palette=palette,
                font_scale=1.2,
                style='whitegrid')
    else:
        sns.set(font='serif',
                palette=palette,
                font_scale=scale,
                style='whitegrid')


def two_axes_style(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['left'].set_linewidth(2)
    ax.spines['left'].set_edgecolor('k')
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['bottom'].set_edgecolor('k')
    ax.spines['bottom'].set_linewidth(2)
    ax.grid(False)


def gridbox_style(ax, gridcolor='0.5', linewidths=[1, 0.5]):
    ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.grid(b=True, which='major', color=gridcolor, linewidth=linewidths[0])
    ax.grid(b=True, which='minor', color=gridcolor, linewidth=linewidths[1])
    for spine in ax.spines:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('k')
        ax.spines[spine].set_linewidth(1)

    ax.grid(True)
