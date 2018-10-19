"""
Utilities, convenience functions
"""

# =============================================================================
# Imports
# =============================================================================

import os
import sys
import logging
import configparser
import seaborn as sns
import calendar
from datetime import datetime
if (sys.version_info > (3, 0)):
    from io import StringIO
else:
    from cStringIO import StringIO

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
flatui_r = list(reversed(flatui))

# Configuration file parsing types
configtypes = """[parameters]
datetimefmt=str
geometrytype=int
g=float
tstart=datetime
tstop=datetime
dt=int
dx=int
frictionmodel=str
growthmodel=str
blockagemodel=str
"""

eventypes= """[event]
eventtype=str
tstart=datetime
minchainage=float
maxchainage=float
reduce_to=float
maximum_blockage=float
triggered=bool
name=str
"""

datetimeformat = '%d/%m/%Y'
# =============================================================================
# Definitions
# =============================================================================


def get_logger(outputpath=os.getcwd(), logfile='log.log',
               overwrite=False, loggerlevel='info', name=__name__):
    """
    Returns a logger object which:
    -
    """
    filepath = os.path.join(outputpath, logfile)
    if overwrite and os.path.isfile(filepath):
        mode = 'w'
    else:
        mode = 'a'

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # the filehandler prints every level to file
        filehandler = logging.FileHandler(filepath, mode=mode)
        filehandler.setLevel(logging.DEBUG)

        # the stream prints info and above (error, warning, critical) to console
        streamhandler = logging.StreamHandler()

        if loggerlevel.lower() == 'debug':
            streamhandler.setLevel(logging.DEBUG)
        elif loggerlevel.lower() == 'info':
            streamhandler.setLevel(logging.INFO)
        elif loggerlevel.lower() == 'notset':
            streamhandler.setLevel(logging.NOTSET)
        elif loggerlevel.lower() == 'silent':
            streamhandler.setLevel(60)
        else:
            streamhandler.setLevel(logging.INFO)


        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                                      "%Y-%m-%d %H:%M:%S")

        filehandler.setFormatter(formatter)
        streamhandler.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(filehandler)
        logger.addHandler(streamhandler)

        logger.info('Start logging to {}'.format(filepath))

    return logger

