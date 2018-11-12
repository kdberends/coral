""" Utilities, convenience functions """

# =============================================================================
# Imports
# =============================================================================

import os
import logging

# =============================================================================
# Definitions
# =============================================================================


def get_logger(outputpath=os.getcwd(), logfile='log.log', overwrite=False, loggerlevel='info', name=__name__):
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

