import os
import logging.config

def configure_logging(logfile_path:str = None, console:bool = True):
    """Configure the loggers of the package.

    Parameters
    ----------
    logfile_path : str
        Path to the logfile. Parent directories are creadted if necessary.
        Value ``None`` can be used to disable the output to a logfile.
    console : bool
        Whether to print log messages to console.
    """
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'timeseries_qmc_formatter': {
                'format': '[%(levelname)s:%(asctime)s] %(message)s'
            },
        },
        'handlers': { },
        'loggers': {
            'timeseries_qmc': {
                'handlers': [],
                'level': 'INFO',
                'propagate': True
            }
        }
    }
    
    if(console):
        console_handler = {
            'class': 'logging.StreamHandler',
            'formatter': 'timeseries_qmc_formatter',
        }        
        LOGGING_CONFIG['handlers']['timeseries_qmc_console'] = console_handler
        LOGGING_CONFIG['loggers']['timeseries_qmc']['handlers'].append('timeseries_qmc_console')        


    if logfile_path:
        # Create parent directories if necessary
        dir = os.path.dirname(logfile_path)
        if(dir):
            os.makedirs(dir, exist_ok=True)
        file_handler = {
                'class': 'logging.FileHandler',
                'formatter': 'timeseries_qmc_formatter',
                'filename': logfile_path
            }
        LOGGING_CONFIG['handlers']['timeseries_qmc_file'] = file_handler

        LOGGING_CONFIG['loggers']['timeseries_qmc']['handlers'].append('timeseries_qmc_file')

    logging.config.dictConfig(LOGGING_CONFIG)