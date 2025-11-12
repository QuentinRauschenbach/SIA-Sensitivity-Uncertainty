import logging

def set_logging_level_user(logger):
    logLevel = input ( 'What log level you  want to use?')
    if logLevel.upper() == 'CRITICAL':
        logger.setLevel(logging.CRITICAL)
    elif logLevel.upper() == 'ERROR':
        logger.setLevel(logging.ERROR)
    elif logLevel.upper() == 'WARNING':
        logger.setLevel(logging.WARNING)
    elif logLevel.upper() == 'INFO':
        logger.setLevel(logging.INFO)
    elif logLevel.upper() == 'DEBUG':
        logger.setLevel(logging.DEBUG)
    else:
        print("Wrong Level!")
        exit()
    return logger

def set_logging_level_str(logger, logLevel):
    if logLevel.upper() == 'CRITICAL':
        logger.setLevel(logging.CRITICAL)
    elif logLevel.upper() == 'ERROR':
        logger.setLevel(logging.ERROR)
    elif logLevel.upper() == 'WARNING':
        logger.setLevel(logging.WARNING)
    elif logLevel.upper() == 'INFO':
        logger.setLevel(logging.INFO)
    elif logLevel.upper() == 'DEBUG':
        logger.setLevel(logging.DEBUG)
    else:
        print("Wrong Level!")
        exit()
    if not logger.handlers:    
        console_handler = logging.StreamHandler()
        logger.addHandler(console_handler)
    return logger

def build_terminal_logger(logLevel, logger_name):
    logger = logging.getLogger(logger_name)
    logger = set_logging_level_str(logger, logLevel)
    if not logger.handlers:    
        console_handler = logging.StreamHandler()
        logger.addHandler(console_handler)
    return logger

def ask_if_paths_correct(logger, sicpath, areafile, maskfile, outpath, years):
    logger.info("Running with the following setup: \n"
                        + f"  sicpath:  {sicpath}\n"
                        + f"  areafile: {areafile}\n"
                        + f"  maskfile: {maskfile}\n"
                        + f"  outpath:  {outpath}\n"
                        + f"  years:    {years}")

    if logger.getEffectiveLevel() <= logging.INFO:
        print(f"Is everything correct? (y/n)")
        INPUT = str(input())
        if INPUT in ["y", "Y", "yes", "YES", "Yes"]:
            return True
        else:
            return False
    else:
        return True