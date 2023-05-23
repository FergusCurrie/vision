import logging 


def get_logger():
    # Logging 
    logging.basicConfig(filename='example.log', level=logging.DEBUG)
    logger = logging.getLogger('deep_cluster_logger')

    # check if logger already has a handler
    if not logger.handlers:
        my_handler = logging.FileHandler('deep_cluster_logger.log')
        my_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        my_handler.setFormatter(formatter)
        logger.addHandler(my_handler)
    return logger

def log_start_end(logger):
    def log_start_end_decorator(function):
        def wrapper(*args, **kwargs):
            logger.debug(f'STARTED: {function.__name__}')
            func = function(*args, **kwargs)
            logger.debug(f'FINISHED: {function.__name__}')
            return func
        return wrapper
    return log_start_end_decorator
