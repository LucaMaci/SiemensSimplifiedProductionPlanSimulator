import logging
import sys
import time


class LoggingFormatter(logging.Formatter):
    def format(self, record):
        # the log level should be displayed in lower case
        record.levelname = record.levelname.lower()
        return super().format(record)

    def formatTime(self, record, datefmt=None):
        # format the time in similar fashion to the C++ log files
        converted_record = self.converter(record.created)
        timestamp_string = time.strftime('%Y-%m-%dT%H:%M:%S', converted_record)
        return '%s.%03dZ' % (timestamp_string, record.msecs)


loggers = {}


def setup_logger(name, level=logging.INFO):
    global loggers
    if loggers.get(name):
        return loggers.get(name)
    else:
        logger = logging.getLogger(name)
        # set the passed log level
        logger.setLevel(level)
        # create a handler that prints the logs to stdout
        handler_stdout = logging.StreamHandler(stream=sys.stdout)
        # create a handler that prints the logs to stderr
        # handler_stderr = logging.StreamHandler(stream=sys.stderr)
        # configure the formatting string of both handlers
        fmt = f'[%(asctime)s][{name}][%(levelname)s] %(message)s'
        handler_stdout.setFormatter(LoggingFormatter(fmt=fmt))
        # handler_stderr.setFormatter(LoggingFormatter(fmt=fmt))
        # add both handlers to the logger
        logger.addHandler(handler_stdout)
        # logger.addHandler(handler_stderr)
        loggers[name] = logger
        return logger
