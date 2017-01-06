class PygpException(Exception):
    def __init__(self, message):
        self.message = '[ pygp ] ' + message
