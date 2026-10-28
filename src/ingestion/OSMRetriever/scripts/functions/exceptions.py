from fiona.errors import DriverError
from pyproj.exceptions import CRSError

class BBOXError(Exception):
    '''Raised when the input bounding box values are not correct.'''

    def __init__(self, value):
        super().__init__(f'Input Bounding Box string is incorrect or contains invalid values: {value}')
        self.value=value