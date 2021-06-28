from io import open, StringIO, BytesIO

from . import data_utils

class FileOrBufferHandler():
    def __init__(self, stream_or_buffer, open_method='r', options=dict()):
        self._stream_or_buffer = stream_or_buffer
        self.open_method = open_method
        self.options = options

    def __enter__(self):
        if (data_utils.is_stream_buffer(self._stream_or_buffer)):
            print(self.options)
            if 'seek' in self.options:
                self._stream_or_buffer.seek(self.options['seek'],0)
                return self._stream_or_buffer
            else:
                return self._stream_or_buffer
        else:
            self._file = open(self._stream_or_buffer, self.open_method)
            return self._file


    def __exit__(self, exc_type, exc_value, exc_traceback):
        if (data_utils.is_stream_buffer(self._stream_or_buffer)):
            if 'seek' in self.options:
                self._stream_or_buffer.seek(self.options['seek'],0)
            else:
                self._stream_or_buffer.seek(0,0)
        else:
            self._file.close()
