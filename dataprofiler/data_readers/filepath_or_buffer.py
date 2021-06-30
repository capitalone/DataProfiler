from io import open, StringIO, BytesIO


from . import data_utils

class FileOrBufferHandler():
    """
    FileOrBufferHandler class to read a filepath or buffer in and always return a readable buffer
    """
    def __init__(self, filepath_or_buffer, open_method='r', seek_offset=None, seek_whence=0):
        """
        Context manager class used for inputing a file or buffer and returning a structure
        that is always a buffer.

        :param filepath_or_buffer: path to the file being loaded or buffer
        :type filepath_or_buffer: string, StringIO, or BytesIO 
        :param open_method: value describes the mode the file is opened in
        :type open_method: string
        :param seek_offset: offset from start of the stream
        :type seek_offset: int
        :return: TextIOBase or BufferedIOBase class/subclass
        """
        self._filepath_or_buffer = filepath_or_buffer
        self.open_method = open_method
        self.seek_offset = seek_offset
        self.seek_whence = seek_whence
        self._file = None

    def __enter__(self):
        if not data_utils.is_stream_buffer(self._filepath_or_buffer):
            self._filepath_or_buffer = open(self._filepath_or_buffer, self.open_method)

        if self.seek_offset is not None:
            self._filepath_or_buffer.seek(self.seek_offset, self.seek_whence)
            return self._filepath_or_buffer
        else:
            return self._filepath_or_buffer


    def __exit__(self, exc_type, exc_value, exc_traceback):
        if (self._file is None):
            self._filepath_or_buffer.seek(0)
        else:
            self._file.close()
