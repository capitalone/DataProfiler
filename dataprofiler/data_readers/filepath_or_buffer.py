from io import open, StringIO, BytesIO

from . import data_utils


class FileOrBufferHandler():
    """
    FileOrBufferHandler class to read a filepath or buffer in and always return a readable buffer
    """

    def __init__(self, io_obj, open_method='r', encoding=None, seek_offset=None, seek_whence=0):
        """
        Context manager class used for inputing a file or buffer and returning a structure
        that is always a buffer.

        :param io_obj: path to the file being loaded or buffer
        :type io_obj: string, StringIO, or BytesIO 
        :param open_method: value describes the mode the file is opened in
        :type open_method: string
        :param seek_offset: offset from start of the stream
        :type seek_offset: int
        :return: TextIOBase or BufferedIOBase class/subclass
        """
        self._io_obj = io_obj
        self.open_method = open_method
        self.seek_offset = seek_offset
        self.seek_whence = seek_whence
        self._encoding = encoding

    def __enter__(self):
        if isinstance(self._io_obj, str):
            self._io_obj = open(
                self._io_obj, self.open_method, encoding=self._encoding)

        elif not data_utils.is_stream_buffer(self._io_obj):
            # Raise AttributeError if attribute value not found.
            raise AttributeError(f'Type {type(self._io_obj)} is invalid. \
                io_obj must be a string or StringIO/BytesIO object')

        if self.seek_offset is not None:
            self._io_obj.seek(self.seek_offset, self.seek_whence)

        return self._io_obj

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if isinstance(self._io_obj, (StringIO, BytesIO)):
            self._io_obj.seek(0)
        else:
            self._io_obj.close()
