from io import open, StringIO, BytesIO, TextIOWrapper

from . import data_utils


class FileOrBufferHandler:
    """
    FileOrBufferHandler class to read a filepath or buffer in and always
    return a readable buffer.
    """

    def __init__(self, filepath_or_buffer, open_method='r', encoding=None,
                 seek_offset=None, seek_whence=0):
        """
        Context manager class used for inputting a file or buffer and returning
        a structure that is always a buffer.

        :param filepath_or_buffer: path to the file being loaded or buffer
        :type filepath_or_buffer: Union[str, StringIO, BytesIO]
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
        self._encoding = encoding
        self.original_type = type(filepath_or_buffer)
        self._is_wrapped = False

    def __enter__(self):
        if isinstance(self._filepath_or_buffer, str):
            self._filepath_or_buffer = open(
                self._filepath_or_buffer, self.open_method,
                encoding=self._encoding)

        elif isinstance(self._filepath_or_buffer, BytesIO) \
                and self.open_method == 'r':
            self._filepath_or_buffer = \
                TextIOWrapper(self._filepath_or_buffer, encoding=self._encoding)
            self._is_wrapped = True

        elif not data_utils.is_stream_buffer(self._filepath_or_buffer):
            # Raise AttributeError if attribute value not found.
            raise AttributeError(f'Type {type(self._filepath_or_buffer)} is '
                                 f'invalid. filepath_or_buffer must be a '
                                 f'string or StringIO/BytesIO object')

        if self.seek_offset is not None:
            self._filepath_or_buffer.seek(self.seek_offset, self.seek_whence)

        return self._filepath_or_buffer

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # Need to detach buffer if wrapped (i.e. BytesIO opened with 'r')
        if self._is_wrapped:
            wrapper = self._filepath_or_buffer
            self._filepath_or_buffer = wrapper.buffer
            wrapper.detach()

        if isinstance(self._filepath_or_buffer, (StringIO, BytesIO)):
            self._filepath_or_buffer.seek(0)
        else:
            self._filepath_or_buffer.close()
