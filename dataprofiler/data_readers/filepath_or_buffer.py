"""Contains functions and classes for handling filepaths and buffers."""
from io import BytesIO, StringIO, TextIOWrapper, open
from typing import IO, Any, Optional, Type, Union, cast

from typing_extensions import TypeGuard


def is_stream_buffer(filepath_or_buffer: Any) -> TypeGuard[Union[StringIO, BytesIO]]:
    """
    Determine whether a given argument is a filepath or buffer.

    :param filepath_or_buffer: path to the file or buffer
    :type filepath_or_buffer: str
    :return: true if string is a buffer or false if string is a filepath
    :rtype: boolean
    """
    if isinstance(filepath_or_buffer, (StringIO, BytesIO)):
        return True
    return False


class FileOrBufferHandler:
    """
    FileOrBufferHandler class to read a filepath or buffer in.

    Always returns a readable buffer.
    """

    def __init__(
        self,
        filepath_or_buffer: Union[str, StringIO, BytesIO],
        open_method: str = "r",
        encoding: Optional[str] = None,
        seek_offset: Optional[int] = None,
        seek_whence: int = 0,
    ) -> None:
        """
        Initialize Context manager class.

        Used for inputting a file or buffer and returning
        a structure that is always a buffer.

        :param filepath_or_buffer: path to the file being loaded or buffer
        :type filepath_or_buffer: Union[str, StringIO, BytesIO]
        :param open_method: value describes the mode the file is opened in
        :type open_method: string
        :param seek_offset: offset from start of the stream
        :type seek_offset: int
        :return: TextIOBase or BufferedIOBase class/subclass
        """
        self._filepath_or_buffer: Union[str, StringIO, BytesIO, IO] = filepath_or_buffer
        self.open_method: str = open_method
        self.seek_offset: Optional[int] = seek_offset
        self.seek_whence: int = seek_whence
        self._encoding: Optional[str] = encoding
        self.original_type: Union[
            Type[str], Type[StringIO], Type[BytesIO], Type[IO]
        ] = type(filepath_or_buffer)
        self._is_wrapped: bool = False

    def __enter__(self) -> IO:
        """Open resources."""
        if isinstance(self._filepath_or_buffer, str):
            self._filepath_or_buffer = open(
                self._filepath_or_buffer, self.open_method, encoding=self._encoding
            )

        elif isinstance(self._filepath_or_buffer, BytesIO) and self.open_method == "r":
            self._filepath_or_buffer = TextIOWrapper(
                self._filepath_or_buffer, encoding=self._encoding
            )
            self._is_wrapped = True

        elif not is_stream_buffer(self._filepath_or_buffer):
            # Raise AttributeError if attribute value not found.
            raise AttributeError(
                f"Type {type(self._filepath_or_buffer)} is "
                f"invalid. filepath_or_buffer must be a "
                f"string or StringIO/BytesIO object"
            )

        if self.seek_offset is not None:
            self._filepath_or_buffer.seek(self.seek_offset, self.seek_whence)

        return self._filepath_or_buffer

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        """Release resources."""
        # Need to detach buffer if wrapped (i.e. BytesIO opened with 'r')
        if self._is_wrapped:
            self._filepath_or_buffer = cast(
                TextIOWrapper, self._filepath_or_buffer
            )  # guaranteed by self._is_wrapped
            wrapper = self._filepath_or_buffer
            self._filepath_or_buffer = wrapper.buffer
            wrapper.detach()

        if isinstance(self._filepath_or_buffer, (StringIO, BytesIO)):
            self._filepath_or_buffer.seek(0)
        else:
            self._filepath_or_buffer = cast(
                IO, self._filepath_or_buffer
            )  # can't be str due to conversion in __enter__
            self._filepath_or_buffer.close()
