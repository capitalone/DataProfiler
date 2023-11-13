import os
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

import pandas as pd
import requests

from dataprofiler.data_readers.data import Data
from dataprofiler.data_readers.text_data import TextData

test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestDataReadingWriting(unittest.TestCase):
    def test_read_data(self):
        """Ensure error logs for trying to save empty data."""
        data_types = ["csv", "json", "parquet", "text"]
        none_types = [pd.DataFrame(), pd.DataFrame(), "", ""]
        for data_type, none_type in zip(data_types, none_types):
            Data(data=none_type, data_type=data_type)


class TestDataReadFromURL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_dir = os.path.join(test_root_path, "data")
        cls.input_file_names = [
            dict(
                path=os.path.join(test_dir, "csv/diamonds.csv"),
                count=53940,
                delimiter=",",
                has_header=[0],
                num_columns=10,
                encoding="utf-8",
                data_type="csv",
            ),
            dict(
                path=os.path.join(test_dir, "avro/users.avro"),
                count=4,
                data_type="avro",
            ),
            dict(
                path=os.path.join(test_dir, "json/iris-utf-16.json"),
                encoding="utf-16",
                count=150,
                data_type="json",
            ),
            dict(
                path=os.path.join(test_dir, "parquet/iris.parq"),
                count=150,
                data_type="parquet",
            ),
            dict(
                path=os.path.join(test_dir, "txt/code.txt"), count=150, data_type="text"
            ),
        ]

    @mock.patch("requests.get")
    def test_read_from_url(self, mock_request_get):
        """Ensure error logs for trying to save empty data."""

        def chunk_file(filepath, c_size):
            with open(filepath, "rb") as fp:
                bytes = fp.read(c_size)
                while bytes != b"":
                    yield bytes
                    bytes = fp.read(c_size)

        for input_file in self.input_file_names:
            mock_request_get.return_value.__enter__.return_value.iter_content.side_effect = lambda chunk_size: chunk_file(
                input_file["path"], chunk_size
            )

            # stub URL, the line above replaces the content requests.get will see
            data_obj = Data("https://test.com")
            self.assertEqual(data_obj.data_type, input_file["data_type"])

    @mock.patch("requests.get")
    def test_read_url_content_overflow(self, mock_request_get):
        # assumed chunk size
        c_size = 8192
        max_allows_file_size = 1024**3  # 1GB

        try:
            # mock the iter_content to return just under 1GB so no error raises
            mock_request_get.return_value.__enter__.return_value.iter_content.return_value = [
                b"test"
            ] * (
                int(max_allows_file_size) // c_size
            )

            # stub URL, the line above replaces the content requests.get will see
            data_obj = Data("https://test.com")

        except ValueError:
            self.fail("URL string unexpected overflow error.")

        # mock the iter_content to return up to 1GB + so error raises
        mock_request_get.return_value.__enter__.return_value.iter_content.return_value = [
            b"test"
        ] * (
            int(max_allows_file_size) // c_size + 1
        )

        with self.assertRaisesRegex(
            ValueError, "The downloaded file from the url may not be larger than 1GB"
        ):

            # stub URL, mock_request_get  replaces the content requests.get will see
            data_obj = Data("https://test.com")

    @mock.patch("requests.get")
    def test_read_url_header_overflow(self, mock_request_get):
        # assumed chunk size
        c_size = 8192
        max_allows_file_size = 1024**3  # 1GB

        # set valid content length size
        content_length = 5000
        mock_request_get.return_value.__enter__.return_value.headers = {
            "Content-length": content_length
        }

        try:
            # mock the iter_content to return just under 1GB so no error raises
            mock_request_get.return_value.__enter__.return_value.iter_content.return_value = [
                b"test"
            ] * (
                int(content_length) // c_size
            )

            # stub URL, the line above replaces the content requests.get will see
            data_obj = Data("https://test.com")

        except ValueError:
            self.fail("URL string unexpected overflow error.")

        # make content length an invalid size
        content_length = max_allows_file_size + 1
        mock_request_get.return_value.__enter__.return_value.headers = {
            "Content-length": content_length
        }

        with self.assertRaisesRegex(
            ValueError, "The downloaded file from the url may not be larger than 1GB"
        ):
            # stub URL, mock_request_get replaces the content requests.get will see
            data_obj = Data("https://test.com")

    @mock.patch("requests.get")
    def test_read_url_verify_ssl(self, mock_request_get):
        mock_request_get.side_effect = requests.exceptions.SSLError()

        with self.assertRaises(
            RuntimeError,
            msg="The URL given has an untrusted "
            "SSL certificate. Although highly discouraged, "
            "you can proceed with reading the data by setting"
            " 'verify_url' to False in options (i.e. options"
            "=dict(verify_url=False)).",
        ):
            data_obj = Data("https://test.com")

    @patch("boto3.client")
    @patch(
        "os.environ",
        {
            "AWS_ACCESS_KEY_ID": "<YOUR_ACCESS_KEY>",
            "AWS_SECRET_ACCESS_KEY": "<YOUR_SECRET_KEY>",
        },
    )
    def test_read_s3_uri(self, mock_boto3_client):
        region_name = "us-east-1"

        # Create a custom mock response
        custom_response = {"Body": MagicMock()}
        custom_response["Body"].read.return_value = b"Test S3 content"

        # Mock the behavior of the S3 client to return the custom response
        mock_boto3_client.return_value.get_object.return_value = custom_response
        data_obj = Data("s3a://my-bucket/my_file.txt")

        self.assertEqual(type(data_obj), TextData)

        mock_boto3_client.assert_called_with(
            "s3",
            aws_access_key_id="<YOUR_ACCESS_KEY>",
            aws_secret_access_key="<YOUR_SECRET_KEY>",
            aws_session_token=None,
            region_name=region_name,
        )


if __name__ == "__main__":
    unittest.main()
