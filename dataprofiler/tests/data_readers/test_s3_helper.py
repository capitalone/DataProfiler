import unittest
from unittest.mock import patch

from dataprofiler import dp_logging
from dataprofiler.data_readers.data_utils import S3Helper


class TestS3Helper(unittest.TestCase):
    @patch("boto3.client")
    def test_create_s3_client_with_credentials(self, mock_boto3_client):
        aws_access_key_id = "<YOUR_ACCESS_KEY>"
        aws_secret_access_key = "<YOUR_SECRET_KEY>"
        region_name = "us-west-1"

        S3Helper.create_s3_client(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )

        mock_boto3_client.assert_called_with(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=None,
            region_name=region_name,
        )

    @patch("boto3.client")
    @patch(
        "os.environ",
        {
            "AWS_ACCESS_KEY_ID": "<YOUR_ACCESS_KEY>",
            "AWS_SECRET_ACCESS_KEY": "<YOUR_SECRET_KEY>",
        },
    )
    def test_create_s3_client_with_environment_variables(self, mock_boto3_client):
        region_name = "us-west-1"

        S3Helper.create_s3_client(region_name=region_name)

        mock_boto3_client.assert_called_with(
            "s3",
            aws_access_key_id="<YOUR_ACCESS_KEY>",
            aws_secret_access_key="<YOUR_SECRET_KEY>",
            aws_session_token=None,
            region_name=region_name,
        )

    @patch("boto3.client")
    @patch("os.environ", {"AWS_REGION": "us-west-1"})
    def test_create_s3_client_with_iam_role_and_region_from_environment_variable(
        self, mock_boto3_client
    ):
        S3Helper.create_s3_client()

        mock_boto3_client.assert_called_with(
            "s3",
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
            region_name="us-west-1",
        )

    @patch("boto3.client")
    @patch("botocore.exceptions.NoCredentialsError", Exception)
    def test_create_s3_client_with_iam_role_fallback_to_credentials(
        self, mock_boto3_client
    ):
        aws_access_key_id = "<YOUR_ACCESS_KEY>"
        aws_secret_access_key = "<YOUR_SECRET_KEY>"
        region_name = "us-west-1"

        S3Helper.create_s3_client(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )

        mock_boto3_client.assert_called_with(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=None,
            region_name=region_name,
        )

    @patch("boto3.client")
    def test_create_s3_client_with_iam_role(self, mock_boto3_client):
        # Simulate a scenario where IAM roles are available,
        # and no credentials are provided
        region_name = "us-west-1"

        S3Helper.create_s3_client(region_name=region_name)

        mock_boto3_client.assert_called_with(
            "s3",
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
            region_name=region_name,
        )

    def test_is_s3_uri_failure_logger_check(self):
        invalid_path = "invalid_path"

        logger = dp_logging.get_child_logger(__name__)

        with self.assertLogs(logger, level="DEBUG") as log_context:
            # Call the function with the invalid path
            is_s3 = S3Helper.is_s3_uri(invalid_path, logger)

            # Assert that the function returns False (invalid path)
            self.assertFalse(is_s3)

            # Assert that the log message is generated and logged
            self.assertIn(
                f"'{invalid_path}' is not a valid S3 URI", log_context.output[0]
            )


if __name__ == "__main__":
    unittest.main()
