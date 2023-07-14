import os
import unittest
from itertools import islice

from dataprofiler.data_readers import data_utils

test_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestDataReadingWriting(unittest.TestCase):
    def test_file_UTF_encoding_detection(self):
        """
        Tests the ability for `data_utils.detect_file_encoding` to detect the
        encoding of text files. This test is specifically for UTF-8, UTF-16,
        and UTF-32 of csv or JSON.
        :return:
        """
        test_dir = os.path.join(test_root_path, "data")
        input_files = [
            dict(path=os.path.join(test_dir, "csv/iris-utf-8.csv"), encoding="utf-8"),
            dict(path=os.path.join(test_dir, "csv/iris-utf-16.csv"), encoding="utf-16"),
            dict(path=os.path.join(test_dir, "csv/iris-utf-32.csv"), encoding="utf-32"),
            dict(path=os.path.join(test_dir, "json/iris-utf-8.json"), encoding="utf-8"),
            dict(
                path=os.path.join(test_dir, "json/iris-utf-16.json"), encoding="utf-16"
            ),
            dict(
                path=os.path.join(test_dir, "json/iris-utf-32.json"), encoding="utf-32"
            ),
            dict(path=os.path.join(test_dir, "txt/utf8.txt"), encoding="utf-8"),
            dict(path=os.path.join(test_dir, "csv/zomato.csv"), encoding="ISO-8859-1"),
            dict(path=os.path.join(test_dir, "csv/reddit_wsb.csv"), encoding="utf-8"),
        ]

        get_match_acc = lambda s, s2: sum([s[i] == s2[i] for i in range(len(s))]) / len(
            s
        )

        for input_file in input_files:
            detected_encoding = data_utils.detect_file_encoding(
                file_path=input_file["path"]
            )
            with open(input_file["path"], "rb") as infile:
                # Read a max of 1 MB of data
                content = infile.read(1024 * 1024)
                # Assert at least 99.9% of the content was correctly decoded
                match_acc = get_match_acc(
                    content.decode(input_file["encoding"]),
                    content.decode(detected_encoding),
                )
                self.assertGreaterEqual(match_acc, 0.999)

    def test_nth_loc_detection(self):
        """
        Tests the ability for the `data_utils.find_nth_location` to detect the
        nth index of a search_query in a string.
        """
        # Input args: string, query, n
        # Expected results: index, occurrences
        test_queries = [
            dict(string="This is a test.", query=".", n=1, index=14, occurrences=1),
            dict(string="This is a test\n", query="\n", n=1, index=14, occurrences=1),
            dict(
                string="This is a test\nThis is a second test\n",
                query="\n",
                n=0,
                index=-1,
                occurrences=0,
            ),
            dict(
                string="This is a test\nThis is a second test\n",
                query="\n",
                n=2,
                index=36,
                occurrences=2,
            ),
            dict(string="t", query="t", n=1, index=0, occurrences=1),
            dict(string="s", query="t", n=1, index=1, occurrences=0),
            dict(
                string="This is a test\nThis is a second test\n\n",
                query="\n",
                n=3,
                index=37,
                occurrences=3,
                ignore_consecutive=False,
            ),
            dict(
                string="This is a test\nThis is a second test\n\nTest\n",
                query="\n",
                n=5,
                index=43,
                occurrences=4,
                ignore_consecutive=False,
            ),
            dict(
                string="This is a test\n\nThis is a second test\n\nTest\n",
                query="\n",
                n=2,
                index=37,
                occurrences=2,
                ignore_consecutive=True,
            ),
            dict(
                string="This is a test\n\nThis is a second test\n\nTest\n",
                query="\n",
                n=4,
                index=38,
                occurrences=4,
                ignore_consecutive=False,
            ),
            dict(string="", query="\n", n=3, index=-1, occurrences=0),
        ]

        for q in test_queries:
            ignore_consecutive = q.get("ignore_consecutive", True)
            self.assertEqual(
                (q["index"], q["occurrences"]),
                data_utils.find_nth_loc(
                    q["string"], q["query"], q["n"], ignore_consecutive
                ),
            )

    def test_load_as_str_from_file(self):
        """
        Tests if the load_as_str_file function can appropriately load files
        thresholded by bytes or max lines.
        """

        test_dir = os.path.join(test_root_path, "data")

        iris_32bit_filepath = os.path.join(test_dir, "csv/iris-utf-32.csv")
        iris_32bit_first_5 = ""
        with open(iris_32bit_filepath, encoding="utf-32") as f:
            iris_32bit_first_5 = "".join(list(islice(f, 5)))[:-1]  # remove final \n

        input_files = [
            dict(
                path=os.path.join(test_dir, "csv/empty.csv"),
                encoding="utf-8",
                max_lines=5,
                max_bytes=65536,
                chunk_size_bytes=1024,
                results="",
            ),
            dict(
                path=os.path.join(test_dir, "csv/iris-utf-8.csv"),
                encoding="utf-8",
                max_lines=5,
                max_bytes=65536,
                chunk_size_bytes=1024,
                results="Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species\n"
                + "1,5.1,3.5,1.4,0.2,Iris-setosa\n"
                + "2,4.9,3.0,1.4,0.2,Iris-setosa\n"
                + "3,4.7,3.2,1.3,0.2,Iris-setosa\n"
                + "4,4.6,3.1,1.5,0.2,Iris-setosa",
            ),
            dict(
                path=os.path.join(test_dir, "csv/iris-utf-16.csv"),
                encoding="utf-16",
                max_lines=5,
                max_bytes=65536,
                chunk_size_bytes=1024,
                results="Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species\n"
                + "1,5.1,3.5,1.4,0.2,Iris-setosa\n"
                + "2,4.9,3.0,1.4,0.2,Iris-setosa\n"
                + "3,4.7,3.2,1.3,0.2,Iris-setosa\n"
                + "4,4.6,3.1,1.5,0.2,Iris-setosa",
            ),
            dict(
                path=iris_32bit_filepath,
                encoding="utf-32",
                max_lines=5,
                max_bytes=65536,
                chunk_size_bytes=1024,
                results=iris_32bit_first_5,
            ),
            dict(
                path=os.path.join(test_dir, "csv/diamonds.csv"),
                encoding="utf-8",
                max_lines=5,
                max_bytes=65536,
                chunk_size_bytes=1024,
                results="carat,cut,color,clarity,depth,table,price,x,y,z\n"
                + "0.23,Ideal,E,SI2,61.5,55,326,3.95,3.98,2.43\n"
                + "0.21,Premium,E,SI1,59.8,61,326,3.89,3.84,2.31\n"
                + "0.23,Good,E,VS1,56.9,65,327,4.05,4.07,2.31\n"
                + "0.29,Premium,I,VS2,62.4,58,334,4.2,4.23,2.63",
            ),
            dict(
                path=os.path.join(test_dir, "csv/quote-test.txt"),
                encoding="utf-8",
                max_lines=9,
                max_bytes=65536,
                chunk_size_bytes=5,
                results="a b c\n"
                + '"d e f" 1 2\n'
                + "h i j\n"
                + '"k l m" 3 4\n'
                + '"n o p" 5 6\n'
                + "q r s\n"
                + "t u v\n"
                + "w x y\n"
                + "z 1 2",
            ),
            dict(
                path=os.path.join(test_dir, "csv/quote-test.txt"),
                encoding="utf-8",
                max_lines=1,
                max_bytes=65536,
                chunk_size_bytes=1024,
                results="a b c",
            ),
            dict(
                path=os.path.join(test_dir, "csv/quote-test.txt"),
                encoding="utf-8",
                max_lines=9,
                max_bytes=5,
                chunk_size_bytes=1024,
                results="a b c",
            ),
            dict(
                path=os.path.join(test_dir, "csv/quote-test.txt"),
                encoding="utf-8",
                max_lines=9,
                max_bytes=10,
                chunk_size_bytes=2,
                results='a b c\n"d e',
            ),
        ]

        for f in input_files:
            expected = f["results"]
            output_str = data_utils.load_as_str_from_file(
                file_path=f["path"],
                file_encoding=f["encoding"],
                max_lines=f["max_lines"],
                max_bytes=f["max_bytes"],
                chunk_size_bytes=f["chunk_size_bytes"],
            )
            self.assertEqual(expected, output_str, f["path"])
