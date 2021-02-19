#!/usr/bin/python
import sys
import subprocess
import os

# Make the rst files from the current repo
subprocess.run(["sphinx-apidoc", "-f", "-e", "-o", "../docs/source", "../dataprofiler", "../dataprofiler/tests/"])

update_index_rst = True
version = str(sys.argv[1])
if not version:
    Exception("There must be a valid version argument.")

# Check if the source index file has already been updated
source_index = open("./source/index.rst", "r+")
source_index_lines = source_index.readlines()
source_index.close()
for sentence in source_index_lines:
    if sentence.startswith("   `" + version):
        update_index_rst = False

# Update the index file if needed
if update_index_rst:
    buffer = 0
    source_index = open("./source/index.rst", "w")
    for sentence in source_index_lines:
        if sentence.startswith("Documentation for"):
            doc_version = "Documentation for " + version + "\n"
            source_index.write(doc_version)
        elif sentence.startswith("Versions"):
            source_index.write("Versions\n")
            source_index.write("========\n")
            version_tag ="   `" + version + "`_\n\n"
            source_index.write(version_tag)
            version_reference = ".. _" + version + ": ../../" + version + "/html/index.html\n\n"
            source_index.write(version_reference)
            buffer = 1
        else:
            if buffer == 0:
                source_index.write(sentence)
            else:
                buffer = buffer - 1
source_index.close()

# Make the html files
build_directory = "BUILDDIR=" + version
subprocess.run(["make", "html", build_directory])

# update the index file to redirect to the most current version of documentation
index_file = open("../index.html", "w")
redirect_link = "<meta http-equiv=\"refresh\" content=\"0; url=./docs/" + version + "/html/index.html\" />"
index_file.write(redirect_link)
index_file.close()

# Removes unneeded files
files = os.listdir("./source")
for file in files:
    if file != "index.rst" and file != "conf.py":
        os.remove("./source/"+file)

