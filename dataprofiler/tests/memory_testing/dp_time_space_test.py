import json
from dataprofiler import Data, Profiler

data = Data("/Users/ksneab/c1_projects/c1_repos/growml-datasets/output_data/cat_100000.csv")

profile = Profiler(data) # Calculate Statistics, Entity Recognition, etc