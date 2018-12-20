import pandas as pd
import numpy as np
import sys

tax_csv = sys.argv[1]

df = pd.read_csv(tax_csv, header=0, index_col=0)
# assignment_values = df.values
# max_positions = np.argmax(assignment_values, axis=0)
# tax = [df.index.values[mp] for mp in max_positions]
# sample_ids = df.columns.values

# Percent threshold below which to ignore values.
# Reduces the number of features to the model.
threshold = 0.1

df = df[(df.values > threshold).any(1)]
