import subprocess

# Install the arff library using pip
subprocess.check_call(['pip', 'install', 'liac-arff', '--quiet'])

import arff

with open('phpgNaXZe.arff') as f:
    dataset = arff.load(f, encode_nominal=True)

# Get the attribute names and data from the dataset
attributes = [attr[0] for attr in dataset['attributes']]
data = dataset['data']

# Print the attribute names and data
print(attributes)
print(data)
