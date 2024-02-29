import csv

# List of input files
input_files = [
    "data/moreDataHead.csv",
    "data/moreDataLArm.csv",
    "data/moreDataLLeg.csv",
    "data/moreDataRArm.csv",
    "data/moreDataRLeg.csv",
    "data/moreDataTorso.csv"
]

# Output file
output_file = "data/moreData.csv"

# Open the output file in append mode
with open(output_file, "a", newline="") as outfile:
    writer = csv.writer(outfile)

    # Iterate over each input file
    for file in input_files:
        # Open the input file
        with open(file, "r") as infile:
            reader = csv.reader(infile)

            # Skip the first two lines
            next(reader)
            next(reader)

            # Append the remaining lines to the output file
            for row in reader:
                writer.writerow(row)

print("Data appended successfully!")