# Paste your list of numbers here as a multiline string
raw_data_true = """
"""

# Paste your list of numbers here as a multiline string
raw_data_false = """
"""

#Convert to list
numbers_true = [float(line.strip()) for line in raw_data_true.strip().splitlines() if line.strip()]

#Format
formatted_lines_true = [f"{num}, 1" for num in numbers_true]

#Convert to list
numbers_false = [float(line.strip()) for line in raw_data_false.strip().splitlines() if line.strip()]

#Format
formatted_lines_false = [f"{num}, 0" for num in numbers_false]

#Write and save
with open("eyeCloseData.txt", "w") as f:
    f.write("\n".join(formatted_lines_true))
    f.write("\n".join(formatted_lines_false))