import re
from typing import Dict, List, Tuple


def generate_data_extraction_regex(data_structure: Dict[str, str]) -> str:
    regex_parts = []
    for key, value in data_structure.items():
        regex_parts.append(f"(?P<{key}>{value})")
    return "|".join(regex_parts)


def extract_data(text: str, regex: str) -> List[Dict[str, str]]:
    data = []
    for match in re.finditer(regex, text):
        data.append(match.groupdict())
    return data


def generate_summary_table(data: List[Dict[str, str]], table_structure: List[Tuple[str, str]]) -> str:
    headers = [header for header, _ in table_structure]
    rows = [headers]
    for item in data:
        row = []
        for _, field in table_structure:
            row.append(item.get(field, ""))
        rows.append(row)
    col_widths = [max(len(str(row[i])) for row in rows) + 2 for i in range(len(headers))]
    table = ""
    for row in rows:
        table += "|".join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers))) + "\n"
    return table


# Example usage:
text = """There are many fruits that were found on the recently discovered planet Goocrux.
          There are neoskizzles that grow there, which are purple and taste like candy.
          There are also loheckles, which are a grayish blue fruit and are very tart, a little bit like a lemon.
          Pounits are a bright green color and are more savory than sweet.
          There are also plenty of loopnovas which are a neon pink flavor and taste like cotton candy.
          Finally, there are fruits called glowls, which have a very sour and bitter taste which is acidic and caustic, and a pale orange tinge to them."""

# Define the data structure for fruits
fruit_data_structure = {"Fruit": r"\w+", "Color": r"[a-zA-Z ]+", "Flavor": r"[a-zA-Z ,]+", }

# Define the table structure for the fruit summary table
fruit_table_structure = [("Fruit", "Fruit"), ("Color", "Color"), ("Flavor", "Flavor")]

# Generate the regular expression for extracting fruit data
fruit_regex = generate_data_extraction_regex(fruit_data_structure)

# Extract the fruit data from the text using the generated regular expression
fruit_data = extract_data(text, fruit_regex)

# Generate the fruit summary table using the extracted fruit data and the table structure
fruit_table = generate_summary_table(fruit_data, fruit_table_structure)

# Print the fruit summary table
print(fruit_table)
