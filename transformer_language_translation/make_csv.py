import csv

# File paths for the English and French text files
english_file = '/home2/shaon/ANLP/transformer_from_scratch/ted-talks-corpus/test.en'
french_file = '/home2/shaon/ANLP/transformer_from_scratch/ted-talks-corpus/test.fr'
output_csv = 'english_french_test.csv'

# Read English sentences
with open(english_file, 'r', encoding='utf-8') as eng_file:
    english_sentences = eng_file.readlines()

# Read French sentences
with open(french_file, 'r', encoding='utf-8') as fr_file:
    french_sentences = fr_file.readlines()

# Ensure that both files have the same number of sentences
if len(english_sentences) != len(french_sentences):
    raise ValueError("The number of lines in english.en and french.fr do not match!")

# Create CSV file with English and French columns
with open(output_csv, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['English', 'French'])  # Writing header

    # Write rows with English and French sentences
    for english_sentence, french_sentence in zip(english_sentences, french_sentences):
        # Strip newline characters and extra spaces
        csv_writer.writerow([english_sentence.lower().strip(), french_sentence.lower().strip()])

print(f"CSV file '{output_csv}' has been created.")