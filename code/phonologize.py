import os
import re
import glob
import csv
import logging
import sys

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class Phonologizer:
    def __init__(self, language):
        self.language = language
        self.phoneme_mappings = self.load_phoneme_mappings()

    def load_phoneme_mappings(self):
        """Charger les mappings depuis le fichier CSV correspondant à la langue."""
        phoneme_mappings = {}
        csv_path = os.path.join(f"datasets/{self.language}", f"{self.language.lower()}_phonologize.csv")
        try:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    phoneme_mappings[row['grapheme']] = row['phoneme']
            logging.info(f"Phoneme mappings loaded from {csv_path}")
        except FileNotFoundError:
            logging.error(f"Phoneme mappings file not found: {csv_path}")
        logging.debug(f"Phoneme mappings: {phoneme_mappings}")
        return phoneme_mappings

    def load_onsets(self, file_path):
        """Charger les onsets à partir d'un fichier."""
        onsets = set()
        onsets_path = f'datasets/{self.language}/{file_path}'
        try:
            with open(onsets_path, 'r', encoding='utf-8') as file:
                for line in file:
                    onsets.add(line.strip())
            logging.info(f"Onsets loaded from {onsets_path}")
        except FileNotFoundError:
            return logging.error(f"Onsets file not found: {onsets_path}")
        return onsets

    def load_vowels(self, file_path):
        """Charger les voyelles à partir d'un fichier."""
        vowels_path = f'datasets/{self.language}/{file_path}'
        try:
            with open(vowels_path, 'r', encoding='utf-8') as file:
                logging.info(f"Vowels loaded from {vowels_path}")
                vowels = file.read().strip()
        except FileNotFoundError:
            logging.error(f"Vowel file not found: {vowels_path}. Using default vowels 'aeiou'.")
            vowels = "aeiou"
        logging.debug(f"Vowels: {vowels}")
        return vowels

    def syllabify_word(self, word):
        """Syllabify a single word."""
        syllables = []
        curr_syllable = ""
        i = len(word) - 1
        while i >= 0:
            curr_char = word[i]
            curr_syllable = curr_char + curr_syllable
            if curr_char in self.vowels:
                onset = ""
                j = i - 1
                while j >= 0 and (word[j] + onset) in self.onsets:
                    onset = word[j] + onset
                    j -= 1
                i -= len(onset)
                curr_syllable = onset + curr_syllable
                syllables.insert(0, curr_syllable)
                curr_syllable = ""
            i -= 1
        return '/' + '/'.join(syllables)

    def syllabify_line(self, line):
        """Syllabify a single line of text."""
        words = line.strip().split()
        syllabified_words = [self.syllabify_word(word) for word in words]
        return ' '.join(syllabified_words)

    def syllabify_text(self, text):
        """Syllabify the content of a text."""
        lines = text.split('\n')
        syllabified_lines = [self.syllabify_line(line) for line in lines if line.strip()]
        return '\n'.join(syllabified_lines)
    def modify_chars(self, word):
        """Replace vowels 'aeiou' with 'V' and all other characters with 'C'."""
        modified_word = ''.join('V' if c.lower() in 'aeiou' else 'C' for c in word)
        return modified_word

    def apply_phoneme_mappings(self, text):
        """Apply phoneme mappings to a given text."""
        for grapheme, phoneme in sorted(self.phoneme_mappings.items(), key=lambda item: len(item[0]), reverse=True):
            text = text.replace(grapheme, phoneme)
        return text

    def process_file(self, file_path, vowels, onsets):
        """Phonologiser un fichier."""
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        processed_lines = []
        patterns = {}
        for line in lines:
            line = line.strip().lower()

            # Apply phoneme mappings before syllabification
            line = self.apply_phoneme_mappings(line)
            logging.debug(f"Line after phoneme mappings: {line}")

            currline = line
            syllline = ""
            wordarray = currline.split()

            for currword in wordarray:
                logging.debug(f"Processing word: {currword}")
                currsyllable = ""
                syllword = ""
                has_vowel = False  # Flag to check if the word has any vowels
                i = len(currword) - 1
                while i >= 0 :
                    currchar = currword[i]
                    currsyllable = currchar + currsyllable


                    if currchar in vowels:
                        has_vowel = True
                        onset = ""
                        j = i - 1
                        while j >= 0 and (currword[j] + onset) in onsets and len(onset) <= 2:
                            onset = currword[j] + onset
                            j -= 1

                        i -= len(onset)
                        currsyllable = onset + currsyllable
                        currsyllable = "/" + currsyllable
                        syllword = currsyllable + syllword
                        currsyllable = ""



                    i -= 1
                if not has_vowel:
                    currsyllable = "/" + currsyllable
                    syllword = currsyllable + syllword

                syllline += syllword + " "


            # Apply phoneme mappings after syllabification
            syllline = self.apply_phoneme_mappings(syllline.strip())

            if syllline.strip():
                processed_lines.append(syllline.strip())

            # Count occurrences of each pattern
            for word in wordarray:
                modified_word = self.modify_chars(word)
                if modified_word in patterns:
                    patterns[modified_word] += 1
                else:
                    patterns[modified_word] = 1

        # Write processed lines to a new file with "_tags.txt" suffix
        output_filename = os.path.splitext(os.path.basename(file_path))[0] + "-tags.txt"
        with open(os.path.join(os.path.dirname(file_path), output_filename), 'w', encoding='utf-8') as outfile:
            for line in processed_lines:
                cleaned_line = line.replace('/', ' ').strip()
                cleaned_line = re.sub(r'\s+', ' ', cleaned_line)
                outfile.write(f"{cleaned_line}\n")

        output_tags_output_filename = os.path.splitext(os.path.basename(file_path))[0] + "-tags-output.txt"
        with open(os.path.join(os.path.dirname(file_path), output_tags_output_filename), 'w',
                  encoding='utf-8') as tags_output_file:
            tag_output_lines = []
            for line in processed_lines:
                cleaned_line = line.replace('/', ' ').strip()
                cleaned_line = re.sub(r'\s+', ' ', cleaned_line)
                cleaned_line = ' '.join(self.modify_chars(word) for word in cleaned_line.split())
                tag_output_lines.append(cleaned_line)
                tags_output_file.write(f"{cleaned_line}\n")

        patterns = {}
        for line in tag_output_lines:
            words = line.split()
            for word in words:
                if word in patterns:
                    patterns[word] += 1
                else:
                    patterns[word] = 1

        # Write pattern counts to "-patterns_counts.txt"
        output_pattern_filename = os.path.splitext(os.path.basename(file_path))[0] + "-patterns_counts.txt"
        with open(os.path.join(os.path.dirname(file_path), output_pattern_filename), 'w',
                  encoding='utf-8') as patternfile:
            for pattern, count in patterns.items():
                patternfile.write(f"{pattern} - {count}\n")

        return processed_lines

    def sum_patterns_counts(self, folder_path):
        """Somme les motifs de tous les fichiers 'patterns_counts.txt' dans le dossier."""
        output_filename = f"{self.language}_patterns_counts.txt"
        output_path = os.path.join(folder_path, output_filename)
        # Supprimer le fichier de sortie s'il existe déjà
        if os.path.exists(output_path):
            os.remove(output_path)
            logging.info(f"Removed existing {output_filename}")
        pattern_counts = {}
        pattern_files = glob.glob(os.path.join(folder_path, "**/*patterns_counts.txt"), recursive=True)
        for pattern_file in pattern_files:
            try:
                with open(pattern_file, 'r', encoding='utf-8') as file:
                    for line in file:
                        pattern, count = line.strip().split(" - ")
                        if pattern in pattern_counts:
                            pattern_counts[pattern] += int(count)
                        else:
                            pattern_counts[pattern] = int(count)
                logging.info(f"Pattern counts from {pattern_file} added to total.")
            except FileNotFoundError:
                logging.error(f"Pattern counts file not found: {pattern_file}")
        with open(os.path.join(folder_path, output_filename), 'w', encoding='utf-8') as outfile:
            for pattern, count in pattern_counts.items():
                outfile.write(f"{pattern} - {count}\n")
        logging.info(f"Total pattern counts written to {output_filename}")

    def phonologize_folder(self, folder_path):
        """Phonologiser tous les fichiers dans un dossier."""
        onsets_file = self.language + "-ValidOnsets.txt"
        vowels_file = self.language + "-Vowels.txt"

        onsets = self.load_onsets(onsets_file)
        if onsets==None:
            logging.error(f"File not found: {onsets_file}")
            return None

        vowels = self.load_vowels(vowels_file)

        ortho_files = glob.glob(os.path.join(folder_path, "**/*ortholines.txt"), recursive=True)
        for ortho_file in ortho_files:
            file_name = os.path.basename(ortho_file)
            key_name = os.path.splitext(file_name)[0]
            processed_lines = self.process_file(ortho_file, vowels, onsets)

            output_folder = os.path.dirname(ortho_file)

            output_file = os.path.join(output_folder, key_name + "-phonologized.txt")
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write("\n".join(processed_lines))
            logging.info(f"Phonologized lines written to {output_file}")

        logging.info("Done phonologizing")
        return 1
def main():
    if len(sys.argv) != 2:
        print("Usage: python phonologize.py language ")
        sys.exit(1)

    language = sys.argv[1]


    phonologizer = Phonologizer(language)
    pf = phonologizer.phonologize_folder(f"datasets/{language}/{language}_results")
    if pf :
        phonologizer.sum_patterns_counts(f"datasets/{language}/{language}_results")

if __name__ == "__main__":
    main()