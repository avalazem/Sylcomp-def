#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corpus Processor

This script processes and cleans linguistic corpus files in various formats
(.cha, .eaf, .csv, .txt) and extracts valid onsets based on language-specific vowels.
It handles participant extraction, text cleaning, and onset identification.

Usage:
    python corpus_processor.py {format} {language}
    
    format: cha, eaf, csv, or txt
    language: Language code for vowel identification and file naming
    
From Alex Cristia alecristia@gmail.com 2016-11

"""

import os
import glob
import re
import argparse
from pathlib import Path
import logging
import pandas as pd
from collections import defaultdict
import pympi
import pylangacq
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Speaker ID to type mapping with default value
SPEAKER_ID_TO_TYPE = defaultdict(
    lambda: "NA", {'MA1-tca':'MAL','FA1-tca':'FEM','FA2-tca':'FEM', 'FC1-tca':'FEM','FC2-tca':'FEM','MA2-tca':'MAL', 
                   'CHI-tca':'CHI', 'C1': 'OCH', 'C2': 'OCH', 'CHI': 'CHI', 'CHI*': 'CHI', 'EE1': 'NA', 'EE2': 'NA', 
                   'FA0': 'FEM', 'FA1': 'FEM', 'FA2': 'FEM', 'FA3': 'FEM', 'FA4': 'FEM', 'FA5': 'FEM', 'FA6': 'FEM', 
                   'FA7': 'FEM', 'FA8': 'FEM', 'FAE': 'NA', 'FC1': 'OCH', 'FC2': 'OCH', 'FC3': 'OCH', 'FC4': 'OCH', 
                   'FCE': 'NA', 'MA0': 'MAL', 'MA1': 'MAL', 'MA2': 'MAL', 'MA3': 'MAL', 'MA4': 'MAL', 'MA5': 'MAL', 
                   'MA6': 'MAL', 'MAE': 'NA', 'MC1': 'OCH', 'MC*': 'OCH', 'MC2': 'OCH', 'MC3': 'OCH', 'MC4': 'OCH', 
                   'MC5': 'OCH', 'MC6': 'OCH', 'MCE': 'NA', 'MI1': 'OCH', 'MOT*': 'FEM', 'OC0': 'OCH', 'UA1': 'NA', 
                   'UA2': 'NA', 'UA3': 'NA', 'UA4': 'NA', 'UA5': 'NA', 'UA6': 'NA', 'UC1': 'OCH', 'UC2': 'OCH', 
                   'UC3': 'OCH', 'UC4': 'OCH', 'UC5': 'OCH', 'UC6': 'OCH', 'Notes': "notes", 'AudioOnly': 'AudioOnly', 
                   'code': 'code', 'code_num': 'code_num'})


def clean_line(line, words_to_delete, language=None):
    """
    Clean a single line of text by removing annotations, symbols, and unwanted words.
    
    Args:
        line (str): The line to clean
        words_to_delete (list): List of words to remove from the cleaned text
        language (str, optional): The language of the text, affects certain cleaning rules
        
    Returns:
        str or None: The cleaned line or None if the line is empty after cleaning
    """
    # Compilation of frequently used regex patterns
    regexes = {
        'control_chars': re.compile(r'[\x15].*[\x15]'),
        'email_like': re.compile(r'\S*@\S*\s?'),
        'speaker_id': re.compile(r'^.{0,4}:'),
        'brackets_special': re.compile(r'\[.*?[=!%+].*?\]'),
        'brackets_content': re.compile(r'\[.*?\]'),
        'parentheses': re.compile(r'\([^)]*\)'),
        'word_with_ampersand': re.compile(r'\b\w*&\w*\b'),
        'ampersand_equals': re.compile(r'&=.*'),
        'special_chars': re.compile(r'[\"^\'/;⌉<>,:~""⌈⌉]'),
        'punctuation': re.compile(r'[.?!;⌉⌈⌉\^+]'),
        'filler_words': re.compile(r'xxx|www|XXX|yyy|jmm'),
        'at_symbol': re.compile(r'@[^ ]*'),
        'hyphens': re.compile(r'[-]'),
        'zero_char': re.compile(r'\b0\b'),
        'misc_symbols': re.compile(r'‡|„|&'),
        'non_word_chars': re.compile(r"[^\w\s\u00C0-\u024F\u1E00-\u1EFF]"),
        'multiple_spaces': re.compile(r'\s+')
    }
    
    # 1. Remove control characters and annotations
    line = regexes['control_chars'].sub('', line)
    line = regexes['email_like'].sub('', line)
    line = regexes['speaker_id'].sub('', line)
    line = regexes['brackets_special'].sub('', line)
    
    # 2. Language-specific processing
    if language != "Hebrew":
        line = re.sub(r'\[=\s*([^]]+)\]', r'[: \1]', line)
    
    # 3. For content followed by [: ], keep the content after [: ]
    line = re.sub(r'(\b\w+\b|\b<\w+>\b)\s*\[:\s*([\w\s\'-]+)\]', r'\2', line)
    
    # 4. Keep everything in angle brackets if not followed by [: ]
    line = re.sub(r'<([^>:]+)>', r'\1', line)
    
    # 5. Remove everything after & and all &=
    line = re.sub(r'&.*', '', line).strip()
    
    # 6. General cleaning and formatting
    line = regexes['brackets_content'].sub('', line)
    line = regexes['parentheses'].sub('', line)
    line = regexes['word_with_ampersand'].sub('', line)
    line = regexes['ampersand_equals'].sub('', line).strip()
    line = line.replace('_', ' ')
    line = regexes['special_chars'].sub('', line)
    line = regexes['punctuation'].sub('', line)
    line = regexes['filler_words'].sub('', line)
    line = regexes['at_symbol'].sub('', line)
    line = regexes['hyphens'].sub('', line)
    line = regexes['zero_char'].sub('', line)
    line = regexes['misc_symbols'].sub('', line)
    line = regexes['non_word_chars'].sub('', line)
    line = regexes['multiple_spaces'].sub(' ', line).strip()
    
    # 7. Convert to lowercase
    line = line.lower()
    
    # 8. Remove words from words_to_delete
    if words_to_delete:
        for word in words_to_delete:
            line = re.sub(rf'\b{re.escape(word)}\b', '', line)
        line = regexes['multiple_spaces'].sub(' ', line).strip()
    
    return line if line else None


def extract_right_side_text(input_file, output_file):
    """
    Extract text that appears on the right side in specially formatted files.
    These formats typically contain lines with tabs or spaces separating
    timing/metadata information from the transcribed text.
    
    Args:
        input_file (str): Path to the input file
        output_file (str): Path to the output file
    
    Returns:
        bool: True if extraction was successful, False otherwise
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        extracted_texts = []
        
        for line in lines:
            # Different strategies for extracting right-side text
            
            # Strategy 1: Split by tab (most common)
            parts = line.split('\t')
            if len(parts) > 7:  # Based on provided example
                text = parts[-1].strip()
                if text and not text.startswith('A_'):  # Ignore metadata lines
                    extracted_texts.append(text)
                continue
                
            # Strategy 2: Look for text after timestamp (00:00:00.000)
            timestamp_match = re.search(r'\d{2}:\d{2}:\d{2}\.\d{3}', line)
            if timestamp_match:
                text = line[timestamp_match.end():].strip()
                if text:
                    extracted_texts.append(text)
                continue
                
            # Strategy 3: Look for text after multiple consecutive spaces
            space_parts = re.split(r'\s{3,}', line)
            if len(space_parts) > 1:
                text = space_parts[-1].strip()
                if text:
                    extracted_texts.append(text)
                    
        # Write extracted texts
        if extracted_texts:
            with open(output_file, 'w', encoding='utf-8') as file:
                for text in extracted_texts:
                    file.write(text + '\n')
            logging.info(f"Right-side text extraction completed: {output_file}")
            return True
        else:
            logging.warning(f"No text was extracted from {input_file}")
            return False
            
    except Exception as e:
        logging.error(f"Failed to process {input_file}: {e}")
        return False


class BaseFileProcessor(ABC):
    """Abstract base class for file processors."""
    
    @staticmethod
    @abstractmethod
    def extract_included_parts(file_path):
        """Extract participants from a file."""
        pass
    
    @staticmethod
    @abstractmethod
    def process_file(input_file, output_file, included_parts, words_to_delete):
        """Process a file and write cleaned text to the output file."""
        pass
    
    @staticmethod
    def write_cleaned_lines(cleaned_lines, output_file):
        """Common method to write cleaned lines to an output file."""
        if cleaned_lines:
            try:
                with open(output_file, 'w', encoding='utf-8') as file:
                    for line in cleaned_lines:
                        file.write(line + '\n')
                logging.info(f"Output written to {output_file}")
                return True
            except Exception as e:
                logging.error(f"Failed to write {output_file}: {e}")
                return False
        else:
            logging.info("No cleaned lines to write.")
            return False


class ParticipantExtractor:
    """Class to handle extraction of participants from .cha files."""

    TO_RM = ['Target_Child', 'Child', 'Sister', 'Brother', 'Cousin', 'Boy', 'Girl',
             'Unidentified', 'Sibling', 'Target', 'Non_Hum', 'Play']

    @staticmethod
    def extract_included_parts(file_path):
        """Extracts participants that are not in the exclusion list from a .cha file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.read().replace('\r', '\n')

        except Exception as e:
            logging.error(f"Failed to read {file_path}: {e}")
            return []

        ids = []
        for line in lines.split('\n'):
            if line.startswith("@ID"):
                parts = line.split('|')
                if not any(part in parts for part in ParticipantExtractor.TO_RM):
                    ids.append(parts[2].strip())
        return ids


class ChaFileProcessor(BaseFileProcessor):
    """Class to handle processing of .cha files."""

    @staticmethod
    def extract_included_parts(file_path):
        """Extract participants from a .cha file."""
        return ParticipantExtractor.extract_included_parts(file_path)

    @staticmethod
    def process_file(input_file, output_file, included_parts, words_to_delete):
        """Process a .cha file and write cleaned text to the output file."""
        try:
            with open(input_file, 'r', encoding='utf-8') as file:
                lines = file.readlines()

        except Exception as e:
            logging.error(f"Failed to read {input_file}: {e}")
            return False

        cleaned_lines = []
        language = "Hebrew" if "Hebrew" in input_file else None
        
        for line in lines:
            if any(f"*{part}:" in line for part in included_parts):
                cleaned_line = clean_line(line, words_to_delete, language=language)
                if cleaned_line:
                    cleaned_lines.append(cleaned_line)

        return BaseFileProcessor.write_cleaned_lines(cleaned_lines, output_file)


class EafFileProcessor(BaseFileProcessor):
    """Class to handle processing of .eaf files."""

    @staticmethod
    def convert(filename: str) -> pd.DataFrame:
        """Convert an .eaf file to a pandas DataFrame."""
        eaf = pympi.Elan.Eaf(filename)

        segments = {}
        for tier_name in eaf.tiers:
            annotations = eaf.tiers[tier_name][0]

            if tier_name not in SPEAKER_ID_TO_TYPE and len(annotations) > 0:
                logging.warning(f"Unknown tier '{tier_name}' will be ignored in '{filename}'")
                continue

            for aid in annotations:
                (start_ts, end_ts, value, svg_ref) = annotations[aid]
                (start_t, end_t) = (eaf.timeslots[start_ts], eaf.timeslots[end_ts])
                segment = {
                    "segment_onset": int(round(start_t)),
                    "segment_offset": int(round(end_t)),
                    "speaker_id": tier_name,
                    "speaker_type": SPEAKER_ID_TO_TYPE[tier_name],
                    "vcm_type": "NA",
                    "lex_type": "NA",
                    "mwu_type": "NA",
                    "addressee": "NA",
                    "transcription": value if value != "0" else "0.",
                    "words": "NA",
                }

                segments[aid] = segment

        for tier_name in eaf.tiers:
            if "@" in tier_name:
                label, ref = tier_name.split("@")
            else:
                label, ref = tier_name, None
            
            reference_annotations = eaf.tiers[tier_name][1]

            if ref not in SPEAKER_ID_TO_TYPE:
                continue

            for aid in reference_annotations:
                (ann, value, prev, svg) = reference_annotations[aid]

                ann = aid
                parentTier = eaf.tiers[eaf.annotations[ann]]
                while "PARENT_REF" in parentTier[2] and parentTier[2]["PARENT_REF"] and len(parentTier[2]) > 0:
                    ann = parentTier[1][ann][0]
                    parentTier = eaf.tiers[eaf.annotations[ann]]

                if ann not in segments:
                    logging.warning(f"Annotation '{ann}' not found in segments for '{filename}'")
                    continue

                segment = segments[ann]

                if label == "lex":
                    segment["lex_type"] = value
                elif label == "mwu":
                    segment["mwu_type"] = value
                elif label == "xds":
                    segment["addressee"] = value
                elif label == "vcm":
                    segment["vcm_type"] = value
                elif label == "msc":
                    segment["msc_type"] = value
        return pd.DataFrame(segments.values())

    @staticmethod
    def extract_included_parts(file_path):
        """Extract participants from an .eaf file."""
        df = EafFileProcessor.convert(file_path)
        if df.empty:
            return []
        else:
            included_parts = df[(df['speaker_type'] == 'MAL') | (df['speaker_type'] == 'FEM')]['speaker_id'].unique().tolist()
            return included_parts

    @staticmethod
    def process_file(input_file, output_file, included_parts, words_to_delete):
        """Process an .eaf file and write cleaned text to the output file."""
        df = EafFileProcessor.convert(input_file)
        filtered_df = df[df['speaker_id'].isin(included_parts)]
        cleaned_lines = []

        for transcription in filtered_df['transcription']:
            cleaned_line = clean_line(transcription, words_to_delete, language=None)
            if cleaned_line:
                cleaned_lines.append(cleaned_line)

        return BaseFileProcessor.write_cleaned_lines(cleaned_lines, output_file)


class CsvFileProcessor(BaseFileProcessor):
    """Class to handle processing of .csv files."""

    @staticmethod
    def convert(filename: str) -> pd.DataFrame:
        """Read a .csv file and convert it to a pandas DataFrame."""
        try:
            df = pd.read_csv(filename)
        except Exception as e:
            logging.error(f"Failed to read {filename}: {e}")
            return pd.DataFrame()
        return df

    @staticmethod
    def extract_included_parts(file_path):
        """Extract participants from a .csv file."""
        df = CsvFileProcessor.convert(file_path)
        if df.empty:
            return []
        included_parts = df[(df['speaker_type'] == 'MAL') | (df['speaker_type'] == 'FEM')]['speaker_id'].unique().tolist()
        return included_parts

    @staticmethod
    def process_file(input_file, output_file, included_parts, words_to_delete):
        """Process a .csv file and write cleaned text to the output file."""
        df = CsvFileProcessor.convert(input_file)
        filtered_df = df[df['speaker_id'].isin(included_parts)]
        cleaned_lines = []

        for transcription in filtered_df['transcription']:
            cleaned_line = clean_line(transcription, words_to_delete, language=None)
            if cleaned_line:
                cleaned_lines.append(cleaned_line)

        return BaseFileProcessor.write_cleaned_lines(cleaned_lines, output_file)


class TextFileProcessor(BaseFileProcessor):
    """Class to handle processing of .txt files.
    For text files, we extract the right-side text and write it directly to the ortholines file."""

    @staticmethod
    def extract_included_parts(file_path):
        """
        Extract participants from a .txt file.
        For text files, we assume all content should be processed.
        Returns a default list with a single participant ID.
        """
        return ["TXT"]  # A fictitious ID to indicate it's a text file

    @staticmethod
    def process_file(input_file, output_file, included_parts, words_to_delete):
        """
        Process a .txt file by extracting the right-side text and writing it to the output file.
        This directly uses the right-side text extraction without creating an intermediate file.
        
        Args:
            input_file (str): Path to the input file
            output_file (str): Path to the output file
            included_parts (list): List of participant IDs (not used for text files)
            words_to_delete (list): List of words to remove from the cleaned text
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            with open(input_file, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                
            extracted_texts = []
            
            for line in lines:
                # Different strategies for extracting right-side text
                
                # Strategy 1: Split by tab (most common)
                parts = line.split('\t')
                if len(parts) > 7:  # Based on provided example
                    text = parts[-1].strip()
                    if text and not text.startswith('A_'):  # Ignore metadata lines
                        extracted_texts.append(text)
                    continue
                    
                # Strategy 2: Look for text after timestamp (00:00:00.000)
                timestamp_match = re.search(r'\d{2}:\d{2}:\d{2}\.\d{3}', line)
                if timestamp_match:
                    text = line[timestamp_match.end():].strip()
                    if text:
                        extracted_texts.append(text)
                    continue
                    
                # Strategy 3: Look for text after multiple consecutive spaces
                space_parts = re.split(r'\s{3,}', line)
                if len(space_parts) > 1:
                    text = space_parts[-1].strip()
                    if text:
                        extracted_texts.append(text)
            
            # Clean the extracted texts
            cleaned_lines = []
            language = "Hebrew" if "Hebrew" in input_file else None
            
            for text in extracted_texts:
                cleaned_line = clean_line(text, words_to_delete, language=language)
                if cleaned_line:
                    cleaned_lines.append(cleaned_line)
                    
            # Write cleaned lines directly to the output file
            return BaseFileProcessor.write_cleaned_lines(cleaned_lines, output_file)
                
        except Exception as e:
            logging.error(f"Failed to process {input_file}: {e}")
            return False


class RightSideTextProcessor(BaseFileProcessor):
    """Class to handle processing of right-side text extraction."""

    @staticmethod
    def extract_included_parts(file_path):
        """For right-side text extraction, we assume all content should be processed."""
        return ["RST"]  # A fictitious ID to indicate it's right-side text

    @staticmethod
    def process_file(input_file, output_file, included_parts, words_to_delete):
        """Extract and process right-side text."""
        extracted_output = f"{os.path.splitext(output_file)[0]}-extracted.txt"
        if extract_right_side_text(input_file, extracted_output):
            # Now clean the extracted text
            try:
                with open(extracted_output, 'r', encoding='utf-8') as file:
                    lines = file.readlines()

                cleaned_lines = []
                language = "Hebrew" if "Hebrew" in input_file else None
                
                for line in lines:
                    cleaned_line = clean_line(line, words_to_delete, language=language)
                    if cleaned_line:
                        cleaned_lines.append(cleaned_line)

                return BaseFileProcessor.write_cleaned_lines(cleaned_lines, output_file)
            
            except Exception as e:
                logging.error(f"Failed to process extracted text from {input_file}: {e}")
                return False
        else:
            logging.error(f"Failed to extract right-side text from {input_file}")
            return False


class InterjectionRemover:
    """Class to handle removal of interjection lines from text files."""

    @staticmethod
    def delete_interjections(folder_path):
        """Deletes lines containing 'hm', 'mhm', 'mhmm', 'nnnn' from text files in the specified folder."""
        files = glob.glob(os.path.join(folder_path, "*-ortholines.txt"))

        for file_path in files:
            try:
                with open(file_path, 'r+', encoding='utf-8') as file:
                    lines = file.readlines()
                    file.seek(0)
                    for line in lines:
                        if not any(word in line for word in ["hm", "mhm", "mhmm", "mmhm", "hm", "nnnnn", "brrr"]):
                            file.write(line)
                    file.truncate()
            except Exception as e:
                logging.error(f"Failed to process {file_path}: {e}")

        logging.info("Deletion of 'hm', 'mhm', 'mhmm' lines completed.")

    @staticmethod
    def get_words_to_delete(file_path):
        """Read the list of words to delete from a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                words = [line.strip() for line in file]
        except FileNotFoundError as e:
            logging.warning(f"Failed to read {file_path}: {e}, words to delete is empty")
            words = []
        return words


class FileProcessor:
    """Class to handle the processing of files in the dataset."""

    @staticmethod
    def get_processor(file_format):
        """Return the appropriate processor class based on file format."""
        processors = {
            'cha': ChaFileProcessor,
            'eaf': EafFileProcessor,
            'csv': CsvFileProcessor,
            'txt': TextFileProcessor,
            'rst': RightSideTextProcessor  # Added for right-side text processing
        }
        return processors.get(file_format)

    @staticmethod
    def process_files(input_corpus, res_folder, file_format):
        """Process all files in the input corpus and clean them."""
        words_to_delete = InterjectionRemover.get_words_to_delete(os.path.join(input_corpus, "to-delete.txt"))
        
        processor = FileProcessor.get_processor(file_format)
        if not processor:
            logging.error(f"Unsupported file format: {file_format}")
            return
            
        file_extension = f"*.{file_format}"
        files = glob.glob(os.path.join(input_corpus, '**', file_extension), recursive=True)
        
        if not files:
            logging.info(f"No {file_extension} files found in {input_corpus}")
            return
            
        for f in files:
            included_parts = processor.extract_included_parts(f)
            if not included_parts:
                logging.info(f"No valid participants found in {f}")
                continue

            output_dir = os.path.join(res_folder, os.path.basename(os.path.dirname(f)))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            ortho_file_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(f))[0]}-ortholines.txt")
            processor.process_file(f, ortho_file_path, included_parts, words_to_delete)

        InterjectionRemover.delete_interjections(res_folder)


class VowelProcessor:
    """Class to handle reading vowels and extracting valid onsets."""

    @staticmethod
    def read_vowels(vowel_file):
        """Reads vowels from a file or returns default vowels if the file is not found."""
        try:
            with open(vowel_file, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except FileNotFoundError:
            logging.warning(f"Vowel file {vowel_file} not found. Using default vowels 'aeiou'.")
            return "aeiou"

    @staticmethod
    def process_file(file_path):
        """Processes a file to extract words."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return [word.strip() for line in file for word in line.replace(" ", "\n").split("\n") if word.strip()]
        except Exception as e:
            logging.error(f"Failed to process {file_path}: {e}")
            return []

    @staticmethod
    def extract_valid_onsets(words, vowels):
        """Extracts valid onsets from a list of words based on the given vowels."""
        pattern = re.compile(rf'^([^{vowels}]+)', re.IGNORECASE)
        alphabetical = re.compile(r'^[a-zA-Z\u00C0-\u017E]+$')
        return {match.group(1) for word in words for match in [pattern.match(word)] if match and alphabetical.match(match.group(1))}


def main(file_format, language):
    """Main method to process the dataset and extract valid onsets."""
    dataset_path = os.path.join("datasets", language)
    res_folder = os.path.join(dataset_path, f"{language}_results")
    
    # Process the files according to the specified format
    if file_format == 'rst':  # Special case for right-side text processing
        # Find all text files that could contain right-side text
        potential_files = []
        for ext in ['txt', 'cha', 'eaf', 'csv']:
            potential_files.extend(glob.glob(os.path.join(dataset_path, '**', f'*.{ext}'), recursive=True))
            
        if not potential_files:
            logging.info(f"No potential files found in {dataset_path}")
            return
            
        # Process each file for right-side text extraction
        words_to_delete = InterjectionRemover.get_words_to_delete(os.path.join(dataset_path, "to-delete.txt"))
        for f in potential_files:
            output_dir = os.path.join(res_folder, os.path.basename(os.path.dirname(f)))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            ortho_file_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(f))[0]}-rst-ortholines.txt")
            RightSideTextProcessor.process_file(f, ortho_file_path, ["RST"], words_to_delete)
    else:
        # Call the existing method for processing other formats
        FileProcessor.process_files(dataset_path, res_folder, file_format)
    
    # Continue with vowel processing and onset extraction
    vowel_file = os.path.join(dataset_path, f"{language}-Vowels.txt")
    vowels = VowelProcessor.read_vowels(vowel_file)

    files = glob.glob(str(Path(res_folder) / "**" / "*-ortholines.txt"), recursive=True)
    all_onsets = {onset for file in files for onset in VowelProcessor.extract_valid_onsets(VowelProcessor.process_file(file), vowels)}

    output_file_path = os.path.join(dataset_path, f"{language}-ValidOnsets-to-be-checked.txt")
    with open(output_file_path, "w", encoding='utf-8') as output_file:
        output_file.write('\n'.join(sorted(all_onsets)))
    
    logging.info(f"Valid onsets have been written to: {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and clean corpus files.')
    parser.add_argument('format', choices=['eaf', 'cha', 'csv', 'txt', 'rst'], 
                        help='Format of the input data (rst for right-side text extraction)')
    parser.add_argument('language', help='Language to determine the vowel file and name the output file')

    args = parser.parse_args()
    main(args.format, args.language)