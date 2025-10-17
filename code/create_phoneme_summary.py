#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script generates a summary CSV file with phoneme counts for all available languages.

It iterates through each language, processes its data to calculate phoneme frequencies,
and then compiles the results into a single output file.

From Ali Al-Azem avalazem@gmail.com 2025-10
"""

import pandas as pd
import os
import argparse
from utils import load_ipa_childes_dataset, PhonemeProcessor, syllabify_word, map_phonemes_to_categories
from run import AVAILABLE_LANGUAGES

def get_phoneme_counts_for_language(language, manually_load_diphthongs=False):
    """
    Calculates the phoneme counts for a single language by processing its dataset.
    This function contains a simplified version of the main processing pipeline.
    
    Args:
        language (str): The language to process.
        manually_load_diphthongs (bool): Whether to use manual or automatic diphthong detection.
        
    Returns:
        dict: A dictionary containing the phoneme counts for the language, or None if processing fails.
    """
    print(f"Processing {language} to get phoneme counts...")

    # Step 1: Load the dataset
    words_data = load_ipa_childes_dataset(language)
    if not words_data:
        print(f"  - No words found for {language}. Skipping.")
        return None

    ipa_words = [word_data['ipa_word'] for word_data in words_data]
    print(f"  - Loaded {len(ipa_words)} words.")

    # Step 2: Initialize phoneme processor and load vowels
    processor = PhonemeProcessor()
    if manually_load_diphthongs:
        file_path = f"../datasets/{language}/{language}-Diphthong.txt"
        vowels, sorted_vowels = processor.load_diphthongs_from_file(file_path, ipa_words)
    else:
        vowels, sorted_vowels = processor.load_vowels(ipa_words)

    # Step 3: Extract valid onsets
    valid_onsets = processor.extract_valid_onsets(ipa_words, vowels)

    # Step 4: Process words to get all phoneme patterns
    all_phoneme_patterns = []
    for word_data in words_data:
        ipa_word = word_data.get('ipa_word')
        if not ipa_word:
            continue
        
        ipa_word_clean = ipa_word.replace(' ', '')
        syllabified = syllabify_word(ipa_word_clean, sorted_vowels, valid_onsets)
        
        if syllabified:
            syllables = [syl for syl in syllabified.split('/') if syl]
            for syllable in syllables:
                phoneme_pattern = map_phonemes_to_categories(syllable, vowels, sorted_vowels)
                all_phoneme_patterns.append(phoneme_pattern)
        else:
            # Handle non-syllabified words
            phoneme_pattern = map_phonemes_to_categories(ipa_word_clean, vowels, sorted_vowels)
            all_phoneme_patterns.append(phoneme_pattern)

    # Step 5: Calculate counts from the collected patterns
    all_patterns_str = "".join(all_phoneme_patterns)
    phoneme_counts = {
        'Language': language,
        'C': all_patterns_str.count('C'),
        'V': all_patterns_str.count('V'),
        'F': all_patterns_str.count('F'),
        'G': all_patterns_str.count('G'),
        'L': all_patterns_str.count('L'),
        'N': all_patterns_str.count('N'),
    }
    
    print(f"  - Finished processing {language}.")
    return phoneme_counts

def main():
    """
    Main function to generate the phoneme count summary for all languages.
    """
    parser = argparse.ArgumentParser(
        description='Generate a summary CSV of phoneme counts for all languages in the IPA-CHILDES dataset.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--manual', '-m',
        action='store_true',
        help='Use manual diphthong loading from files instead of automatic detection.'
    )
    
    args = parser.parse_args()
    
    all_language_counts = []
    
    print(f"Starting phoneme count summary generation...")
    print(f"Manual diphthong loading: {'Yes' if args.manual else 'No'}")
    print("-" * 50)

    for i, language in enumerate(AVAILABLE_LANGUAGES, 1):
        print(f"\nProcessing language {i}/{len(AVAILABLE_LANGUAGES)}: {language}")
        try:
            counts = get_phoneme_counts_for_language(language, manually_load_diphthongs=args.manual)
            if counts:
                all_language_counts.append(counts)
        except Exception as e:
            print(f"  - ✗ Error processing {language}: {e}")

    if not all_language_counts:
        print("\nNo data was generated. Exiting.")
        return

    # Create the summary DataFrame
    summary_df = pd.DataFrame(all_language_counts)
    
    # Define output path
    base_output_dir = os.path.join(os.path.dirname(os.getcwd()), 'output')
    os.makedirs(base_output_dir, exist_ok=True)
    
    method_suffix = "_manual" if args.manual else "_auto"
    output_filename = os.path.join(base_output_dir, f'phoneme_summary{method_suffix}.csv')
    
    # Save the final CSV
    summary_df.to_csv(output_filename, index=False, encoding='utf-8')
    
    print("-" * 50)
    print(f"\n🎉 Successfully generated phoneme summary for {len(all_language_counts)} languages.")
    print(f"Summary file saved to: {output_filename}")
    print("\nSummary Preview:")
    print(summary_df.head())

if __name__ == "__main__":
    main()
