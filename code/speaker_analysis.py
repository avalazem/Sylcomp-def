#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script performs a phoneme summary analysis on a subsample of adult speakers from the IPA-CHILDES dataset.

It identifies prolific adult speakers (those with a minimum number of words), calculates phoneme counts for each,
and compiles the results into a single CSV file for each language. The first row of the CSV provides a baseline
by showing the phoneme counts for the entire adult-directed speech dataset.

From Ali Al-Azem avalazem@gmail.com 2025-10
"""

import pandas as pd
import os
import argparse
from collections import Counter
from utils import PhonemeProcessor, map_phonemes_to_categories, syllabify_word
from run import AVAILABLE_LANGUAGES
from datasets import load_dataset

def get_speaker_data(language, num_speakers, min_words):
    """
    Loads the dataset for a language and filters for adult speakers who meet the minimum word count.
    
    Args:
        language (str): The language to process.
        num_speakers (int): The number of top speakers to select.
        min_words (int): The minimum number of words a speaker must have to be included.
        
    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A DataFrame with all utterances from the selected speakers.
            - list: A list of the selected speaker IDs.
            - pd.DataFrame: A DataFrame with all utterances from all adults (for baseline).
            - pd.Series: Word counts for each speaker.
    """
    print(f"Loading and processing adult speaker data for {language}...")
    try:
        dataset = load_dataset("phonemetransformers/IPA-CHILDES", language, split='train')
    except Exception as e:
        print(f"  - Could not load dataset for {language}: {e}")
        return None, None, None, None

    df = dataset.to_pandas()
    
    # Filter for adult speakers only and those with an age value
    adult_df = df[(df['is_child'] == False) & (df['target_child_age'].notna())].copy()

    if adult_df.empty:
        print(f"  - No adult speaker data with target_child_age found for {language}.")
        return None, None, None, None

    # Calculate word counts for each speaker
    adult_df['word_count'] = adult_df['ipa_transcription'].str.split('WORD_BOUNDARY').str.len()
    speaker_word_counts = adult_df.groupby('speaker_id')['word_count'].sum()
    
    # Filter for speakers who meet the minimum word count
    prolific_speakers = speaker_word_counts[speaker_word_counts >= min_words]
    
    if prolific_speakers.empty:
        print(f"  - No adult speakers found with at least {min_words} words in {language}.")
        # Return the full adult dataframe for baseline, but no specific speakers
        return None, None, adult_df, speaker_word_counts
        
    print(f"  - Found {len(prolific_speakers)} adult speakers with >= {min_words} words.")
    
    # If the number of prolific speakers is less than num_speakers, take all of them.
    # Otherwise, take a random sample.
    if len(prolific_speakers) <= num_speakers:
        sampled_speakers = prolific_speakers
        print(f"  - Taking all {len(prolific_speakers)} prolific speakers.")
    else:
        # Use random_state for reproducibility
        sampled_speakers = prolific_speakers.sample(n=num_speakers, random_state=42)
        print(f"  - Randomly selected {num_speakers} speakers.")

    selected_speaker_ids = sampled_speakers.index.tolist()
    selected_speakers_df = adult_df[adult_df['speaker_id'].isin(selected_speaker_ids)]
        
    return selected_speakers_df, selected_speaker_ids, adult_df, speaker_word_counts


def calculate_stats(df, processor, vowels, sorted_vowels, valid_onsets):
    """Calculates phoneme counts, frequencies, and syllable pattern counts for a given DataFrame."""
    all_phoneme_patterns = []
    syllable_pattern_counts = Counter()

    for ipa_transcription in df['ipa_transcription'].dropna():
        words = ipa_transcription.split('WORD_BOUNDARY')
        for word in words:
            word = word.strip()
            if not word:
                continue
            
            # Phoneme analysis for the whole word
            phoneme_pattern = map_phonemes_to_categories(word, vowels, sorted_vowels)
            all_phoneme_patterns.append(phoneme_pattern)

            # Syllable analysis
            syllabified_word = syllabify_word(word, sorted_vowels, valid_onsets)
            if syllabified_word:
                syllable_list = [s for s in syllabified_word.split('/') if s]
                for syllable in syllable_list:
                    pattern = map_phonemes_to_categories(syllable, vowels, sorted_vowels)
                    if pattern:
                        syllable_pattern_counts[pattern] += 1
            
    all_patterns_str = "".join(all_phoneme_patterns)
    
    # Calculate phoneme stats
    counts = {
        'C': all_patterns_str.count('C'), 'V': all_patterns_str.count('V'),
        'F': all_patterns_str.count('F'), 'G': all_patterns_str.count('G'),
        'L': all_patterns_str.count('L'), 'N': all_patterns_str.count('N'),
    }
    total_phonemes = sum(counts.values())
    
    stats = {}
    for phoneme, count in counts.items():
        stats[f'{phoneme}_count'] = count
        # stats[f'{phoneme}_freq'] = count / total_phonemes if total_phonemes > 0 else 0
        
    # Add syllable pattern counts, prefixed with 'syl_'
    for pattern, count in syllable_pattern_counts.items():
        stats[f'syl_{pattern}'] = count

    return stats


def main():
    parser = argparse.ArgumentParser(description='Generate a phoneme summary CSV for prolific adult speakers.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-l', '--language', type=str, choices=AVAILABLE_LANGUAGES, help='Language to analyze.')
    group.add_argument('--all', action='store_true', help='Run for all available languages.')

    parser.add_argument('-m', '--manual', action='store_true', help='Use manual diphthong loading.')
    parser.add_argument('--num-speakers', type=int, default=20, help='Number of adult speakers to analyze.')
    parser.add_argument('--min-words', type=int, default=10, help='Minimum number of words for a speaker to be included.')
    
    args = parser.parse_args()

    if args.all:
        languages = AVAILABLE_LANGUAGES
        print(f"Processing all {len(languages)} languages.")
    else:
        languages = [args.language]

    all_language_results = []

    for language in languages:
        print("-" * 50)
        print(f"Processing language: {language}")
        try:
            # --- Data Loading and Filtering ---
            selected_speakers_df, speaker_ids, full_adult_df, speaker_word_counts = get_speaker_data(language, args.num_speakers, args.min_words)
            
            if full_adult_df is None:
                print(f"Skipping {language} due to data loading issues.")
                continue

            # --- Phoneme Processing Setup ---
            processor = PhonemeProcessor()
            all_adult_words = full_adult_df['ipa_transcription'].str.cat(sep=' WORD_BOUNDARY ').split('WORD_BOUNDARY')
            
            if args.manual:
                # Note: This path assumes a specific directory structure.
                diphthong_path = f"../datasets/{language}/{language}-Diphthong.txt"
                vowels, sorted_vowels = processor.load_diphthongs_from_file(diphthong_path, all_adult_words)
            else:
                vowels, sorted_vowels = processor.load_vowels(all_adult_words)

            if not vowels:
                print(f"Error: No vowels were loaded for {language}. Cannot proceed.")
                continue

            # --- Extract Onsets for Syllabification ---
            print("Extracting valid onsets for syllabification...")
            valid_onsets = processor.extract_valid_onsets(all_adult_words, vowels)
            print(f"  - Found {len(valid_onsets)} unique onsets.")

            # --- Analysis ---
            language_results = []
            
            # 1. Baseline: Entire adult dataset
            print("Calculating baseline stats from all adult data...")
            baseline_stats = calculate_stats(full_adult_df, processor, vowels, sorted_vowels, valid_onsets)
            baseline_row = {'speaker_id': 'ALL', 'target_child_age': 'N/A', 'num_words': full_adult_df['word_count'].sum(), **baseline_stats}
            language_results.append(baseline_row)
            print("  - Baseline calculated.")

            # 2. Individual prolific speakers
            if speaker_ids:
                print(f"Calculating stats for {len(speaker_ids)} selected speakers...")
                for speaker_id in speaker_ids:
                    speaker_df = selected_speakers_df[selected_speakers_df['speaker_id'] == speaker_id]
                    age = speaker_df['target_child_age'].mode().iloc[0] if not speaker_df['target_child_age'].mode().empty else 'N/A'
                    speaker_stats = calculate_stats(speaker_df, processor, vowels, sorted_vowels, valid_onsets)
                    speaker_row = {'speaker_id': speaker_id, 'target_child_age': age, 'num_words': speaker_word_counts.get(speaker_id, 0), **speaker_stats}
                    language_results.append(speaker_row)
                    print(f"  - Processed speaker: {speaker_id}")

            # --- Save or Collect Results ---
            if not language_results:
                print(f"No data was generated to save for {language}.")
                continue

            if args.all:
                # If running for all, add language column and append to master list
                for row in language_results:
                    row['language'] = language
                all_language_results.extend(language_results)
                print(f"Completed processing for {language}. Results added to combined list.")
            else:
                # If running for a single language, save it directly
                summary_df = pd.DataFrame(language_results).fillna(0)
                
                output_dir = "../output"
                os.makedirs(output_dir, exist_ok=True)
                
                mode_suffix = "_manual" if args.manual else "_auto"
                output_filename = os.path.join(output_dir, f'{language}_speaker_phoneme_summary{mode_suffix}.csv')
                
                summary_df.to_csv(output_filename, index=False, encoding='utf-8')
                
                print(f"\n🎉 Successfully generated speaker phoneme summary for {language}.")
                print(f"Summary file saved to: {output_filename}")

        except Exception as e:
            print(f"An error occurred while processing {language}: {e}")
            continue

    # --- Final Save for --all option ---
    if args.all and all_language_results:
        print("\nCombining results from all languages into a single CSV...")
        summary_df = pd.DataFrame(all_language_results).fillna(0)

        # Reorder columns to have 'language' first
        cols = summary_df.columns.tolist()
        cols.insert(0, cols.pop(cols.index('language')))
        summary_df = summary_df[cols]

        output_dir = "../output"
        os.makedirs(output_dir, exist_ok=True)
        mode_suffix = "_manual" if args.manual else "_auto"
        output_filename = os.path.join(output_dir, f'all_languages_speaker_phoneme_summary{mode_suffix}.csv')
        
        summary_df.to_csv(output_filename, index=False, encoding='utf-8')
        
        print(f"\n🎉 Successfully generated combined speaker phoneme summary for all languages.")
        print(f"Summary file saved to: {output_filename}")


    print("-" * 50)
    print("All processing complete.")


if __name__ == "__main__":
    main()