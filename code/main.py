#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline for generating syllable analysis CSV files


This script is called from run.py.
It loads the IPA dataset, takes in relevant diphthongs if necessary, processes each word, and generates syllable analysis CSV files.

From Ali Al-Azem avalazem@gmail.com 2025-07


"""
import pandas as pd
import os
from utils import load_ipa_childes_dataset, PhonemeProcessor, syllabify_word, map_phonemes_to_categories

# Combined function to generate both detailed syllable CSV and syllable frequency CSV
def generate_syllable_analysis(language, manually_load_diphthongs=False):
    """
    Generates both detailed syllable CSV and syllable frequency CSV for a given language.
    
    Args:
        language (str): The language code (e.g., 'English', 'French', etc.)
        manually_load_diphthongs (bool): If True, loads diphthongs from file; if False, uses algorithm
        
    Returns:
        tuple: (detailed_df, frequency_df) - DataFrames containing syllable analysis
    """
    print(f"Processing {language} dataset...")
    
    # Step 1: Load the dataset
    words_data = load_ipa_childes_dataset(language)
    if not words_data:
        print(f"No words found for {language}")
        return pd.DataFrame(), pd.DataFrame()  # Return empty DataFrames
    
    # Extract IPA words for processing
    ipa_words = [word_data['ipa_word'] for word_data in words_data]
    print(f"Loaded {len(ipa_words)} words")
    
    # Step 2: Initialize phoneme processor
    processor = PhonemeProcessor()
    
    # Step 3: Load vowels and either find diphthongs or load from file
    if manually_load_diphthongs: 
        file_path = f"../datasets/{language}/{language}-Diphthong.txt"
        vowels, sorted_vowels = processor.load_diphthongs_from_file(file_path, ipa_words)
        print(f"Loaded {len(vowels)} vowel units (27 individual + loaded diphthongs)")
    else:
        vowels, sorted_vowels = processor.load_vowels(ipa_words)
        print(f"Found {len(vowels)} vowel units (27 individual + found diphthongs)")
    
    # Step 4: Extract valid onsets
    valid_onsets = processor.extract_valid_onsets(ipa_words, vowels)
    print(f"Found {len(valid_onsets)} valid onsets")
    
    # Step 5: Process each word and extract syllables
    syllable_data = []
    all_phoneme_patterns = []
    
    for word_data in words_data:
        ipa_word = word_data['ipa_word']
        char_word = word_data['char_word']
        
        if not ipa_word:
            continue
            
        # Clean the IPA word - remove spaces that might be causing issues
        ipa_word_clean = ipa_word.replace(' ', '')
        
        # Syllabify the word
        syllabified = syllabify_word(ipa_word_clean, sorted_vowels, valid_onsets)
        
        if syllabified:
            # Split into individual syllables
            syllables = syllabified.split('/')
            syllables = [syl for syl in syllables if syl]  # Remove empty strings
            
            for syllable in syllables:
                # Map phonemes to categories
                phoneme_pattern = map_phonemes_to_categories(syllable, vowels, sorted_vowels)
                all_phoneme_patterns.append(phoneme_pattern)
                
                syllable_data.append({
                    'character_word': char_word,
                    'ipa_word': ipa_word_clean,
                    'syllabified_word': syllabified,
                    'syllable': syllable,
                    'phoneme_pattern': phoneme_pattern,
                })
        else:
            # If syllabification failed, still add the word for debugging
            phoneme_pattern = map_phonemes_to_categories(ipa_word_clean, vowels, sorted_vowels)
            all_phoneme_patterns.append(phoneme_pattern)
            
            syllable_data.append({
                'character_word': char_word,
                'ipa_word': ipa_word_clean,
                'syllable': ipa_word_clean,
                'phoneme_pattern': phoneme_pattern,
            })
    
    # Step 6: Create detailed DataFrame
    detailed_df = pd.DataFrame(syllable_data)
    
    if detailed_df.empty:
        print(f"No syllables extracted for {language}")
        return detailed_df, pd.DataFrame()
    
    # Step 7/8: Create frequency DataFrame based on phoneme patterns
    pattern_counts = pd.Series(all_phoneme_patterns).value_counts()
    frequency_df = pd.DataFrame({
        'phoneme_pattern': pattern_counts.index,
        'count': pattern_counts.values
    })
    
    # Step 9: Create consonant clusters DataFrame
    consonant_clusters_df = pd.DataFrame({
        'consonant_cluster': list(valid_onsets),
        'cluster_length': [len(cluster) for cluster in valid_onsets]
    }).sort_values('cluster_length', ascending=False)
    # Filter out empty clusters if any
    consonant_clusters_df = consonant_clusters_df[consonant_clusters_df['consonant_cluster'] != '']
    # Remove the length column
    consonant_clusters_df = consonant_clusters_df.drop('cluster_length', axis=1)
    
    # Step 10: Create diphthongs DataFrame
    diphthongs_list = [vowel for vowel in vowels if len(vowel) > 1]
    diphthongs_df = pd.DataFrame({
        'diphthong': diphthongs_list,
        'length': [len(diphthong) for diphthong in diphthongs_list]
    }).sort_values('length', ascending=False)
    # Filter out empty diphthongs if any
    diphthongs_df = diphthongs_df[diphthongs_df['diphthong'] != '']
    # Remove the length column
    diphthongs_df = diphthongs_df.drop('length', axis=1)
    
    # Step 11: Save all CSVs
    # Create output directory structure: ../output/language_name/
    base_output_dir = os.path.join(os.path.dirname(os.getcwd()), 'output')
    language_output_dir = os.path.join(base_output_dir, language)
    os.makedirs(language_output_dir, exist_ok=True)
    
    # Add suffix based on diphthong loading method
    method_suffix = "_manual" if manually_load_diphthongs else "_auto"
    
    # Save detailed syllable CSV
    detailed_filename = os.path.join(language_output_dir, f'{language}-syllables{method_suffix}.csv')
    detailed_df.to_csv(detailed_filename, index=False, encoding='utf-8')
    print(f"Saved {len(detailed_df)} syllables to {detailed_filename}")
    
    # Save frequency CSV
    frequency_filename = os.path.join(language_output_dir, f'{language}-syllable_frequencies{method_suffix}.csv')
    frequency_df.to_csv(frequency_filename, index=False, encoding='utf-8')
    print(f"Saved {len(frequency_df)} unique patterns to {frequency_filename}")
    
    # Save consonant clusters CSV
    clusters_filename = os.path.join(language_output_dir, f'{language}-consonant_onsets{method_suffix}.txt')
    consonant_clusters_df.to_csv(clusters_filename, index=False, encoding='utf-8', header=False)
    print(f"Saved {len(consonant_clusters_df)} unique consonant clusters to {clusters_filename}")
    
    # Save diphthongs CSV only if using automatic detection (not manual loading)
    if not manually_load_diphthongs:
        diphthongs_filename = os.path.join(language_output_dir, f'{language}-diphthongs{method_suffix}.txt')
        diphthongs_df.to_csv(diphthongs_filename, index=False, encoding='utf-8', header=False)
        print(f"Saved {len(diphthongs_df)} unique diphthongs to {diphthongs_filename}")
    else:
        print("Skipping diphthongs file creation (using manual loading from datasets folder)")
    
    # Step 12: Print summary statistics
    print("\nDetailed Syllable Summary:")
    print(f"Total syllables: {len(detailed_df)}")
    print(f"Unique syllables: {detailed_df['syllable'].nunique()}")
    print(f"Total unique patterns: {len(frequency_df)}")
    print(f"Total pattern tokens: {frequency_df['count'].sum()}")
    print(f"Most frequent patterns:")
    print(frequency_df.head(10))
    
    print("\nConsonant Clusters Summary:")
    print(f"Total unique consonant clusters: {len(consonant_clusters_df)}")
    print(f"Longest clusters:")
    print(consonant_clusters_df.head(10))
    
    print("\nDiphthongs Summary:")
    print(f"Total unique diphthongs: {len(diphthongs_df)}")
    print(f"All diphthongs:")
    print(diphthongs_df)
    
    return detailed_df, frequency_df, consonant_clusters_df, diphthongs_df

# Combines everything above into a single function to process a language
def process_language(language_code, manually_load_diphthongs=False):
    """
    Process a specific language and generate syllable CSV files.
    
    Args:
        language_code (str): Language code from AVAILABLE_LANGUAGES
        manually_load_diphthongs (bool): If True, loads diphthongs from file; if False, uses algorithm
    
    Returns:
        tuple: (detailed_df, frequency_df, consonant_clusters_df, diphthongs_df) - DataFrames containing the analysis
    """
    print(f"\n{'='*50}")
    print(f"Processing {language_code}")
    print(f"{'='*50}")
    
    try:
        detailed_df, frequency_df, consonant_clusters_df, diphthongs_df = generate_syllable_analysis(language_code, manually_load_diphthongs)
        return detailed_df, frequency_df, consonant_clusters_df, diphthongs_df
        
    except Exception as e:
        print(f"Error processing {language_code}: {e}")
        return None, None, None, None
        
# Example usage:
if __name__ == '__main__':
    # For testing: process a single language with both manual and automatic diphthong loading
    process_language('English', manually_load_diphthongs=True)
    process_language('English', manually_load_diphthongs=False)