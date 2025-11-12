#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script contains utility functions for loading and processing the IPA-CHILDES dataset, including phoneme mapping and syllabification.
It takes phonetic transcriptions and maps them to simplified categories for analysis.

syllabify_word and PhonemeProcessor functions are originally adapted from Alex Cristia alecristia@gmail.com 2016-11, 
Modified by Laia Fibla laia.fibla.reixachs@gmail.com 2016-09-28
translated from bash to R by Kaijia Tey kaijiatey@gmail.com 2025
from R to Python by Alix Bourrée alix.bourree@gmail.com 2025
(re corpus_processor.py and phonologize.py)

Adapted into IPA-CHILDES pipeline by Ali Al-Azem avalazem@gmail.com 2025-07. Github: avalazem


"""

import sys
import os
import re
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datasets import load_dataset


def check_available_languages():
    """Check what languages are available in the IPA-CHILDES dataset."""
    try:
        from datasets import get_dataset_config_names
        configs = get_dataset_config_names("phonemetransformers/IPA-CHILDES")
        print(f"Available language configurations: {configs}")
        return configs
    except Exception as e:
        print(f"Error checking available languages: {e}")
        return []


# We define a dictionary containing mappings from IPA phonemes to simplified categories
PHONEME_MAPPING = {
    'Glides': {'j', 'w', 'ɥ', 'ɰ'},
    'Nasals': {'m', 'ɱ', 'n', 'ɲ', 'ŋ', 'ɳ', 'ɴ'},
    'Liquids': {'l','ɫ','ɭ','ʎ','ʟ','ɹ','ɻ','r','ɾ','ɽ','ᴚ','rr'},
    'Fricatives': {'f','v','θ','ð','s','z','ʃ','ʒ','h','ɦ','ɸ','β','ʂ','ʐ','ɕ','ʑ','x','χ','ʁ','ç','ʝ'},
    'Vowels': {'i','y','ɨ','ʉ','ɯ','u','ɪ','ʏ','ʊ','e','ø','ɘ','ɵ','ɤ','o','ə','ɛ','œ','ɜ','ɞ','ʌ','ɔ','æ','ɐ','a','ɑ','ɒ', 'ɚ'},
    'Consonants': {'p', 'b', 't', 'd', 'ʈ', 'ɖ', 'c', 'ɟ', 'k', 'g', 'q', 'ɢ', 'ʡ', 'ɓ', 'ɗ', 'ʄ', 'ɠ', 'ʛ', 'ts', 'dz','ʙ', 'ʀ', 'd̠ʒ', 'tʃ', 'pf', 'kx', 'tʂ', 'dʐ', 'tɕ', 'dʑ', 'Ʋ','g','t̠ʃ','t̠ʃ','ph','qh','kh','th','sh','t̺s̺','t̪̻s̪̻','t͡s','t͡sʰ'}
    # 'ts' should be removed once Mandarin 't͡s','t͡sʰ' cases are fixed as a C
}

# Create sets for each phoneme category for faster lookups
FRICATIVES = PHONEME_MAPPING['Fricatives']
GLIDES = PHONEME_MAPPING['Glides']
LIQUIDS = PHONEME_MAPPING['Liquids']
NASALS = PHONEME_MAPPING['Nasals']
VOWELS = PHONEME_MAPPING['Vowels']
CONSONANTS = PHONEME_MAPPING['Consonants']

# Function to load the dataset from IPA-CHILDES
def load_ipa_childes_dataset(LANGUAGE):
    try: 
        print(f"Loading dataset for language: {LANGUAGE}")
        ipa_childes_ds = load_dataset("phonemetransformers/IPA-CHILDES", f"{LANGUAGE}")
        print("Dataset loaded successfully")
        words_data = extract_words(ipa_childes_ds)
        return words_data  # Return the extracted words data
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return None

# Function to filter out adult-spoken words and extract both IPA and character split utterances
def extract_words(ipa_childes_ds):
    if ipa_childes_ds is not None:
        print(f"Dataset type: {type(ipa_childes_ds)}")
        print(f"Dataset keys: {list(ipa_childes_ds.keys()) if hasattr(ipa_childes_ds, 'keys') else 'No keys'}")
        
        # Handle dataset structure - check if it has splits like 'train'
        if hasattr(ipa_childes_ds, 'keys') and 'train' in ipa_childes_ds:
            dataset = ipa_childes_ds['train']
            print(f"Using 'train' split with {len(dataset)} examples")
        else:
            # If no splits, use the dataset directly
            dataset = ipa_childes_ds
            print(f"Using dataset directly with {len(dataset)} examples")
        
        print(f"Dataset columns: {dataset.column_names}")
        
        # Filter for adult-spoken utterances
        adult_spoken_ds = dataset.filter(lambda x: x['is_child'] == False)
        print(f"Adult-spoken dataset has {len(adult_spoken_ds)} examples")
        
        if len(adult_spoken_ds) == 0:
            print("No adult-spoken utterances found matching the criteria!")
            return []
        
        # Use the full adult-spoken dataset
        dataset_to_process = adult_spoken_ds

        # Extract words with both IPA and character split utterances
        words_data = []
        for example in dataset_to_process:
            ipa_words = example['ipa_transcription'].split('WORD_BOUNDARY')
            char_words = example['processed_gloss'].split(' ')
            
            # Pair up IPA and character words (they should be the same length)
            for i, ipa_word in enumerate(ipa_words):
                if ipa_word.strip():
                    char_word = char_words[i] if i < len(char_words) else ""
                    words_data.append({
                        'ipa_word': ipa_word.strip(),
                        'char_word': char_word.strip()
                    })
        
        print(f"Extracted {len(words_data)} words total")
        return words_data
    else:
        print("Dataset is None")
        return None

class PhonemeProcessor:
    """Class to handle classifying vowels and extracting valid onsets."""
    
    def load_vowels(self, words):
        # Clean words by removing spaces
        cleaned_words = [word.replace(' ', '') for word in words]
        
        # Get all vowels from PHONEME_MAPPING
        ipa_vowels = PHONEME_MAPPING['Vowels']
        
        diphthongs = set()
        
        for word in cleaned_words:
            # Look for sequences of 2 vowels in dataset
            for i in range(len(word) - 1):
                # Check for diphthongs
                if word[i] in ipa_vowels and word[i+1] in ipa_vowels:
                    diphthongs.add(word[i:i+2])
        
        # Combine all vowel units (diphthongs + single vowels)
        vowels = diphthongs.union(ipa_vowels)
        
        # Sort vowels by length (longest first) to match diphthongs before single vowels
        sorted_vowels = sorted(vowels, key=len, reverse=True)
        
        return vowels, sorted_vowels
    
    # Alternative function that reads diphthongs from a file
    def load_diphthongs_from_file(self, file_path, words):
        """Loads diphthongs from a txt file of diphongs."""
        # Clean words by removing spaces
        cleaned_words = [word.replace(' ', '') for word in words]
        
        # Get all vowels from PHONEME_MAPPING
        ipa_vowels = PHONEME_MAPPING['Vowels']
        
        diphthongs = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    diphthong = line.strip()
                    if diphthong and len(diphthong) > 1:  # Only consider valid diphthongs
                        diphthongs.add(diphthong)
                        
            # Combine all vowel units (diphthongs + single vowels)
            vowels = diphthongs.union(ipa_vowels)
            # Sort vowels by length (longest first) to match diphthongs before single vowels
            sorted_vowels = sorted(vowels, key=len, reverse=True) 
            return vowels, sorted_vowels
        
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return set(), []
        
    def extract_valid_onsets(self, words, vowels):
        """Extracts valid IPA consonant onsets from a list of words based on the given vowels."""
        
        # Clean words by removing spaces
        cleaned_words = [word.replace(' ', '') for word in words]
        
        # Create regex pattern for vowels (escape special regex characters)
        escaped_vowels = [re.escape(char) for char in vowels]
        vowel_pattern = '|'.join(escaped_vowels)
        
        # Pattern to match consonant clusters at the start of words
        valid_consonant_cluster_onsets = re.compile(rf'^((?:(?!{vowel_pattern}).)+)', re.IGNORECASE)
        
        # Extract onsets that contain only IPA consonant characters
        valid_onsets = set()
        for word in cleaned_words:
            match = valid_consonant_cluster_onsets.match(word)
            if match:
                onset = match.group(1)
                # Check if onset is valid:
                # 1. Not empty
                # 2. Contains no vowels (known vowels)
                # 3. Allow any character that's not a known vowel (including undefined phonemes)
                if onset and not any(char in vowels for char in onset):
                    valid_onsets.add(onset)
        
        return valid_onsets


def syllabify_word(word, sorted_vowels, valid_onsets):
    """
    Syllabifies a single IPA word based on the Maximum Onset Principle.
    Vowels and syllabic consonants are treated as syllable nuclei.
    """
    if not word:
        return ''
    word = word.replace(' ', '')

    diacritics_to_ignore = {
        'ː', 'ˑ', 'ʰ', 'ʷ', 'ʲ', 'ˠ', 'ˤ', '̃', '̥', '̬', '̰', '̤', '̪', '̺', '̻', '̠', '̟', '̈', '̽', '̯', '˞',
        '˥', '˦', '˧', '˨', '˩'
    }
    syllabic_marker = '̩'

    # 1. Find all nuclei positions in the original word
    nuclei_positions = []
    i = 0
    while i < len(word):
        # Check for multi-character vowels (diphthongs) first
        vowel_found = None
        for vowel in sorted_vowels:
            if word[i:].startswith(vowel):
                vowel_found = vowel
                break
        
        if vowel_found:
            nuclei_positions.append((i, i + len(vowel_found), vowel_found))
            i += len(vowel_found)
            continue

        # Check for syllabic consonants
        # A syllabic consonant is a consonant followed by optional non-syllabic diacritics and then the syllabic marker.
        char = word[i]
        if char not in VOWELS and char not in diacritics_to_ignore and char != syllabic_marker:
            # Look ahead for the syllabic marker
            j = i + 1
            while j < len(word) and word[j] in diacritics_to_ignore:
                j += 1
            
            if j < len(word) and word[j] == syllabic_marker:
                # Found a syllabic consonant cluster
                end_pos = j + 1
                nuclei_positions.append((i, end_pos, word[i:end_pos]))
                i = end_pos
                continue
        
        i += 1

    if not nuclei_positions:
        return '/' + word
    if len(nuclei_positions) == 1:
        return '/' + word

    # 2. Syllabify based on nuclei positions
    syllables = []
    for nucleus_idx, (start, end, nucleus) in enumerate(nuclei_positions):
        if nucleus_idx == 0:
            onset = word[:start]
        else:
            prev_nucleus_end = nuclei_positions[nucleus_idx - 1][1]
            consonants_between = word[prev_nucleus_end:start]
            
            best_onset = ""
            if consonants_between:
                for onset_len in range(len(consonants_between), 0, -1):
                    potential_onset = consonants_between[-onset_len:]
                    if potential_onset in valid_onsets or onset_len == 1:
                        best_onset = potential_onset
                        break
            
            onset = best_onset
            coda_for_prev = consonants_between[:-len(onset)] if onset else consonants_between
            if syllables and coda_for_prev:
                syllables[-1] += coda_for_prev
        
        syllable = onset + nucleus
        
        if nucleus_idx == len(nuclei_positions) - 1:
            coda = word[end:]
            syllable += coda
        
        syllables.append(syllable)
        
    return '/' + '/'.join(syllables) if syllables else ''


def map_phonemes_to_categories(word, vowels=None, sorted_vowels=None):
    """
    Maps each phoneme in a word to its category: V, C, F, G, L, N, or X for syllabic.
    It correctly prioritizes all multi-character phonemes (vowels and consonants)
    to ensure they are mapped to a single category unit.
    """
    word = word.replace(' ', '')
    if not vowels:
        vowels = VOWELS
    if not sorted_vowels:
        sorted_vowels = sorted(vowels, key=len, reverse=True)

    # Create a master list of all multi-character phonemes, sorted longest to shortest.
    # This ensures 'tʃ' is matched before 't', and diphthongs are matched before single vowels.
    all_phonemes = set()
    for category_phonemes in PHONEME_MAPPING.values():
        all_phonemes.update(category_phonemes)
    # The provided sorted_vowels list is definitive for the current mode (auto/manual).
    all_phonemes.update(sorted_vowels)
    sorted_all_phonemes = sorted(list(all_phonemes), key=len, reverse=True)

    diacritics_to_ignore = {
        'ː', 'ˑ', 'ʰ', 'ʷ', 'ʲ', 'ˠ', 'ˤ', '̃', '̥', '̬', '̰', '̤', '̪', '̺', '̻', '̠', '̟', '̈', '̽', '̯', '˞',
        '˥', '˦', '˧', '˨', '˩'
    }
    syllabic_marker = '̩'
    
    output = []
    i = 0
    while i < len(word):
        # 1. Prioritize matching longest multi-character phonemes (vowels and consonants).
        matched = False
        for phoneme in sorted_all_phonemes:
            if len(phoneme) > 1 and word[i:].startswith(phoneme):
                # Found a multi-character phoneme. Now find its category.
                if phoneme in VOWELS: # This handles diphthongs
                    output.append('V')
                elif phoneme in FRICATIVES:
                    output.append('F')
                elif phoneme in GLIDES:
                    output.append('G')
                elif phoneme in LIQUIDS:
                    output.append('L')
                elif phoneme in NASALS:
                    output.append('N')
                elif phoneme in CONSONANTS:
                    output.append('C')
                
                i += len(phoneme)
                matched = True
                break
        if matched:
            continue

        # 2. If no multi-character match, process the current single character and its diacritics.
        start = i
        i += 1
        while i < len(word) and (word[i] in diacritics_to_ignore or word[i] == syllabic_marker):
            i += 1
        cluster = word[start:i]

        # 3. Check for a syllabic marker in the cluster.
        if syllabic_marker in cluster:
            output.append('X')
            continue

        # 4. Classify the base phoneme, ignoring diacritics.
        base_phoneme = ''.join(c for c in cluster if c not in diacritics_to_ignore and c != syllabic_marker)
        
        if base_phoneme:
            char = base_phoneme[0]
            if char in VOWELS:
                 output.append('V')
            elif char in FRICATIVES:
                output.append('F')
            elif char in GLIDES:
                output.append('G')
            elif char in LIQUIDS:
                output.append('L')
            elif char in NASALS:
                output.append('N')
            elif char in CONSONANTS:
                output.append('C')

    return "".join(output)