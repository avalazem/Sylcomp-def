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
    'Liquids': {'l','ɫ','ɭ','ʎ','ʟ','ɹ','ɻ','r','ɾ','ɽ','ᴚ'},
    'Fricatives': {'f','v','θ','ð','s','z','ʃ','ʒ','h','ɦ','ɸ','β','ʂ','ʐ','ɕ','ʑ','x','χ','ʁ','ç','ʝ'},
    'Vowels': {'i','y','ɨ','ʉ','ɯ','u','ɪ','ʏ','ʊ','e','ø','ɘ','ɵ','ɤ','o','ə','ɛ','œ','ɜ','ɞ','ʌ','ɔ','æ','ɐ','a','ɑ','ɒ', 'ɚ'},
    'Consonants': {'p', 'b', 't', 'd', 'ʈ', 'ɖ', 'c', 'ɟ', 'k', 'g', 'q', 'ɢ', 'ʡ', 'ʔ', 'ɓ', 'ɗ', 'ʄ', 'ɠ', 'ʛ', 'ts', 'dz','ʙ', 'ʀ', 'd̠ʒ', 'tʃ', 'pf', 'kx', 'tʂ', 'dʐ', 'tɕ', 'dʑ', 'Ʋ','g'}
}

# Function to load the dataset from IPA-CHILDES
def load_ipa_childes_dataset(LANGUAGE):
    try: 
        print(f"Loading dataset for language: {LANGUAGE}")
        ipa_childes_ds = load_dataset("phonemetransformers/IPA-CHILDES", f"{LANGUAGE}")
        print("Dataset loaded successfully")
        print(f"Dataset keys: {list(ipa_childes_ds.keys())}")
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
            print("No adult-spoken utterances found!")
            return []
        
        # Extract words with both IPA and character split utterances
        words_data = []
        for example in adult_spoken_ds:
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
    """Syllabify a single word with support for diphthongs and maximum onset principle."""
    if not word:
        return ''
    
    # Remove spaces first
    word = word.replace(' ', '')
    
    # Define tone markers that should stay with vowels
    tone_markers = {'˥', '˦', '˧', '˨', '˩'}
    
    # Find all vowel positions first, including any following tone markers
    vowel_positions = []
    i = 0
    while i < len(word):
        vowel_found = None
        for vowel in sorted_vowels:
            if i + len(vowel) <= len(word) and word[i:i+len(vowel)] == vowel:
                vowel_found = vowel
                break
        
        if vowel_found:
            vowel_start = i
            vowel_end = i + len(vowel_found)
            
            # Include any tone markers that follow this vowel
            while vowel_end < len(word) and word[vowel_end] in tone_markers:
                vowel_end += 1
            
            # The complete vowel unit includes the vowel + any tone markers
            complete_vowel_unit = word[vowel_start:vowel_end]
            vowel_positions.append((vowel_start, vowel_end, complete_vowel_unit))
            i = vowel_end
        else:
            i += 1
    
    if not vowel_positions:
        return '/' + word  # No vowels found, return as single syllable
    
    if len(vowel_positions) == 1:
        return '/' + word  # Only one vowel, return as single syllable
    
    syllables = []
    
    for vowel_idx, (start, end, vowel) in enumerate(vowel_positions):
        syllable = ""
        
        # Add onset (consonants before this vowel)
        if vowel_idx == 0:
            # First vowel - take all consonants from beginning
            onset = word[:start]
        else:
            # Not first vowel - apply maximum onset principle
            prev_vowel_end = vowel_positions[vowel_idx - 1][1]
            consonants_between = word[prev_vowel_end:start]
            
            if len(consonants_between) == 0:
                onset = ""
            else:
                # Apply maximum onset principle
                # Try to find the longest valid onset from the consonants
                best_onset = ""
                
                # Try all possible onsets from longest to shortest
                for onset_len in range(len(consonants_between), 0, -1):
                    potential_onset = consonants_between[-onset_len:]  # Take from the end
                    if potential_onset in valid_onsets or onset_len == 1:
                        # Valid onset found or single consonant (always valid)
                        best_onset = potential_onset
                        break
                
                onset = best_onset
                
                # Add remaining consonants to previous syllable as coda
                coda_for_prev = consonants_between[:-len(onset)] if onset else consonants_between
                if syllables and coda_for_prev:
                    syllables[-1] += coda_for_prev
        
        # Add onset and vowel
        syllable = onset + vowel
        
        # Add coda if this is the last vowel
        if vowel_idx == len(vowel_positions) - 1:
            coda = word[end:]
            syllable += coda
        
        syllables.append(syllable)
    
    return '/' + '/'.join(syllables) if syllables else ''


def map_phonemes_to_categories(word, vowels=None, sorted_vowels=None):
    """
    Maps each phoneme in a word to its category using the first letter of the category name
    Handles complex IPA phonemes with diacritics and multi-character phonemes
    Only phonemes explicitly defined in PHONEME_MAPPING are mapped to their categories
    Diphthongs are mapped as single 'V' units, not 'VV'
    Undefined phonemes are mapped to ''
    
    Args:
        word (str): A word containing IPA phonemes
        vowels (set, optional): Set of vowels including diphthongs
        sorted_vowels (list, optional): List of vowels sorted by length (longest first)
        
    Returns:
        str: The word with each phoneme replaced by its category letter
    """
    # Create a mapping from phoneme to category letter
    phoneme_to_category = {}
    
    for category, phonemes in PHONEME_MAPPING.items():
        category_letter = category[0].upper()  # First letter capitalized
        for phoneme in phonemes:
            phoneme_to_category[phoneme] = category_letter
    
    # Remove spaces first
    word = word.replace(' ', '')
    
    # Define diacritics that should be ignored or combined with base phonemes
    diacritics = {
        'ː',  # length marker
        'ˑ',  # half-length marker
        'ʰ',  # aspirated
        'ʷ',  # labialized
        'ʲ',  # palatalized
        'ˠ',  # velarized
        'ˤ',  # pharyngealized
        '̃',   # nasalized
        '̥',   # voiceless
        '̬',   # voiced
        '̰',   # creaky
        '̤',   # breathy
        '̪',   # dental
        '̺',   # apical
        '̻',   # laminal
        '̠',   # retracted
        '̟',   # advanced
        '̈',   # centralized
        '̽',   # mid-centralized
        '̩',   # syllabic
        '̯',   # non-syllabic
        '˞',  # rhotacized
        # Tone markers
        '˥',  # high tone
        '˦',  # mid-high tone
        '˧',  # mid tone
        '˨',  # mid-low tone
        '˩',  # low tone
        '˥˩', # high-low tone
        '˩˥', # low-high tone
        '˦˥', # mid-high-high tone
        '˧˥', # mid-high tone
        '˥˧', # high-mid tone
        '˧˩', # mid-low tone
        '˩˧', # low-mid tone
        '˥˧˩', # high-mid-low tone
        '˩˧˥', # low-mid-high tone
        '˧˥˩', # mid-high-low tone
        '˥˩˧', # high-low-mid tone
    }
    
    # If vowels and sorted_vowels are provided, use them to identify diphthongs
    if sorted_vowels:
        # First pass: identify and map diphthongs and vowels
        mapped_word = ""
        i = 0
        while i < len(word):
            matched = False
            
            # Try to match vowels (including diphthongs) first, longest first
            for vowel in sorted_vowels:
                if i + len(vowel) <= len(word) and word[i:i+len(vowel)] == vowel:
                    mapped_word += 'V'  # Map all vowels/diphthongs to 'V'
                    i += len(vowel)
                    matched = True
                    break
            
            if not matched:
                # Try to match other phonemes
                for length in range(min(4, len(word) - i), 0, -1):
                    substring = word[i:i+length]
                    
                    # Check if this substring is a known phoneme (non-vowel)
                    if substring in phoneme_to_category and phoneme_to_category[substring] != 'V':
                        mapped_word += phoneme_to_category[substring]
                        i += length
                        matched = True
                        break
                    
                    # Check if this is a base phoneme + diacritics
                    if length > 1:
                        base_phoneme = substring[0]
                        diacritics_part = substring[1:]
                        if (base_phoneme in phoneme_to_category and 
                            phoneme_to_category[base_phoneme] != 'V' and
                            all(char in diacritics for char in diacritics_part)):
                            # Base phoneme with diacritics - map to base phoneme category
                            mapped_word += phoneme_to_category[base_phoneme]
                            i += length
                            matched = True
                            break
            
            if not matched:
                # Check if it's a standalone diacritic (skip it)
                if word[i] in diacritics:
                    i += 1  # Skip diacritic
                else:
                    # If no match found, map to nothing for unknown phonemes
                    mapped_word += ''
                    i += 1
        
        return mapped_word
    
    else:
        # Fallback: original behavior when no vowel info is provided
        # Try to match multi-character phonemes first (longest first)
        mapped_word = ""
        i = 0
        while i < len(word):
            matched = False
            
            # Try to match progressively longer substrings (up to 4 chars to handle phoneme + diacritics)
            for length in range(min(4, len(word) - i), 0, -1):
                substring = word[i:i+length]
                
                # Check if this substring is a known phoneme
                if substring in phoneme_to_category:
                    mapped_word += phoneme_to_category[substring]
                    i += length
                    matched = True
                    break
                
                # Check if this is a base phoneme + diacritics
                if length > 1:
                    base_phoneme = substring[0]
                    diacritics_part = substring[1:]
                    if (base_phoneme in phoneme_to_category and 
                        all(char in diacritics for char in diacritics_part)):
                        # Base phoneme with diacritics - map to base phoneme category
                        mapped_word += phoneme_to_category[base_phoneme]
                        i += length
                        matched = True
                        break
            
            if not matched:
                # Check if it's a standalone diacritic (skip it)
                if word[i] in diacritics:
                    i += 1  # Skip diacritic
                else:
                    # If no match found, map to nothing for unknown phonemes
                    mapped_word += ''
                    i += 1
        
        return mapped_word