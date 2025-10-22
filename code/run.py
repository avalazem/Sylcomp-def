#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python script to run syllable analysis pipeline per desired language in IPA_CHILDES dataset

Usage:
    python run.py --language {language} [--manual]

From Ali Al-Azem avalazem@gmail.com 2025-07

"""
import argparse
from main import process_language

# Available languages in the IPA_CHILDES dataset
AVAILABLE_LANGUAGES = ['EnglishNA', 'EnglishUK', 'French', 'German', 'Spanish', 'Dutch', 'Mandarin',
'Japanese', 'Cantonese', 'Estonian', 'Croatian', 'Danish', 'Basque', 'Hungarian', 
'Turkish', 'Farsi', 'Icelandic', 'Indonesian', 'Irish', 'Welsh', 'Korean', 'Swedish', 'Norwegian',
'Quechua', 'Catalan', 'Italian', 'PortuguesePt', 'PortugueseBr', 'Romanian', 'Serbian', 'Polish'
]


def main():
    """Main function to process language(s)"""
    parser = argparse.ArgumentParser(
        description='Process IPA-CHILDES dataset for syllable complexity analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available languages:
{', '.join(AVAILABLE_LANGUAGES)}
        Pipeline for generating syllable analysis CSV files

Examples:
  python run.py --language Mandarin
  python run.py --language Japanese --manual
  python run.py -l EnglishNA -m
        """
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Process all available languages'
    )
    
    parser.add_argument(
        '--language', '-l',
        type=str,
        choices=AVAILABLE_LANGUAGES,
        default='Mandarin',
        help='Language to process (default: Mandarin)'
    )
    
    parser.add_argument(
        '--manual', '-m',
        action='store_true',
        help='Use manual diphthong loading from file instead of automatic detection'
    )
    
    parser.add_argument(
        '--list-languages',
        action='store_true',
        help='List all available languages and exit'
    )
    
    args = parser.parse_args()
    
    # List languages if requested
    if args.list_languages:
        print("Available languages:")
        for lang in AVAILABLE_LANGUAGES:
            print(f"  {lang}")
        return
    
    # Process all languages or the selected language
    if args.all:
        print(f"Processing all {len(AVAILABLE_LANGUAGES)} languages...")
        print(f"Manual diphthong loading: {'Yes' if args.manual else 'No'}")
        
        for i, language in enumerate(AVAILABLE_LANGUAGES, 1):
            print(f"\n{'='*60}")
            print(f"Processing language {i}/{len(AVAILABLE_LANGUAGES)}: {language}")
            print(f"{'='*60}")
            
            try:
                process_language(language, manually_load_diphthongs=args.manual)
            except Exception as e:
                print(f"  - ✗ Error processing {language}: {e}")

    else:
        # Process the selected language
        print(f"Processing language: {args.language}")
        print(f"Manual diphthong loading: {'Yes' if args.manual else 'No'}")
        
        process_language(args.language, manually_load_diphthongs=args.manual)
    
if __name__ == "__main__":
    main()

