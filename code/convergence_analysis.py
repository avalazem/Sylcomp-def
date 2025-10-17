import argparse
import random
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os

from utils import load_ipa_childes_dataset, PhonemeProcessor, map_phonemes_to_categories
from run import AVAILABLE_LANGUAGES

# Function to run convergence analysis

def run_convergence_analysis(language, manually_load_diphthongs, step_size=100, max_words=None):
    """
    Performs a convergence analysis to determine data stability.
    """
    print(f"Starting convergence analysis for {language}...")

    # 1. Load the entire dataset
    words_data, _ = load_ipa_childes_dataset(language, speaker_specific=False)
    if not words_data:
        print(f"No data found for {language}. Exiting.")
        return

    # 2. Randomly shuffle the data
    random.shuffle(words_data)
    print(f"Loaded and shuffled {len(words_data)} words.")

    if max_words:
        words_data = words_data[:max_words]
        print(f"Limiting analysis to a maximum of {max_words} words.")

    # Determine step size if not provided by the user
    if step_size is None:
        # Set step size to 1/10th of the total words, ensuring it's at least 1
        step_size = max(1, round(len(words_data) / 10))
        print(f"Step size not provided. Defaulting to 1/10th of word count: {step_size}")

    # Initialize phoneme processor and load vowels/diphthongs
    processor = PhonemeProcessor()
    ipa_words = [item['ipa_word'] for item in words_data]

    if manually_load_diphthongs:
        diphthong_path = f"../datasets/{language}/{language}-Diphthong.txt"
        vowels, sorted_vowels = processor.load_diphthongs_from_file(diphthong_path, ipa_words)
    else:
        vowels, sorted_vowels = processor.load_vowels(ipa_words)

    if not vowels:
        print("Error: No vowels were loaded. Cannot proceed with analysis.")
        return

    print(f"Successfully loaded {len(vowels)} vowel units. Starting processing...")
    print(f"Will report progress every {step_size} words.")

    # 3. Process data in increasing chunks
    results = []
    all_phonemes = []
    
    for i in range(len(words_data)):
        word_ipa = words_data[i]['ipa_word']
        phoneme_pattern = map_phonemes_to_categories(word_ipa, vowels, sorted_vowels)
        all_phonemes.extend(list(phoneme_pattern))

        # At each step_size, calculate and store proportions
        if (i + 1) % step_size == 0 or (i + 1) == len(words_data):
            num_words = i + 1
            counts = Counter(all_phonemes)
            total_phonemes = sum(counts.values())
            
            if total_phonemes > 0:
                proportions = {phoneme: count / total_phonemes for phoneme, count in counts.items()}
                proportions['num_words'] = num_words
                results.append(proportions)
                print(f"  - Processed {num_words} words. Vowel proportion: {proportions.get('V', 0):.4f}")

    # 4. Save results to DataFrame
    if not results:
        print("No results were generated. Cannot create CSV or plot.")
        return
        
    df = pd.DataFrame(results).fillna(0)
    df = df.set_index('num_words')
    
    # Define output path
    mode_suffix = "_manual" if manually_load_diphthongs else "_auto"
    output_dir = os.path.join(os.path.dirname(os.getcwd()), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    plot_path = os.path.join(output_dir, f'{language}_convergence{mode_suffix}.png')
  
    # 5. Generate and save the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for column in df.columns:
        ax.plot(df.index, df[column], label=column)
        
    ax.set_title(f'Phoneme Proportion Convergence for {language} ({mode_suffix.strip("_")})')
    ax.set_xlabel('Number of Words Processed')
    ax.set_ylabel('Proportion')
    ax.legend(title='Phoneme Category')
    ax.grid(True)
    
    plt.savefig(plot_path)
    print(f"Convergence plot saved to {plot_path}")
    plt.close()


# Run the analysis based on command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run convergence analysis on IPA-CHILDES data.', formatter_class=argparse.RawTextHelpFormatter)
    
    # Group for mutually exclusive language selection
    lang_group = parser.add_mutually_exclusive_group(required=True)
    lang_group.add_argument('-l', '--language', type=str, choices=AVAILABLE_LANGUAGES, help='A single language to analyze.')
    lang_group.add_argument('-a', '--all', action='store_true', help='Process all available languages.')

    parser.add_argument('-m', '--manual', action='store_true', help='Use manual diphthong loading.')
    parser.add_argument('--step-size', type=int, default=None, help='Number of words to add at each step. Defaults to 1/10th of the total words.')
    parser.add_argument('--max-words', type=int, default=None, help='Maximum number of words to process for the analysis.')
    
    args = parser.parse_args()

    # Determine which languages to process
    languages_to_process = AVAILABLE_LANGUAGES if args.all else [args.language]


    # Loop through the selected languages and run the analysis
    for lang in languages_to_process:
        if len(languages_to_process) > 1:
            print(f"\n{'='*60}")
            print(f"Processing language: {lang}")
            print(f"{'='*60}")
        
        try:
            run_convergence_analysis(
                language=lang, 
                manually_load_diphthongs=args.manual, 
                step_size=args.step_size, 
                max_words=args.max_words
            )
            if len(languages_to_process) > 1:
                print(f"✓ Successfully processed {lang}")
        except Exception as e:
            print(f"  - ✗ Error processing {lang}: {e}")