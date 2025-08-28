Pipeline for analyzing child-directed phonetic data from the IPA-CHILDES corpus
- The main algorithm goes through all phonetically-transcribed data from adults directed to children and separates them into individual syllables via "syllabify".
- The phonemes in these syllables are then mapped to either "V" for vowel or "C" for consonant.
- "Manual" tells the algorithm to only consider language-specific diphthongs from "datasets", and to consider any other vowel combination VV (i.e., as a separate syllable)
- If not specified, any double vowel combination will be recognized as a diphthong and therefore used as the syllable nucleus V in "syllabify".
  
# Usage:
    python run.py --language {language} [--manual]

# Available languages in the IPA_CHILDES dataset
AVAILABLE_LANGUAGES = ['EnglishNA', 'EnglishUK', 'French', 'German', 'Spanish', 'Dutch', 'Mandarin',
'Japanese', 'Cantonese', 'Estonian', 'Croatian', 'Danish', 'Basque', 'Hungarian', 
'Turkish', 'Farsi', 'Icelandic', 'Indonesian', 'Irish', 'Welsh', 'Korean', 'Swedish', 'Norwegian',
'Quechua', 'Catalan', 'Italian', 'PortuguesePt', 'PortugueseBr', 'Romanian', 'Serbian', 'Polish'
]
