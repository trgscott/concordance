# Script for Corpus Linguistics

The script (`main.py`) contains a handful of very short and simple functions, followed by an if-name-main block containing the main logic. The approach relies heavily on spaCy's dependency parsing. For example, you can easily move through the dependency tree using the `.children` attribute of a token (look at https://spacy.io/usage/linguistic-features for details). This makes it really easy to find the subject/object/etc.

The script is very fast (OldBailey in ~40 seconds) because it does not parse the entire corpus at once. Instead, it finds instances of *help*, takes a chunk of the surrounding text, and only parses that chunk.

## TODO
- [ ] Add polarity indicators
- [ ] Animacy implementation
- [ ] Preprocessing 
- [x] For negation, can you rule out “not only”? … not only help but also support – POS
- [x] For ING and INING, try to change the -ing form (deciding) to the infinitive (decide) for the lemma of the head of the non-finite clause.
- [x] Add ING and INING classification of DepVar
- [x] Subject information (subject type, subject head)
- [x] Metadata variables
- [x] Object information (pronoun/noun, head of object)
- [x] Add word class of help (`.pos_`)

## Issues
If you filter the data by DepVar==BARE and sort in descending order of IntervWords, you will find a lot of bad classifications.

- Most errors arise when the dependency parsing is incorrect. Could implement fallbacks to improve this?
- `count_intervening()` is badly affected by incorrect parsing. Also still counts 'to' as an intervening word.
