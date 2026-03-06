import re
import os
import spacy
import pandas as pd
from spacy.tokens import Token
from tqdm import tqdm

def get_texts(corpus: str):
    """Return a list of the files in a corpus directory."""
    result = []
    with os.scandir(corpus) as files:
        for f in files:
            if f.is_file() and f.name.endswith('.txt'):
                result.append(f)
    
    return result

def get_metadata(filename:str, documentation: pd.DataFrame):

    # Extract FileID and strip the leading zeros
    id = filename.split('_')[0].lstrip('0')

    # Extract metadata from documentation file
    return documentation.loc[id, ['SpeakerID', 'Year', 'TrialDate', 
                                  'Gender', 'Age', 'Role', 'SocialClass1', 
                                  'SocialClass2', 'OldBaileyFile'
                                  ]]

def preprocess(text: str):
    text = re.sub(r'[^\x00-\x7F]+', '', text) # Delete weird characters like Äî
    return text

def find_helps(text:str, window: int=200):
    """Search a large string for an occurrence of help, and return the chunk of text containing that occurrence. This chunk will be used for deriving the variables, not for display.

    Args:
        text (str): a large string of text from the corpus (ideally preprocessed/cleaned).
        window (int): an integer specifying the chunk size, i.e. the number of characters to the left and right of HELP.

    Returns:
        A list of dictionaries containing the actual text chunk; the indices where help starts and ends in the entire text; the index of where the chunk starts. 
    """

    # Pattern finds unhelpful, helpfully, helps, etc
    pattern = re.compile(r"\b\w*help\w*\b", flags=re.IGNORECASE)
    examples = []

    for match in pattern.finditer(text):
        help_start, help_end = match.span() # Indices of help token

        # Define indices of context snippet
        # TODO: stop at new lines?
        start = max(0, help_start - window)
        end = min(len(text), help_end + window)

        examples.append({
            'text': text[start:end],
            'match_span': (help_start, help_end),
            'context_start': start
        })

    return examples

def get_np_tokens(obj_token: Token):
    """Return tokens of a noun phrase, exluding clausal complements and punctuation"""

    np_tokens = [obj_token]

    for child in obj_token.children:
        if child.dep_ not in ('ccomp', 'xcomp', 'advcl', 'relcl'):

            # Don't include punctuation (e.g. 'emotionally-disturbed' should not be 3 tokens)
            non_puncts = [t for t in child.subtree if not t.is_punct]
            np_tokens.extend(non_puncts)

    return np_tokens
                    
def extract_object(token: Token):
    """Attempt to locate the object of HELP and return a dictionary with the object's POS-tag (PRON or NP) and a list of the all the tokens that make up the object (i.e. the object's subtree). This is useful for counting how long the object is.

    Sometimes the object is stored as the nsubj of the complement. E.g. in "I helped the doctor save the patient", "the doctor" is not a dobj of HELP, but a nsubj of the complement. We check for this.

    Args:
        token: the HELP token.

    Returns:
        Dictionary with empty values if no object found. Otherwise, dictionary with list of words making up the object and the object's POS-tag.
    """
    result = {'words': [], 'tag': None, 'head': None}
    if token.pos_ != "VERB": 
        return result
    
    dobj = None
    comp_verb = None

    # Search for direct objects and complement verbs in one loop
    for child in token.children:
        if child.dep_ == 'dobj':
            dobj = child
        elif child.dep_ in {'ccomp', 'xcomp'}:
            comp_verb = child
    
    obj = dobj
    if not obj and comp_verb:
            # Found a complement verb and did NOT find a direct object
            for gc in comp_verb.children:
                # Subject must occur inbetween HELP (token.i) and the complement verb (comp_verb.i)
                if gc.dep_ == 'nsubj' and token.i < gc.i < comp_verb.i:
                    obj = gc
                    result['head'] = obj.text
                    break
                
    # Process object if found
    if obj:
        result['words'] = get_np_tokens(obj)
        result['tag'] = 'PRO' if obj.pos_ == 'PRON' else 'NP'

    return result

def extract_subject(token: Token):
    result = {'pos': None, 'head': None, 'animacy': None}

    for child in token.children:
        if child.dep_ == 'nsubj':
            
            if child.pos_ == 'NOUN':
                pos = 'NP'
            elif child.lower_ == 'it':
                pos = 'IT'
            elif child.pos_ == 'PRON':
                pos = 'PRO'
            else:
                return result # If pos is None, also code head as None
            result['pos'] = pos

            result['head'] = child.lemma_
            result['animacy'] = animacy(child)

    return result

def animacy(subj: Token):
    
    animate_labels = {'PERSON'}

    if subj.ent_type in animate_labels:
        return "Animate"
    
    if subj.lower_ in {'he', 'she', 'him', 'her', 'who'}:
        return "Animate"
    
    return "Inanimate"

def bare_vs_full(token: Token):
    """Classify HELP as 'TO' if 'help to VERB'; 'BARE' if 'help VERB'; 'INING' if 'help in VERBing'; 'ING' if 'help VERBing'.
    
    Args:
        token: the HELP token.
    """
    if token.pos_ != 'VERB':
        return None
    
    # Get the token after HELP
    next_token = token.doc[token.i + 1] if token.i + 1 < len(token.doc) else None
    
    for child in token.rights:
        
        # Check for INING
        if child.lemma_ == 'in' and child.dep_ == 'prep':
            if any(c.tag_ == 'VBG' for c in child.children):
                return "INING"
        
        # Check for ING
        if child == next_token and child.tag_ == 'VBG':
            return 'ING'
        
        # Check for TO and BARE
        if child.dep_ in ('ccomp', 'xcomp'):

            # Ignore 'that' clauses
            if any(c.dep_ == 'mark' and c.lemma_ == 'that' for c in child.children):
                continue
            
            # Search for 'to'
            has_to = any(c.lemma_ == 'to' and c.pos_ == 'PART' for c in child.children)
            return "TO" if has_to else "BARE"

    return None
    
def verb_lemma(token: Token):
    """Return the lemma of the complement clause."""
    if token.pos_ != 'VERB':
        return None 

    for child in token.children:
        if child.dep_ in ('ccomp', 'xcomp'):

            # Check for 'that' clause
            if any(c.dep_ == 'mark' and c.lemma_ == 'that' for c in child.children):
                continue

            return child.lemma_
    
    return None
        
def get_polarity(token: Token):
    """Classify polarity of HELP."""

    neg_words = {'not', "n't", 'nor', 'never', 'hardly', 'scarcely', 'barely', 'no', 'nobody', 'nothing', 'nowhere'}

    # Check for 'not only'
    has_neg = any(c.dep_ == 'neg' or c.text in neg_words for c in token.children)
    has_only = any(c.dep_ == 'advmod' and c.lemma_ == 'only' for c in token.children)

    if has_neg and has_only:
        return 'POS'
    elif has_neg and not has_only:
        return 'NEG'
    else:
        return 'POS'

def get_voice(token: Token):
    """Classify the voice of the HELP instance."""
    if any(child.dep_ in ('nsubjpass', 'auxpass') for child in token.children):
        return "Passive"
    return "Active"

def horror_aequi(token: Token):
    """Return True if TO occurs before HELP, else False."""
    if token.i > 0:
        prev_token = token.doc[token.i - 1]

        if prev_token.lemma_ == 'to':
            return 'YEStoBefore'
        else:
            return 'NOtoBefore'
    return False

def count_intervening(token: Token):
    """Count the number words in between HELP and the complement's verb."""
    for right in token.rights:
        if right.dep_ in ('ccomp', 'xcomp'):
            distance = abs(right.i - token.i) - 1
            return max(0, distance)
    return 0

def get_kwic(token: Token, window: int=100):
    """Take an instance of HELP and return a concordance line for display. 
    
    This function is independent from find_helps in order to (1) allow you to dependency parse a smaller chunk of the concordance line, which is quicker; (2) ensure HELP is always in the middle of the concordance line, even if multiple HELPs occur in one of the chunks we extracted.

    Args:
        token: the HELP token.
        window: an integer specifying the chunk size, i.e. the number of characters to the left and right of HELP.
    
    Returns:
        A concordance line with HELP in the middle surrounded by @ symbols.
    """
    start = token.idx
    end = start + len(token.text)

    left_context = token.doc.text[max(0, start - window):start]
    right_context = token.doc.text[end:end + window]

    return f"{left_context}@{token.text}@{right_context}"

if __name__ == "__main__":
    # Load spaCy Language object and documentation file 
    nlp = spacy.load('en_core_web_lg')
    documentation = pd.read_parquet('OldBailey/Documentation.parquet')

    # Get the files in the corpus
    files = get_texts('OldBailey/Processed files/')

    # Loop over files in the corpus directory
    results = []
    for file in tqdm(files, desc="Processing files", unit='file'):
        with open(file, encoding='utf-8') as f:
            text = f.read()

        # Preprocess and find instances of HELP
        cleaned_text = preprocess(text)
        examples = find_helps(cleaned_text, window=100)

        # If we find no examples of HELP, skip the file
        if not examples:
            continue

        # Get metatata
        metadata = get_metadata(file.name, documentation)

        # Tokenise, POS-tag, dependency parse, etc
        # Using nlp.pipe basically for batch processing - much faster when we get lots of texts
        docs = nlp.pipe([e['text'] for e in examples])

        # Loop over each chunk containing HELP
        for doc, indices in zip(docs, examples):
            m_start, m_end = indices['match_span']

            for token in doc:

                # DUPLICATE PROTECTION:
                # Calculate the global start of every index to find the one that matches the target HELP instance 
                token_global_start = token.idx + indices['context_start']

                # If token's global position is the same as the HELP instance we're targetting...
                if m_start <= token_global_start < m_end:

                    # Double-check we're on a HELP token for safety
                    if token.lemma_ == 'help':
                        
                        subj = extract_subject(token)
                        obj = extract_object(token)

                        result = {
                            'KWIC': get_kwic(token, window=100),
                            'DepVar': bare_vs_full(token),
                            'HelpClass': token.pos_,
                            'HelpInflection': token.tag_,
                            'Voice': get_voice(token),
                            'HorrorAequi': horror_aequi(token),
                            'Polarity': get_polarity(token),
                            'VerbLemma': verb_lemma(token),
                            'SubjType': subj['pos'],
                            'SubjHead': subj['head'],
                            'SubjAnimacy': subj['animacy'],
                            'ObjPresent': True if obj['words'] else False,
                            'ObjTag': obj['tag'],
                            'ObjLength': len(obj['words']) if obj['words'] else None,
                            'ObjHead': obj['head'],
                            'IntervWords': count_intervening(token),
                            'Genre': file.name,
                        }

                        # Add metadata to dict
                        result.update(metadata)

                        results.append(result)
                        break
    
    print(f"Processed {len(results)} instances of help")

    # Save CSV file
    df = pd.DataFrame(results)
    df.insert(0, 'Hit', range(1, len(df)+1)) # Hit column
    df.to_csv('help_concordance.csv', index=False)
