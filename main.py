import pandas as pd
import re
import os
import spacy
from spacy.tokens import Token
import time

def get_texts(corpus: str):
    """Yield the files in a corpus directory."""
    with os.scandir(corpus) as files:
        for f in files:
            if f.is_file():
                yield f

def preprocess(text: str):
    """Preprocess a string (may have to modify this when we switch to XML).
    
    Args:
        text (str): a large string of text from the corpus

    Returns:
        The preprocessed text.
    """
    text = text.replace('^', '')
    text = text.replace('\n', ' ')
    text = text.replace("^\n", "")
    text = text.replace("*+", "£")
    pattern = r'r\*\<\*\d+'
    text = re.sub(pattern, "", text)
    pattern = r'\*\d+'
    text = re.sub(pattern, "", text)
    pattern = r'\* \* \[\s*.*?\s*\* \*\]'
    text = re.sub(pattern, "", text)

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
        start = max(0, help_start - window)
        end = min(len(text), help_end + window)

        examples.append({
            'text': text[start:end],
            'match_span': (help_start, help_end),
            'context_start': start
        })

    return examples
                    
def extract_object(token: Token):
    """Attempt to locate the object of HELP and return a list of the all the tokens that make up the object (i.e. the object's subtree). This is useful for counting how long the object is.

    Sometimes the object is stored as the nsubj of the complement. E.g. in "I helped the doctor save the patient", "the doctor" is not a dobj of HELP, but a nsubj of the complement. We check for this.

    Args:
        token: the HELP token.

    Returns:
        Empty list if HELP is not a verb or no object found. Otherwise returns a list of the tokens that make up the object.
    """
    if token.pos_ != "VERB": 
        return []

    # Check for direct object
    for child in token.children:
        if child.dep_ == 'dobj':
            return list(child.subtree)
        
    # Search for object stored as subject of complement
    for child in token.children:
        if child.dep_ in ('ccomp', 'xcomp'):
            for grand_child in child.children:
                if grand_child.dep_ == 'nsubj':
                    return list(grand_child.subtree)
                
    return []

def bare_vs_full(token: Token):
    """Classify an instance of HELP and to or bare infinitive
    
    Args:
        token: the HELP token.
    
    Returns:
        "TO" if to-infinitive; "BARE" if bare infinitive; None otherwise.
    """
    if token.pos_ != 'VERB':
        return None

    # Iterate over children that occur to the right of HELP
    # Doing this because if spaCy's parsing is incorrect, it can sometimes focus on tokens to the left, 
    # which are not relevant for this variable
    for child in token.rights:
        if child.dep_ in ('ccomp', 'xcomp'):

            # Check for 'that' clause (e.g. 'It helps that she has a car')
            if any(c.dep_ == 'mark' and c.lemma_ == 'that' for c in child.children):
                continue
            
            # If we find a TO, classify accordingly
            has_to = [child for child in child.children if child.text == 'to']
            return "TO" if has_to else "BARE"
        
    return None
    
def verb_lemma(token: Token):
    """Return the lemma of the verb in HELP's complement."""
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
    if any(child.dep_ == 'neg' for child in token.children):
        return "NEG"
    return "POS"

def get_voice(token: Token):
    """Classify the voice of the HELP instance."""
    if any(child.dep_ in ('nsubjpass', 'auxpass') for child in token.children):
        return "Passive"
    return "Active"

def horror_aequi(token: Token):
    """Return True if TO occurs before HELP, else False."""
    if token.i > 0:
        prev_token = token.doc[token.i - 1]
        return prev_token.text.lower() == 'to'
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

    ### MAIN LOGIC ###

    # Init output filename and empty list to store results
    output_file = 'kwic_help.csv'
    results = []

    # Load spaCy Language object (disabled named-entity recognition to speed things up, but we might need this for animacy)
    nlp = spacy.load('en_core_web_lg', disable=['ner'])

    # Start timer (optional)
    print("Processing files...")
    start = time.time()

    # Loop over files in the corpus directory
    for file in get_texts('lob_corpus'):

        with open(file) as f:
            text = f.read()

        # Preprocess and find instances of HELP
        cleaned_text = preprocess(text)
        examples = find_helps(cleaned_text, window=100)

        # Tokenise, POS-tag, dependency parse, etc
        # Using nlp.pipe basically for batch processing - much faster when we get lots of texts
        docs = list(nlp.pipe([e['text'] for e in examples]))

        # Loop over each chunk containing HELP
        for doc, meta in zip(docs, examples):
            m_start, m_end = meta['match_span']

            for token in doc:

                # DUPLICATE PROTECTION:
                # Calculate the global start of every index to find the one that matches the target HELP instance 
                token_global_start = token.idx + meta['context_start']

                # If token's global position is the same as the HELP instance we're targetting...
                if m_start <= token_global_start < m_end:

                    # Double-check we're on a HELP token for safety
                    if token.lemma_ == 'help':

                        obj = extract_object(token)

                        result = {
                            'KWIC': get_kwic(token, window=100),
                            'DepVar': bare_vs_full(token),
                            'HelpPOS': token.tag_,
                            'Voice': get_voice(token),
                            'HorrorAequi': horror_aequi(token),
                            'Polarity': get_polarity(token),
                            'VerbLemma': verb_lemma(token),
                            'ObjPresent': True if obj else False,
                            'ObjectLength': len(obj) if obj else None,
                            'IntervWords': count_intervening(token),
                            'Genre': file.name,
                        }
                        results.append(result)
                        break
    
    # Print how long the operation took
    diff = time.time() - start
    print(f"Processed {len(results)} instances of help in {diff:.2f} seconds")

    # Save CSV file
    df = pd.DataFrame(results)
    df.insert(0, 'Hit', range(1, len(df)+1)) # Hit column
    df.to_csv('refactor_test.csv', index=False)
