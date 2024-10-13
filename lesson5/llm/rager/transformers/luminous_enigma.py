from typing import List, Tuple, Union

import numpy as np
import spacy

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def spacy_embeddings(
    document_data: Tuple[str, str, str, List[str]],
    *args,
    **kwargs,
) -> Tuple[str, str, str, List[str], List[Union[float, int]]]:
    """
    Generate embeddings using SpaCy models.

    Args:
        document_data (Tuple[str, str, str, List[str]]):
            Tuple containing document_id, document_content, chunk_text, and tokens.

    Returns:
        Tuple[str, str, str, List[str], List[Union[float, int]]]:
            Tuple containing document_id, document_content, chunk_text, tokens, and embeddings.
    """
    document_id, document_content, chunk_text, tokens = document_data
    model_name = kwargs['model_name']

    # Load SpaCy model
    nlp = spacy.load(model_name)

    # Combine tokens back into a single string of text used for embedding
    text = ' '.join(tokens)
    doc = nlp(text)

    # Average the word vectors in the doc to get a general embedding
    embeddings = np.mean([token.vector for token in doc], axis=0).tolist()

    return document_id, document_content, chunk_text, tokens, embeddings


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'