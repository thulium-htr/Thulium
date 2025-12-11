from typing import List
import editdistance

def cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER).
    CER = (S + D + I) / N
    """
    if not reference:
        return 1.0 if hypothesis else 0.0
    
    dist = editdistance.eval(reference, hypothesis)
    return float(dist) / len(reference)

def wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER).
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if not ref_words:
        return 1.0 if hyp_words else 0.0

    dist = editdistance.eval(ref_words, hyp_words)
    return float(dist) / len(ref_words)

def ser(reference: str, hypothesis: str) -> float:
    """
    Sequence Error Rate (SER).
    1 if strings differ, 0 otherwise.
    """
    return 1.0 if reference != hypothesis else 0.0
