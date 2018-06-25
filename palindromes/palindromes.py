#!/usr/bin/env python3

import string

#import numpy as np
#import tensorflow as tf

alphabet = string.ascii_uppercase + string.ascii_lowercase
punctuation = "'\",.; :?!"

def is_palindrome(text: str) -> bool:
    def checkpal(text: str) -> bool:
        numchars = len(text)
        if numchars <= 1:
            return True
        return (text[0] == text[numchars-1]) and checkpal(text[1:numchars-1])
    cleaned = text.translate({ord(c): None for c in punctuation})
    return checkpal(cleaned.lower())

def declare(text: str):
    print("'%s' is %s palindrome" % (text, 'a' if is_palindrome(text) else 'not a'))



declare("")
declare("ryan")
declare("maam")
declare("i am ai")
declare("a man a plan a canal panama")
declare("'Reviled did I live,' said I, 'as evil I did deliver.'")



