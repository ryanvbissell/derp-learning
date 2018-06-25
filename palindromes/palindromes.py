#!/usr/bin/env python3

import string
import random

#import numpy as np
#import tensorflow as tf

alphabet = string.ascii_lowercase
punctuation = "'\",.; :?!"

def is_palindrome(text: str) -> bool:
    def _checkpal(text: str) -> bool:
        numchars = len(text)
        if numchars <= 1:
            return True
        return (text[0] == text[numchars-1]) and _checkpal(text[1:numchars-1])
    cleaned = text.translate({ord(c): None for c in punctuation})
    return _checkpal(cleaned.lower())

def declare(text: str):
    print("'%s' is %s palindrome" % (text, 'a' if is_palindrome(text) else 'not a'))


declare("")
declare("ryan")
declare("maam")
declare("i am ai")
declare("a man a plan a canal panama")
declare("'Reviled did I live,' said I, 'as evil I did deliver.'")


def randchar(options:str) -> str:
    length = len(options)
    choice = random.randint(0,length-1)
    return options[choice]

def randstring(options:str, length:int) -> str:
    return ''.join(random.choice(options) for _ in range(length))

def gen_nonpalindrome() -> str:
    while True:
        length = random.randint(1,200)
        text = randstring(alphabet, length)
        if not is_palindrome(text):
            return text

def gen_palindrome() -> str:
    '''
    def _punctuate(text:str) -> str:
        punct = randchar(punctuation)
        if 1 == random.randint(0,1):
            text += punct
        else:
            text = punct + text
        return text
    '''
    text = "" if 0 == random.randint(0,1) else randchar(alphabet)
    count = random.randint(0,200)
    while count > 1:
        '''
        if 0 == random.randint(0,15):
            text = _punctuate(text)
            count -= 1
        '''
        if count > 1:
            char = randchar(alphabet)
            text = char + text + char
            count -= 2
    return text

positive = gen_palindrome();
negative = gen_nonpalindrome();

print("\n'%s' IS %s PALINDROME (expected %s)" % (positive, "A" if is_palindrome(positive) else "NOT A", "positive"))
print("\n'%s' IS %s PALINDROME (expected %s)" % (negative, "A" if is_palindrome(negative) else "NOT A", "negative"))

