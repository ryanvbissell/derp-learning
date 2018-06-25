#!/usr/bin/env python3

print("Importing...")
import string
import random

import numpy as np
import tensorflow as tf

MAXPAL=128

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
        length = random.randint(1,MAXPAL)
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
    count = random.randint(0,MAXPAL)
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



print("\n\n\nTensorflowing...")
nnInput = tf.placeholder(tf.int32, [None, MAXPAL])
nnOutput = tf.placeholder(tf.float32, [None, 2])

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

HIDDEN1=10
wHidden1 = init_weights([MAXPAL, HIDDEN1])
wOutput = init_weights([HIDDEN1, 2])

def model(X, wH, wO):
    h = tf.nn.relu(tf.matmul(tf.cast(X,tf.float32), wH))
    return tf.matmul(h, wO)


py_x = model(nnInput, wHidden1, wOutput)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=nnOutput))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

predict_op = tf.argmax(py_x, 1)

def nn_declare(prediction):
    return ["NOT a", "a"][prediction]

def intize(text:str):
    arr = []
    length = len(text)
    for offset in range(length,MAXPAL):  # pad out before intizing
        text += '\0'
    for i in range(MAXPAL):
        value = ord(text[i])
        arr.append(value)
    arr = np.array(arr).reshape(1, MAXPAL)
    print(arr)
    return(arr)


def strize(value:int) -> str:
    text = ""
    for i in range(0,MAXPAL-1):
        text += chr(value[i])
    return text


with tf.Session() as sess:
    tf.initialize_all_variables().run()
    print("Training on 1000 samples...")
    for sample in range(1,1000):
        ispal = random.randint(0,1)
        text = gen_palindrome() if ispal else gen_nonpalindrome()
        answer = [float(ispal), float(not ispal)]
        answer = np.array(answer).reshape(1,2)
        intized = intize(text)
        sess.run(train_op, feed_dict={nnInput : intized, nnOutput : answer})
        #print(text, answer, np.mean(np.argmax(answer, axis=1) == sess.run(predict_op, feed_dict={nnInput : intized, nnOutput : answer})))
    while True:
        maybe = input("Enter a string:")
        cooked = intize(maybe)
        result = sess.run(predict_op, feed_dict={nnInput : cooked})
        print(cooked)
        print(result)
        print("%s : %s" % ("YES" if result[0] else "NO", strize(cooked[0])))

