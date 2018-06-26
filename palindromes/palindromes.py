#!/usr/bin/env python3

print("Importing...")
import string
import random

import numpy as np
import tensorflow as tf

MAXPAL=40

alphabet = string.ascii_lowercase
punctuation = "'\",.; :?!"

def cleanup(text: str) -> str:
    cleaned = text.translate({ord(c): None for c in punctuation})
    return cleaned.lower()

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
nnInput = tf.placeholder(tf.int64, [None, MAXPAL])
nnOutput = tf.placeholder(tf.float64, [None, 2])

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

HIDDEN1=64
HIDDEN2=32
HIDDEN3=16
HIDDEN4=8
wHidden1 = init_weights([MAXPAL, HIDDEN1])
wHidden2 = init_weights([HIDDEN1, HIDDEN2])
wHidden3 = init_weights([HIDDEN2, HIDDEN3])
wHidden4 = init_weights([HIDDEN3, HIDDEN4])
wOutput = init_weights([HIDDEN4, 2])

def model(X, wH1, wH2, wH3, wH4, wO):
    h = tf.nn.relu(tf.matmul(tf.cast(X,tf.float32), wH1))
    h = tf.nn.relu(tf.matmul(h, wH2))
    h = tf.nn.relu(tf.matmul(h, wH3))
    h = tf.nn.relu(tf.matmul(h, wH4))
    return tf.matmul(h, wO)


py_x = model(nnInput, wHidden1, wHidden2, wHidden3, wHidden4, wOutput)
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
    return(arr)


def strize(value:int) -> str:
    text = ""
    for i in range(0,MAXPAL-1):
        text += chr(value[i])
    return text

test = cleanup("'Reviled did I live,' said I, 'as evil I did deliver.'")
print(test)
test = intize(test)
test = np.array(test).reshape(1, MAXPAL)

with tf.Session() as sess:
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = .20)
#with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    tf.initialize_all_variables().run()
    tInput = []
    tOutput = []
    print("Generating training data...")
    for gen in range(4096):
        ispal = random.randint(0,1)
        intized = intize(gen_palindrome() if ispal else gen_nonpalindrome())
        answer = [ispal, int(not ispal)]
        tInput.append(intized)
        tOutput.append(answer)

    print("Training...")
    for epoch in range(8192):
        count = len(tInput)
        #p = np.random.permutation(range(count))
        #tInput, tOutput = tInput[p], tOutput[p]
        for i in range(count):
            rnd = random.randint(0,count-1)
            tInput[rnd], tInput[i] = tInput[i], tInput[rnd]
            tOutput[rnd], tOutput[i] = tOutput[i], tOutput[rnd]

        BATCHSIZE=128
        for start in range(0, len(tInput), BATCHSIZE):
            end = start + BATCHSIZE
            sess.run(train_op, feed_dict={nnInput: tInput[start:end], nnOutput: tOutput[start:end]})

        tscore = sess.run(predict_op, feed_dict={nnInput : test})[0]
        score = np.mean(np.argmax(tOutput, axis=1) == sess.run(predict_op, feed_dict={nnInput : tInput, nnOutput : tOutput}))
        print(epoch, score, tscore)
        if score >= 0.98:
            break

    while True:
        maybe = input("Enter a string: ")
        cleaned = cleanup(maybe)
        if len(cleaned) > 40:
            print("?? I can only handle %d alphabet characters.  Try again.\n" % MAXPAL)
            continue
        cooked = intize(cleaned)
        cooked = np.array(cooked).reshape(1, MAXPAL)
        result = sess.run(predict_op, feed_dict={nnInput : cooked})
        print("%s" % ("PALINDROME!" if (result[0] == 0) else "nope."))
        #print("%s : %s" % ("YES" if (result[0] == 1) else "NO", strize(cooked[0])))


