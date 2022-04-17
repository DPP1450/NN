import numpy as np
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import tkinter.ttk as ttk
from pathlib import Path


class HOP(object):
    def __init__(self, N):
        self.N = N
        self.W = np.zeros((N, N))

    def Product(self, factor):
        Product = np.zeros((self.N, self.N))
        for i in range(0, self.N):
            Product[i] = factor[i] * factor
        return Product

    def trainOnce(self, inputArray):
        mean = float(inputArray.sum()) / inputArray.shape[0]
        self.W = self.W + \
            self.Product(inputArray - mean) / \
            (self.N * self.N) / mean / (1 - mean)
        index = range(0, self.N)
        self.W[index, index] = 0

    def hopTrain(self, stableStateList):
        stableState = np.asarray(stableStateList)
        if len(stableState.shape) == 1 and stableState.shape[0] == self.N:
            self.trainOnce(stableState)
        elif len(stableState.shape) == 2 and stableState.shape[1] == self.N:
            for i in range(0, stableState.shape[0]):
                self.trainOnce(stableState[i])

    def hopRun(self, inputList):
        inputArray = np.asarray(inputList)
        matrix = np.tile(inputArray, (self.N, 1))
        matrix = self.W * matrix
        ouputArray = matrix.sum(1)
        m = float(np.amin(ouputArray))
        M = float(np.amax(ouputArray))
        ouputArray = (ouputArray - m) / (M - m)
        ouputArray[ouputArray < 0.5] = 0
        ouputArray[ouputArray > 0] = 1
        return np.asarray(ouputArray)


def get_dataset(path, n):
    f = open(path, 'r')
    dataset = list()
    df = f.read()
    f.close
    tmp = list()
    cnt = 0
    for char in df:
        if char == '\n':
            cnt = (cnt + 1) % (n + 1)
            if cnt == n:
                dataset.append(tmp)
                tmp = list()
        elif cnt != n:
            if char == ' ':
                tmp.append(0)
            else:
                tmp.append(1)
    dataset.append(tmp)
    return dataset


def printFormat(vector, m):
    s = ''
    for index in range(len(vector)):
        if index % m == 0:
            s += '\n'

        if vector[index] == 0:
            s += ' '

        elif vector[index] == 1:
            s += '*'

        else:
            s += str(vector[index])

    s += '\n'

    result_out.insert(END, s)


def draw(i):
    result_out.delete("1.0", "end")
    result_out.insert(END, 'input')
    printFormat(testset[i], m)
    result_out.insert(END, 'associate')
    printFormat(hop.hopRun(testset[i]), m)


def _draw():
    i = int(draw_entry.get())
    if i < len(testset):
        draw(i)
    else:
        messagebox.showinfo('ERROR', 'index out of range')


def HOP_demo():
    size = size_entry.get()
    size = size.split(',')
    n = int(size[0])
    m = int(size[1])
    trainset = get_dataset(trainset_path, n)
    testset = get_dataset(testset_path, n)
    hop = HOP(m*n)
    hop.hopTrain(trainset)
    return testset, hop, m


def get_trainset_path():
    global trainset_path
    trainset_path = filedialog.askopenfilename()
    trainset_path_lable.config(text=Path(trainset_path).stem)


def get_testset_path():
    global testset_path
    testset_path = filedialog.askopenfilename()
    testset_path_lable.config(text=Path(testset_path).stem)


def start():
    global testset, hop, m
    testset, hop, m = HOP_demo()
    result_out.delete("1.0", "end")
    result_out.insert(
        END, 'Please enter the data you want to associate(0 ~ %d)' % (len(testset) - 1))


win = Tk()
win.title("HW1")
win.geometry("300x630")
win.resizable(width=False, height=False)

result_out = Text(win, height=27, width=41)
result_out.pack(side=BOTTOM)

trainset_path_btn = Button(text='select trainset', command=get_trainset_path)
trainset_path_btn.place(x=120, y=0)
trainset_path_lable = Label(text="", fg="black")
trainset_path_lable.place(x=120, y=30)

testset_path_btn = Button(text='select testset', command=get_testset_path)
testset_path_btn.place(x=120, y=60)
testset_path_lable = Label(text="", fg="black")
testset_path_lable.place(x=120, y=90)

size_lable = Label(text="Size (row,cloumn)", fg="black")
size_lable.place(x=90, y=120)
size_entry = ttk.Combobox(win, values=['12,9', '10,10'])
size_entry.place(x=80, y=150)

start_btn = Button(text='start traning', command=start)
start_btn.place(x=120, y=180)


draw_entry = Entry()
draw_entry.insert(0, "0")
draw_entry.place(x=90, y=210)
draw_btn = Button(text='associate', command=_draw)
draw_btn.place(x=120, y=240)

win.mainloop()
