import warnings #沒有也沒差
import itertools  #調整不同組的點的顏色
from pathlib import Path #抓檔案路徑
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg #做gui
from matplotlib.figure import Figure #畫圖
from random import seed, random,shuffle #隨機初始值，拆testset/trainset
from tkinter import *  #做gui
from tkinter import filedialog #做gui
import tkinter.messagebox #做gui
import tkinter.ttk as ttk #做gui
import numpy as np #np.arange產生一堆點，繪製decision boundary
from datetime import datetime #產生seed
warnings.filterwarnings("ignore")

def data_processing(file):
    x = open(file,'r')
    dataset = x.read()
    x.close()
    dataset = dataset.split('\n')
    n = 0
    if dataset[-1]=='':
        del dataset[-1]
    minn = int(min(set([row[-1] for row in dataset])))
    for data in dataset:
        dataset[n] = data.split(' ')
        dataset[n] = [float(x) for x in dataset[n]]
        dataset[n] = dataset[n][0:len(dataset[n])]
        dataset[n][-1]=int(dataset[n][-1])
        dataset[n][-1] -= minn
        n += 1
    for i in range(0,int(max(set([row[-1] for row in dataset])))): #從開始分組
        if i not in set([row[-1] for row in dataset]):
            for data in dataset:
                if data[-1] > i:
                    data[-1] -=1
    n = len(dataset)*2//3
    dataset_not_shuffle = dataset.copy() #給數字辨識用的
    shuffle(dataset) #隨機生成trainset
    if n >=20:
        trainset = dataset[0:n]
        testset = dataset[n:-1]
    else:
        trainset = dataset
        testset = dataset
    return dataset,trainset,testset,dataset_not_shuffle

def initialize_network(shape):
    network = list()
    n = len(shape)
    for i in range (1,n):
        layer = [{'weight': [random() for j in range(shape[i-1] + 1)]} for j in range(shape[i])]
        network.append(layer)
    return (network)

def activate(weight, inputs):
    activation=weight[-1]
    for i in range(len(weight)-1):
        activation += weight[i]*inputs[i]
    return activation

def transfer(activation):
    output = max(0, activation)
    return output

def transfer_derivative(output):
    return 1 if output > 0 else 0

def forward_propagate(network,row): #把資料通過每一層計算出output
    inputs = row
    for layer in network:
        new_inputs=[]
        for neuron in layer:
            activation = activate(neuron['weight'], inputs)
            neuron['output']= transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))): #從最後一層往回算
        layer = network[i]
        errors= []
        if i !=len(network)-1: #hidden layer
            for j in range(len(layer)):
                error=0.0
                for neuron in network[i+1]:
                    error += (neuron['weight'][j]*neuron['delta'])
                errors.append(error)
        else: #output layer
            for j in range(len(layer)):
                neuron= layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron=layer[j]
            neuron['delta']= errors[j]* transfer_derivative(neuron['output'])

def update_weight(network,row,l_rate):
    for i in range(len(network)):
        inputs= row[:-1]
        if i!=0:
            inputs=[neuron['output'] for neuron in network[i-1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weight'][j] += l_rate *neuron['delta']* inputs[j]
            neuron['weight'][-1] += l_rate* neuron['delta'] #bias

def train_network(network,trainset,testset,dataset, l_rate, n_epoch, n_outputs ,target):
    n_input = len(dataset[0]) - 1
    datarange = [0.0 for i in range (n_input)]
    for i in range (n_input): #計算每一個維度的range->max - min
        if max(set([data[i] for data in dataset])) == min(set([data[i] for data in dataset])) :
            datarange[i] = [max(set([data[i] for data in dataset]))+1, min(set([data[i] for data in dataset]))]
        else:
            datarange[i] = [max(set([data[i] for data in dataset])),min(set([data[i] for data in dataset]))]

    for epoch in range(n_epoch):
        sum_error = 0
        testcount = 0
        traincount = 0
        for row in trainset: #正規化
            temp = row.copy()
            for i in range (n_input):
                if (datarange[i][0]-datarange[i][1]) > 10:
                    temp[i] = (temp[i]-datarange[i][1])/(datarange[i][0]-datarange[i][1])
            outputs = forward_propagate(network, temp)
            if predict(network,temp) !=temp[-1] :
                traincount += 1
            expected = [0 for i in range(n_outputs)]
            expected[temp[-1]] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weight(network, temp, l_rate)
        for row in testset:
            temp = row.copy()
            for i in range(n_input):
                if (datarange[i][0] - datarange[i][1]) > 10:
                    temp[i] = (temp[i] - datarange[i][1]) / (datarange[i][0] - datarange[i][1])
            result = predict(network,temp)
            if result != temp[-1]:
                testcount +=1
        trainset_accuracy = (len(trainset)-traincount)/len(trainset)
        testset_accuracy = (len(testset)-testcount)/len(testset)
        result_out.insert(END,'epoch=%d, lrate=%.5f, error=%.3f, testset_accuracy=%.3f, trainset_accuracy=%.3f\n' % (epoch, l_rate, sum_error, testset_accuracy ,trainset_accuracy))
        result_out.yview_moveto(1)
        result_out.update()
        if (testset_accuracy == 1 or testset_accuracy >= target)and target != -1:
            break

def start(learningrate,n_layer,epoch,target):
    seed(datetime.now())
    n_inputs = len(dataset[0]) - 1
    n_outputs = len(set([row[-1] for row in dataset]))
    shape = []
    shape.append(n_inputs)
    shape += n_layer
    shape.append(n_outputs)
    network = initialize_network(shape)
    train_network(network, trainset,testset,dataset,learningrate, epoch, n_outputs,target)
    layer_count = 1
    for layer in network:
        result_out.insert(END,"weight:"+ "layer" +' ' + str(layer_count)+"\n")
        for row in layer:
            for data in row['weight']:
                result_out.insert(END,'%.3f' % (data))
                result_out.insert(END,' ')
            result_out.insert(END,"\n")
            result_out.yview_moveto(1)
            result_out.update()
        layer_count += 1
    return network

def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

def get_parameter():
    global network
    learningrate = float(learningrate_entry.get())
    n_layer = n_layer_entry.get()
    n_layer = n_layer.split(',')
    n_layer = list(map(int,n_layer))
    epoch = int(epoch_entry.get())
    target = float(target_entry.get())
    if target>1:
        target =1
    network = start(learningrate,n_layer,epoch,target)

def get_path():
    global dataset,trainset,testset,dataset_not_shuffle
    filename = filedialog.askopenfilename()
    showfilapath.config(text=Path(filename).stem)
    dataset,trainset,testset,dataset_not_shuffle= data_processing(filename)

def draw():
    n_input = len(dataset[0]) - 1
    datarange = [0.0 for i in range(n_input)]
    for i in range(n_input):
        if max(set([data[i] for data in dataset])) == min(set([data[i] for data in dataset])):
            datarange[i] = [max(set([data[i] for data in dataset]))+1, min(set([data[i] for data in dataset]))]
        else:
            datarange[i] = [max(set([data[i] for data in dataset])), min(set([data[i] for data in dataset]))]
    if len(dataset[0]) ==3:
        n_outputs = len(set([row[-1] for row in dataset]))
        minx = min(set([row[0] for row in dataset]))
        maxx = max(set([row[0] for row in dataset]))
        miny = min(set([row[1] for row in dataset]))
        maxy = max(set([row[1] for row in dataset]))
        xd = (maxx-minx)/150
        yd = (maxy-miny)/150
        f.clear(True)
        a = f.add_subplot(1,1,1)
        x = np.arange(minx-5*xd, maxx+5*xd, xd)
        y = np.arange(miny-5*yd, maxy+5*yd,yd)
        z = []
        for j in (y):
            contourf = []
            for i in (x):
                normaldata = [i,j]
                if datarange[0][0] - datarange[0][1] > 10:
                    normaldata[0] = (i - datarange[0][1])/(datarange[0][0] - datarange[0][1])
                    if normaldata[0]>1:
                        normaldata[0] = 1
                    if normaldata[0] < 0:
                        normaldata[0] = 0
                if datarange[1][0] - datarange[1][1] > 10:
                    normaldata[1] = (j - datarange[1][1])/(datarange[1][0] - datarange[1][1])
                    if normaldata[1]>1:
                        normaldata[1] = 1
                    if normaldata[1] < 0:
                        normaldata[1] = 0
                temp = predict(network,normaldata)
                contourf.append(temp)
            z.append(contourf)
        z = np.array(z)
        a.contourf(x,y,z)
        graph = [[] for i in range(n_outputs)]
        color_cycle = itertools.cycle(["black", "brown", "blue", "magenta", "red", "pink", "yellow", "green"])
        for data in dataset:
            graph[data[2] - 1].append([data[0], data[1]])
        for i in range(n_outputs):
            x1 = [j[0] for j in graph[i]]
            x2 = [j[1] for j in graph[i]]
            a.scatter(x1, x2,color=next(color_cycle),s=100)

        canvas = FigureCanvasTkAgg(f, master=win)
        canvas.draw()
        canvas.get_tk_widget().place(x=0, y=0)
    elif len(dataset[0]) == 4:
        n_outputs = len(set([row[-1] for row in dataset]))
        graph = [[] for i in range(n_outputs)]
        f.clear(True)
        a = f.add_subplot(1,1,1, projection='3d')
        color_cycle = itertools.cycle(["black", "brown", "blue", "magenta", "red", "pink", "yellow", "green"])
        for data in dataset:
            graph[data[3] - 1].append([data[0],data[1],data[2]])
        for i in range(n_outputs):
            x1 = [j[0] for j in graph[i]]
            x2 = [j[1] for j in graph[i]]
            x3 = [j[2] for j in graph[i]]
            a.scatter(x1, x2, x3, color=next(color_cycle), s=100)
            canvas = FigureCanvasTkAgg(f, master=win)
            canvas.draw()
            canvas.get_tk_widget().place(x=0, y=0)
    else:
        tkinter.messagebox.showinfo(title=None, message="超過三維")

def number():
    if len(dataset[0]) == 26:
        numbern = int(Number_box.get())
        data = dataset_not_shuffle[numbern][0:25]
        x1 = []
        x2 = []
        for i in range(25):
            if data[i] == 1.0:
                x1.append(1-(i//5) * 0.25)
                x2.append((i%5) * 0.25)
        f.clear(True)
        a = f.add_subplot(1, 1, 1)
        a.scatter(x2, x1,s=200)
        canvas = FigureCanvasTkAgg(f, master=win)
        canvas.draw()
        canvas.get_tk_widget().place(x=0, y=0)
        result = predict(network,data)
        tkinter.messagebox.showinfo(title=None, message="辨識結果" + str(result))

    else:
        tkinter.messagebox.showinfo(title=None, message="這不是數字測試")

win = Tk()
win.title("HW1")
win.geometry("1000x800")
win.config(bg="#323232")

n_layer_lable = Label(text="網路形狀", fg="white", bg="#323232")
n_layer_lable.pack(anchor=NE)
n_layer_entry = Entry()
n_layer_entry.insert(0, "5")
n_layer_entry.pack(anchor=NE)

learningrate_lable = Label(text="learning rate", fg="white", bg="#323232")
learningrate_lable .pack(anchor=NE)
learningrate_entry = Entry()
learningrate_entry.insert(0, "0.01")
learningrate_entry.pack(anchor=NE)

epoch_lable = Label(text="epoch", fg="white", bg="#323232")
epoch_lable .pack(anchor=NE)
epoch_entry = Entry()
epoch_entry.insert(0, "200")
epoch_entry.pack(anchor=NE)

target_lable = Label(text="target accuracy or -1", fg="white", bg="#323232")
target_lable .pack(anchor=NE)
target_entry = Entry()
target_entry.insert(0, "-1")
target_entry.pack(anchor=NE)

getfile_btn = Button(win, text='選擇檔案', command=get_path)
getfile_btn.pack(anchor=NE)
showfilapath = Label(text="", fg="white", bg="#323232")
showfilapath.pack(anchor=NE)
starttrain_btn = Button(text="開始訓練", command=get_parameter)
starttrain_btn.pack(anchor=NE)

draw_btn = Button(text="作圖", command=draw)
draw_btn.pack(anchor=NE)
result_out = Text(win, height=15,width=140)
result_out.pack(side=BOTTOM)

number_lable = Label(text="要辨識的數字", fg="white", bg="#323232")
number_lable.pack(anchor=NE)
Number_box = ttk.Combobox(win, value=['0','1','2','3'],height=5,width=18)
Number_box.pack(anchor=NE)
number_btn = Button(text="數字辨識", command=number)
number_btn.pack(anchor=NE)

f = Figure(figsize=(17,12), dpi=50)
canvas = FigureCanvasTkAgg(f, master=win)
canvas.draw()
canvas.get_tk_widget().place(x=0, y=0)
win.resizable(width=False, height=False)
win.mainloop()