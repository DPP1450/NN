from sympy import Point, Segment, Circle
import matplotlib
import matplotlib.pyplot as plt
import pickle
from neural_network import *
from mycar import *
from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path
from tkinter import filedialog
import random
import numpy as np


def new_network():
    network = initialize_network([3, 12, 6, 1])
    f = open('model', 'wb')
    pickle.dump(network, f)
    f.close()


def train_network(learningrate, epoch, dataset, network):
    count = 1
    for _ in range(epoch):
        error = 0
        random.shuffle(dataset)
        for data in dataset:
            data_input = data[0:3]
            lable = [data[3]]
            outputs = [forward_propagate(network, data_input)[0]]
            error += (lable[0] - outputs[0])**2
            backward_propagate_error(network, lable)
            update_weight(network, data_input, learningrate)
        result_out.insert(END, 'epoch=%d, error=%.3f\n' % (count, error))
        result_out.yview_moveto(1)
        result_out.update()
        count += 1


def train():
    f = open('model', 'rb')
    network = pickle.load(f)
    f.close()
    f = open(dataset_file, 'r')
    x = f.read()
    f.close()
    x = x.split('\n')
    del x[-1]
    dataset = list()

    for i in range(len(x)):
        row = x[i].split(' ')
        temp = list()
        for i in range(4):
            temp.append(float(row[i]))
        dataset.append(temp)

    fmean = np.mean([data[0] for data in dataset])
    rmean = np.mean([data[1] for data in dataset])
    lmean = np.mean([data[2] for data in dataset])

    fsd = np.std([data[0] for data in dataset])
    rsd = np.std([data[1] for data in dataset])
    lsd = np.std([data[2] for data in dataset])

    for data in dataset:
        if data[0] > fmean*2/3:
            data[0] = 0
        else:
            data[0] = -(data[0] - fmean) / fsd

        if data[1] > rmean:
            data[1] = 0
        else:
            data[1] = -(data[1] - rmean) / rsd

        if data[2] > lmean:
            data[2] = 0
        else:
            data[2] = -(data[2] - lmean) / lsd

        data[3] = (data[3] + 40) / 80

    fmax = max([data[0] for data in dataset])
    rmax = max([data[1] for data in dataset])
    lmax = max([data[2] for data in dataset])

    for data in dataset:
        data[0] /= fmax
        data[1] /= rmax
        data[2] /= lmax

    f = open('mean&std', 'w')
    f.write("%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f" %
            (fmean, rmean, lmean, fsd, rsd, lsd, fmax, rmax, lmax))
    f.close()

    learningrate = float(learningrate_entry.get())
    epoch = int(epoch_entry.get())
    train_network(learningrate, epoch, dataset, network)
    f = open('model', 'wb')
    pickle.dump(network, f)
    f.close()


def distance(angle):
    p1 = Point(car.x, car.y)
    p2 = Point(car.x + 100*math.cos(angle), car.y + 100*math.sin(angle))
    distance = list()
    for i in range(0, len(vertex)-1):  # 所有的牆壁
        p3 = Point(vertex[i][0], vertex[i][1])
        p4 = Point(vertex[i+1][0], vertex[i+1][1])
        s1 = Segment(p1, p2)  # 車子拉出去一條線
        s2 = Segment(p3, p4)  # 牆壁
        Intersection = s1.intersection(s2)
        if Intersection != []:
            xdif = car.x-Intersection[0][0]
            ydif = car.y-Intersection[0][1]
            d = (xdif**2+ydif**2)**0.5
            distance.append(d)
    return min(distance)


def trace_data(trace_data_filename):

    f = open(trace_data_filename, 'r')
    wall = f.read()
    f.close()
    wall = wall.split('\n')
    n = 0

    for data in wall:
        wall[n] = data.split(',')
        for i in range(len(wall[n])):
            wall[n][i] = float(wall[n][i])
        n += 1

    global car
    car = Car(wall[0][0], wall[0][1], wall[0][2])

    Finish_line = [[wall[1][0], min(abs(car.y - wall[1][1]), abs(car.y - wall[2][1]))],
                   [wall[2][0], min(abs(car.y - wall[1][1]),
                                    abs(car.y - wall[2][1]))],
                   [wall[1][0], max(abs(car.y - wall[1][1]),
                                    abs(car.y - wall[2][1]))],
                   [wall[2][0], max(abs(car.y - wall[1][1]), abs(car.y - wall[2][1]))]]  # 終點線

    global vertex
    vertex = list()  # 牆壁
    for i in range(3, len(wall)):
        vertex.append(wall[i])

    car.ld = distance(car.left)
    car.rd = distance(car.right)
    car.fd = distance(car.forward)

    return Finish_line


def standardization(data):
    if data[0] > fmean*2/3:
        a = 0
    else:
        a = -(data[0] - fmean) / fsd  # dataset 平均數 and 標準差，三個sensor分別計算

    if data[1] > rmean:
        b = 0
    else:
        b = -(data[1] - rmean) / rsd

    if data[2] > lmean:
        c = 0
    else:
        c = -(data[2] - lmean) / lsd
    a /= fmax
    b /= rmax
    c /= lmax
    return [a, b, c]


def collision():
    c1 = Circle(Point(car.x, car.y), car.radius)
    flag = False
    for i in range(len(vertex)-1):
        s1 = Segment(vertex[i], vertex[i+1])
        Intersection = s1.intersection(c1)
        temp = False if len(Intersection) < 2 else True
        flag = flag or temp
    return flag


def finish():
    c1 = Circle(Point(car.x, car.y), car.radius)
    s1 = Segment(Finish_line[0], Finish_line[1])
    Intersection = s1.intersection(c1)
    if Intersection == []:
        return False
    else:
        return True


def turn_and_forward(angle):
    car.x = car.x + math.cos(car.forward+angle) + \
        math.sin(car.forward)*math.sin(angle)
    car.y = car.y + math.sin(car.forward+angle) + \
        math.sin(angle)*math.cos(car.forward)
    if collision():
        return 0
    car.forward = car.forward + math.asin(2*math.sin(angle)/6)
    car.right = car.right + math.asin(2*math.sin(angle)/6)
    car.left = car.left + math.asin(2*math.sin(angle)/6)
    car.ld = distance(car.left)
    car.rd = distance(car.right)
    car.fd = distance(car.forward)


def Draw_track():
    fig.clear()
    global a
    a = fig.add_subplot(1, 1, 1)
    a.set_aspect(1)
    a.axis('off')
    for i in range(len(vertex) - 1):
        a.plot([vertex[i][0], vertex[i + 1][0]],
               [vertex[i][1], vertex[i + 1][1]], 'black')
    a.plot([Finish_line[0][0], Finish_line[1][0]], [
           Finish_line[0][1], Finish_line[1][1]], 'red')
    a.plot([Finish_line[2][0], Finish_line[3][0]], [
           Finish_line[2][1], Finish_line[3][1]], 'red')
    draw_circle = plt.Circle((car.x, car.y), car.radius, fill=False,
                             color='red' if collision() or finish() else 'black')
    a.add_artist(draw_circle)
    canvas.draw()


def save_data():
    f1 = open('train4D.txt', 'w')
    f2 = open('train6D.txt', 'w')
    for data in trace:
        for i in range(2, 6):
            f1.write('%3.5f ' % (data[i]))
        f1.write("\n")
        for i in range(6):
            f2.write('%3.5f ' % (data[i]))
        f2.write("\n")
    f1.close()
    f2.close()


def go():
    row = standardization([car.fd, car.rd, car.ld])
    i = forward_propagate(network, row)[0] * 80 - 40
    turn_and_forward(-i / 180 * math.pi)
    draw_circle = plt.Circle((car.x, car.y), car.radius, fill=False,
                             color='red' if collision() or finish() else 'black')
    a.add_artist(draw_circle)
    canvas.draw()
    trace.append([car.x, car.y, car.fd, car.rd, car.ld, i])

    result_out.insert(END, "x=%.3f,y=%.3f,前方距離=%.3f,右前方距離=%.3f,左前方距離=%.3f,預測角度=%.3f\n" % (
        car.x, car.y, car.fd, car.rd, car.ld, i))
    result_out.yview_moveto(1)
    result_out.update()

    if finish():
        save_data()


def start():
    global network
    f = open('model', 'rb')
    network = pickle.load(f)
    f.close()

    if (car.go):
        fig.clear()
        trace_data(trace_data_filename)
        Draw_track()
        trace.clear()
        trace.append([car.x, car.y, car.fd, car.rd, car.ld, 0])
    car.go = True
    result_out.insert(END, "x=%.3f,y=%.3f,前方距離=%.3f,右前方距離=%.3f,左前方距離=%.3f,預測角度=%.3f\n" % (
        car.x, car.y, car.fd, car.rd, car.ld, 0))
    result_out.yview_moveto(1)
    result_out.update()
    global fmean, rmean, lmean, fsd, rsd, lsd, fmax, rmax, lmax
    f = open('mean&std', 'r')
    x = f.read().split(' ')
    f.close()
    fmean = float(x[0])
    rmean = float(x[1])
    lmean = float(x[2])
    fsd = float(x[3])
    rsd = float(x[4])
    lsd = float(x[5])
    fmax = float(x[6])
    rmax = float(x[7])
    lmax = float(x[8])
    while (not collision() and not finish()):
        go()


def get_trace_path():
    global trace_data_filename
    trace_data_filename = filedialog.askopenfilename()
    trace_data_path.config(text=Path(trace_data_filename).stem)
    global Finish_line, trace
    Finish_line = trace_data(trace_data_filename)
    Draw_track()
    trace = [[car.x, car.y, car.fd, car.rd, car.ld, 0]]


def get_dataset_path():
    global dataset_file
    dataset_file = filedialog.askopenfilename()
    dataset_path_lable.config(text=Path(dataset_file).stem)


win = Tk()
win.title("HW1")
win.geometry("1000x800")
win.resizable(width=False, height=False)

fig = Figure(figsize=(17, 12), dpi=50)
canvas = FigureCanvasTkAgg(fig, master=win)
canvas.draw()
canvas.get_tk_widget().place(x=0, y=0)

new_btn = Button(text='建立新的網路', command=new_network)
new_btn.place(x=890, y=0)

learningrate_lable = Label(text="learning rate", fg="black")
learningrate_lable.place(x=890, y=30)
learningrate_entry = Entry()
learningrate_entry.insert(0, "0.1")
learningrate_entry.place(x=850, y=60)

epoch_lable = Label(text="epoch", fg="black")
epoch_lable.place(x=890, y=90)
epoch_entry = Entry()
epoch_entry.insert(0, "300")
epoch_entry.place(x=850, y=120)

dataset_path_btn = Button(text='選擇dataset', command=get_dataset_path)
dataset_path_btn.place(x=890, y=150)
dataset_path_lable = Label(text="", fg="black")
dataset_path_lable.place(x=890, y=180)

train_btn = Button(text='開始訓練', command=train)
train_btn.place(x=890, y=210)

tracedata_btn = Button(text='選擇軌道檔案', command=get_trace_path)
tracedata_btn.place(x=890, y=240)
trace_data_path = Label(text="", fg="black")
trace_data_path.place(x=890, y=270)

start_btn = Button(text="start", command=start)
start_btn.place(x=895, y=300)

result_out = Text(win, height=15, width=140)
result_out.pack(side=BOTTOM)
win.mainloop()
