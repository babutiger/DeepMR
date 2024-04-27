import os
from copy import deepcopy
import numpy as np
import time

import sys
base_dir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path[0] = base_dir

from utils.util import save_radius_result, save_number_result

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    # def write(self, message):
    #     self.terminal.write(message)
    #     self.log.write(message)

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

# Get the filename of the current script,Remove the file extension from the filename
script_filename = os.path.basename(__file__)
script_name_without_extension = os.path.splitext(script_filename)[0]

style_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
sys.stdout = Logger("../../result/log/"+script_name_without_extension+"_log_" + str(style_time) + ".txt", sys.stdout)
# sys.stderr = Logger("../log/a_err_" + str(style_time) + ".txt", sys.stderr)  # redirect std err, if necessary

class neuron(object):
    """
    Attributes:
        algebra_lower (numpy ndarray of float): neuron's algebra lower bound(coeffients of previous neurons and a constant)
        algebra_upper (numpy ndarray of float): neuron's algebra upper bound(coeffients of previous neurons and a constant)
        concrete_algebra_lower (numpy ndarray of float): neuron's algebra lower bound(coeffients of input neurons and a constant)
        concrete_algebra_upper (numpy ndarray of float): neuron's algebra upper bound(coeffients of input neurons and a constant)
        concrete_lower (float): neuron's concrete lower bound
        concrete_upper (float): neuron's concrete upper bound
        weight (numpy ndarray of float): neuron's weight
        bias (float): neuron's bias
        certain_flag (int): 0 uncertain 1 activated(>=0) 2 deactivated(<=0)
    """

    def __init__(self):
        self.algebra_lower = None
        self.algebra_upper = None
        self.concrete_algebra_lower = None
        self.concrete_algebra_upper = None
        self.concrete_lower = None
        self.concrete_upper = None

        self.weight = None
        self.bias = None
        self.certain_flag = 0

    def print(self):
        # print('algebra_lower:', self.algebra_lower)
        # print('algebra_upper:', self.algebra_upper)
        # print('concrete_algebra_lower:', self.concrete_algebra_lower)
        # print('concrete_algebra_upper:', self.concrete_algebra_upper)
        print('concrete_lower:', self.concrete_lower)
        print('concrete_upper:', self.concrete_upper)
        # print('weight:', self.weight)
        # print('bias:', self.bias)
        # print('certain_flag:', self.certain_flag)


class layer(object):
    """
    Attributes:
        neurons (list of neuron): Layer neurons
        size (int): Layer size
        layer_type (int) : Layer type 0 input 1 affine 2 relu
    """
    INPUT_LAYER = 0
    AFFINE_LAYER = 1
    RELU_LAYER = 2

    def __init__(self):
        self.size = None
        self.neurons = None
        self.layer_type = None

    def print(self):
        print('Layer size:', self.size)
        print('Layer type:', self.layer_type)
        print('Neurons:')
        for neu in self.neurons:
            neu.print()
            print('\n')


class network(object):
    """
    Attributes:
        numLayers (int): Number of weight matrices or bias vectors in neural network
        layerSizes (list of ints): Size of input layer, hidden layers, and output layer
        inputSize (int): Size of input
        outputSize (int): Size of output
        mins (list of floats): Minimum values of inputs
        maxes (list of floats): Maximum values of inputs
        means (list of floats): Means of inputs and mean of outputs
        ranges (list of floats): Ranges of inputs and range of outputs
        layers (list of layer): Network Layers
    """

    def __init__(self):
        self.numlayers = None
        self.layerSizes = None
        self.inputSize = None
        self.outputSize = None
        self.mins = None
        self.maxes = None
        self.ranges = None
        self.layers = None
        self.property_flag = None

    def deeppoly(self):

        def back_propagation(cur_neuron, i):
            if i == 0:
                cur_neuron.concrete_algebra_lower = deepcopy(cur_neuron.algebra_lower)
                cur_neuron.concrete_algebra_upper = deepcopy(cur_neuron.algebra_upper)
            lower_bound = deepcopy(cur_neuron.algebra_lower)
            upper_bound = deepcopy(cur_neuron.algebra_upper)
            for k in range(i + 1)[::-1]:  #倒序 i=1时，k=1 k=0
                tmp_lower = np.zeros(len(self.layers[k].neurons[0].algebra_lower))
                tmp_upper = np.zeros(len(self.layers[k].neurons[0].algebra_lower))
                for p in range(self.layers[k].size):
                    if lower_bound[p] >= 0:
                        tmp_lower += lower_bound[p] * self.layers[k].neurons[p].algebra_lower
                        # print("lower_bound[p]:", lower_bound[p])
                        # print("self.layers[k].neurons[p].algebra_lower", self.layers[k].neurons[p].algebra_lower)
                        # print("self.layers[k].neurons[p].algebra_lower[]", self.layers[k].neurons[p].algebra_lower[0])
                        # print("tmp_lower", tmp_lower)
                    else:
                        tmp_lower += lower_bound[p] * self.layers[k].neurons[p].algebra_upper

                    if upper_bound[p] >= 0:
                        tmp_upper += upper_bound[p] * self.layers[k].neurons[p].algebra_upper
                        # print("upper_bound[p]", upper_bound[p])
                        # print("self.layers[k].neurons[p].algebra_upper", self.layers[k].neurons[p].algebra_upper)
                        # print("tmp_upper", tmp_upper)
                    else:
                        tmp_upper += upper_bound[p] * self.layers[k].neurons[p].algebra_lower

                # print("===========")
                # print("lower_bound:shi",lower_bound)
                # print("lower_bound[-1]:", lower_bound[-1])
                # print("tmp_lower[-1]:", tmp_lower[-1])
                tmp_lower[-1] += lower_bound[-1]
                # print("tmp_lower[-1]:", tmp_lower[-1])

                # print("=======2")
                # print("upper_bound", upper_bound)

                # print("upper_bound[-1]:", upper_bound[-1])
                # print("tmp_upper[-1]:", tmp_upper[-1])
                tmp_upper[-1] += upper_bound[-1]
                # print("tmp_upper[-1]:", tmp_upper[-1])

                lower_bound = deepcopy(tmp_lower)  #往前传播非常重要的一步
                # print("lower_bound:", lower_bound)
                upper_bound = deepcopy(tmp_upper)
                # print("upper_bound：", upper_bound)

                if k == 1:
                    cur_neuron.concrete_algebra_lower = deepcopy(lower_bound)
                    cur_neuron.concrete_algebra_upper = deepcopy(upper_bound)

            assert (len(lower_bound) == 1)
            assert (len(upper_bound) == 1)
            cur_neuron.concrete_lower = lower_bound[0]
            cur_neuron.concrete_upper = upper_bound[0]

            # print("cur_neuron.concrete_lower:", cur_neuron.concrete_lower)
            # print("cur_neuron.concrete_upper:", cur_neuron.concrete_upper)

        for i in range(len(self.layers) - 1):
            # print('i=',i)
            pre_layer = self.layers[i]
            cur_layer = self.layers[i + 1]
            pre_neuron_list = pre_layer.neurons
            cur_neuron_list = cur_layer.neurons

            if cur_layer.layer_type == layer.AFFINE_LAYER:
                for j in range(cur_layer.size):
                    cur_neuron = cur_neuron_list[j]
                    cur_neuron.algebra_lower = np.append(cur_neuron.weight, [cur_neuron.bias])
                    cur_neuron.algebra_upper = np.append(cur_neuron.weight, [cur_neuron.bias])
                    # note: layer index of cur_neuron is i + 1, so pack propagate form i
                    back_propagation(cur_neuron, i)

            elif cur_layer.layer_type == layer.RELU_LAYER:
                for j in range(cur_layer.size):
                    cur_neuron = cur_neuron_list[j]
                    pre_neuron = pre_neuron_list[j]

                    if pre_neuron.concrete_lower >= 0:
                        cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_lower[j] = 1
                        cur_neuron.algebra_upper[j] = 1
                        cur_neuron.concrete_algebra_lower = deepcopy(pre_neuron.concrete_algebra_lower)
                        cur_neuron.concrete_algebra_upper = deepcopy(pre_neuron.concrete_algebra_upper)
                        cur_neuron.concrete_lower = pre_neuron.concrete_lower
                        cur_neuron.concrete_upper = pre_neuron.concrete_upper
                        cur_neuron.certain_flag = 1

                    elif pre_neuron.concrete_upper <= 0:
                        cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
                        # cur_neuron.algebra_lower[j] = 0
                        # cur_neuron.algebra_upper[j] = 0   这一步在初始化的时候已经实现了，所以不需要了
                        cur_neuron.concrete_algebra_lower = np.zeros(self.inputSize)
                        cur_neuron.concrete_algebra_upper = np.zeros(self.inputSize)
                        cur_neuron.concrete_lower = 0
                        cur_neuron.concrete_upper = 0
                        cur_neuron.certain_flag = 2

                    elif pre_neuron.concrete_lower + pre_neuron.concrete_upper <= 0:
                        # print("---------11112220-------")
                        # print("pre_neuron.concrete_lower:", pre_neuron.concrete_lower)
                        # print("pre_neuron.concrete_upper:", pre_neuron.concrete_upper)
                        # nanmta = 0 , relu抽象的情况，图b

                        cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_lower[j] = 0
                        # print("cur_neuron.algebra_lower:", cur_neuron.algebra_lower)
                        alpha = pre_neuron.concrete_upper / (pre_neuron.concrete_upper - pre_neuron.concrete_lower)
                        # print("alpha:", alpha)
                        cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
                        # print("cur_neuron.algebra_upper,", cur_neuron.algebra_upper)
                        cur_neuron.algebra_upper[j] = alpha
                        # print("cur_neuron.algebra_upper[j]:", cur_neuron.algebra_upper[j])
                        # print("cur_neuron.algebra_upper:", cur_neuron.algebra_upper)
                        cur_neuron.algebra_upper[-1] = -alpha * pre_neuron.concrete_lower
                        # print("cur_neuron.algebra_upper[-1]:", cur_neuron.algebra_upper[-1])
                        # print("cur_neuron.algebra_upper:", cur_neuron.algebra_upper)
                        back_propagation(cur_neuron, i)
                        # print("---------11112223-------")

                    else:
                        # nanmta = 1时 ,relu抽象的情况,图c
                        cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_lower[j] = 1
                        alpha = pre_neuron.concrete_upper / (pre_neuron.concrete_upper - pre_neuron.concrete_lower)
                        cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
                        cur_neuron.algebra_upper[j] = alpha
                        cur_neuron.algebra_upper[-1] = -alpha * pre_neuron.concrete_lower
                        back_propagation(cur_neuron, i)

    def print(self):
        print('numlayers:', self.numLayers)
        print('layerSizes:', self.layerSizes)
        print("inputSize:", self.inputSize)
        print('outputSize:', self.outputSize)
        print('mins:', self.mins)
        print('maxes:', self.maxes)
        print('ranges:', self.ranges)
        print('Layers:')
        for l in self.layers:
            l.print()
            print('\n')

    def load_robustness(self, filename, delta, TRIM=False):
        if self.property_flag == True:
            self.layers.pop()
        self.property_flag = True

        # 读取输入图片的像素，变为区间
        with open(filename) as f:
            for i in range(self.layerSizes[0]):
                line = f.readline()
                linedata = [float(line.strip()) - delta, float(line.strip()) + delta]
                # if TRIM:
                #     if linedata[0] < 0: linedata[0] = 0
                #     if linedata[1] > 1: linedata[1] = 1

                self.layers[0].neurons[i].concrete_lower = linedata[0]
                self.layers[0].neurons[i].concrete_upper = linedata[1]
                self.layers[0].neurons[i].concrete_algebra_lower = np.array([linedata[0]])
                self.layers[0].neurons[i].concrete_algebra_upper = np.array([linedata[1]])
                self.layers[0].neurons[i].algebra_lower = np.array([linedata[0]])
                self.layers[0].neurons[i].algebra_upper = np.array([linedata[1]])

            # 增加最后的判断是不是鲁棒的两两相减层
            line = f.readline()
            verify_layer = layer()
            verify_layer.neurons = []
            while line:
                linedata = [float(x) for x in line.strip().split(' ')]
                assert (len(linedata) == self.outputSize + 1)
                verify_neuron = neuron()
                verify_neuron.weight = np.array(linedata[:-1])
                verify_neuron.bias = linedata[-1]
                verify_layer.neurons.append(verify_neuron)
                linedata = np.array(linedata)
                # print('linedata：', linedata)
                assert (len(linedata) == self.outputSize + 1)
                line = f.readline()
            verify_layer.size = len(verify_layer.neurons)
            verify_layer.layer_type = layer.AFFINE_LAYER
            if len(verify_layer.neurons) > 0:
                self.layers.append(verify_layer)
                # print("0000")

    def load_nnet(self, filename):
        with open(filename) as f:
            line = f.readline()
            cnt = 1
            while line[0:2] == "//":
                line = f.readline()
                cnt += 1
            # 读取nnet格式的一些网络参数基本信息
            # numLayers doesn't include the input layer!
            numLayers, inputSize, outputSize, _ = [int(x) for x in line.strip().split(",")[:-1]]
            # print("numLayers:", numLayers)
            # print("inputSize:", inputSize)
            # print("outputSize:", outputSize)

            line = f.readline()

            # input layer size, layer1size, layer2size...
            layerSizes = [int(x) for x in line.strip().split(",")[:-1]]
            # print("layerSizes:",layerSizes)
            # print("layerSizeslen:",len(layerSizes))

            line = f.readline()
            symmetric = int(line.strip().split(",")[0])
            line = f.readline()
            inputMinimums = [float(x) for x in line.strip().split(",")[:-1]]
            # print("inputMinimums:", inputMinimums)
            # print("inputMinimumslen:", len(inputMinimums))
            line = f.readline()
            inputMaximums = [float(x) for x in line.strip().split(",")[:-1]]
            # print("inputMaximums:", inputMaximums)
            # print("inputMaximumslen:", len(inputMaximums))
            line = f.readline()
            inputMeans = [float(x) for x in line.strip().split(",")[:-1]]
            # print("inputMeans:", inputMeans)
            # print("inputMeans:", len(inputMeans))
            line = f.readline()
            inputRanges = [float(x) for x in line.strip().split(",")[:-1]]
            # print("inputRanges:", inputRanges)
            # print("inputRanges:", len(inputRanges))

            # process the input layer 处理输入层
            self.layers = []
            new_layer = layer()
            new_layer.layer_type = layer.INPUT_LAYER
            # print("new_layer.layer_type:", new_layer.layer_type)
            new_layer.size = layerSizes[0]
            # print("layerSizes:", layerSizes)

            new_layer.neurons = []
            for i in range(layerSizes[0]):
                new_neuron = neuron()
                new_layer.neurons.append(new_neuron)
            self.layers.append(new_layer)
            # print("self.layers:", self.layers)

            # 处理隐藏层的权重和bias 添加relu层
            for layernum in range(numLayers):
                previousLayerSize = layerSizes[layernum]
                currentLayerSize = layerSizes[layernum + 1]
                new_layer = layer()
                new_layer.size = currentLayerSize
                new_layer.layer_type = layer.AFFINE_LAYER
                new_layer.neurons = []

                # weights
                for i in range(currentLayerSize):
                    line = f.readline()
                    new_neuron = neuron()
                    weight = [float(x) for x in line.strip().split(",")[:-1]]
                    # if i == 1:
                    #     print("weight:", weight)
                    #     print("weight_len:", len(weight))

                    assert (len(weight) == previousLayerSize)
                    new_neuron.weight = np.array(weight)
                    new_layer.neurons.append(new_neuron)
                # biases
                for i in range(currentLayerSize):
                    line = f.readline()
                    x = float(line.strip().split(",")[0])
                    # if i == 1:
                    #     print("x:", x)
                    new_layer.neurons[i].bias = x

                self.layers.append(new_layer)

                # add relu layer 隐藏层后面新增加一个relu层，方便处理
                if layernum + 1 == numLayers:
                    break
                new_layer = layer()
                new_layer.size = currentLayerSize
                new_layer.layer_type = layer.RELU_LAYER
                new_layer.neurons = []
                for i in range(currentLayerSize):
                    new_neuron = neuron()
                    new_layer.neurons.append(new_neuron)
                self.layers.append(new_layer)

            self.numLayers = numLayers
            self.layerSizes = layerSizes
            self.inputSize = inputSize
            self.outputSize = outputSize
            self.mins = inputMinimums
            self.maxes = inputMaximums
            self.means = inputMeans
            self.ranges = inputRanges

    def find_max_disturbance(self, PROPERTY, L=0, R=1000, TRIM=False):
        ans = 0
        while L <= R:
            # print(L,R)
            mid = int((L + R) / 2)
            self.load_robustness(PROPERTY, mid / 1000, TRIM=TRIM)
            self.deeppoly()
            flag = True
            for neuron_i in self.layers[-1].neurons:
                # print("neuron_i.concrete_upper：", neuron_i.concrete_upper)
                if neuron_i.concrete_upper > 0:
                    flag = False
            if flag == True:
                ans = mid / 1000
                L = mid + 1
            else:
                R = mid - 1
        return ans

    def find_robustness_number(self, PROPERTY, t, TRIM=False):
        self.load_robustness(PROPERTY, delta=t, TRIM=TRIM)
        self.deeppoly()
        flag = True
        for neuron_i in self.layers[-1].neurons:
            # print("neuron_i.concrete_upper：", neuron_i.concrete_upper)
            if neuron_i.concrete_upper > 0:
                flag = False
        if flag == True:
            # print('Verified')
            num = 1
        else:
            # print('Unverified')
            num = 0
        return num



def mnist_robustness_radius():
    net = network()
    net.load_nnet("../../models/mnist_new_10x80/mnist_net_new_10x80.nnet")
    property_list = ["../../mnist_properties/mnist_properties_10x80/mnist_property_" + str(i) + ".txt" for i in range(100)]


    if not os.path.isdir('../../result/original_result'):
        os.makedirs('../../result/original_result')
    file = open("../../result/original_result/mnist_new_10x80_deeppoly_radius_result.txt", mode="w+", encoding="utf-8")

    for property_i in property_list:
        start_time = time.time()
        delta_base = net.find_max_disturbance(PROPERTY=property_i, TRIM=True)
        end_time = time.time()
        property_i = property_i[46:]
        property_i = property_i[:-4]
        # c_time = end_time - start_time
        print(f"{property_i} -- delta_base : {delta_base}")
        print(f"{property_i} -- time : {end_time - start_time}")

        save_radius_result(property_i, delta_base, end_time - start_time, file)
    file.close()



def acas_robustness_radius():
    for recur in range(1, 11):
        print("Recur:", recur)
        net = network()
        net.load_nnet("nnet/ACASXU_experimental_v2a_2_3.nnet")
        property_list = ["acas_properties/local_robustness_" + str(i) + ".txt" for i in range(2, 3)]
        for property_i in property_list:
            star_time = time.time()
            delta_base = net.find_max_disturbance(PROPERTY=property_i, TRIM=False)
            end_time = time.time()
            property_i = property_i[16:]
            property_i = property_i[:-4]
            print(f"{property_i} -- delta_base : {delta_base}")
            print(f"{property_i} -- time : {end_time - star_time}")


def cifar_robustness_radius():
    net = network()
    net.load_nnet("models/cifar_net.nnet")
    property_list = ["cifar_properties/cifar_properties_10x100/cifar_property_" + str(i) + ".txt" for i in range(50)]
    for property_i in property_list:
        star_time = time.time()
        delta_base = net.find_max_disturbance(PROPERTY=property_i, TRIM=True)
        end_time = time.time()
        property_i = property_i[41:-4]
        # property_i = property_i[:-4]
        print(f"{property_i} -- delta_base : {delta_base}")
        print(f"{property_i} -- time : {end_time - star_time}")



def test_example():
    net = network()
    net.load_nnet('paper_example/abstracmp_paper_illustration.nnet')
    net.load_robustness('paper_example/abstracmp_paper_illustration.txt', 1)
    net.deeppoly()
    net.print()


def test_acas():
    net = network()
    net.load_nnet('../../models/mnist_new_10x80/mnist_net_new_10x80.nnet')
    net.load_robustness('../../mnist_properties/mnist_new_10x80/mnist_property_19.txt', 0.001, TRIM=True)
    # print("-------111-----")
    net.deeppoly()
    # print("-------222-----")
    net.print()

    start_time = time.time()
    delta_base = net.find_max_disturbance('../../mnist_properties/mnist_property_19.txt', TRIM=True)
    end_time = time.time()
    print("delta_base:")
    print(delta_base)
    print(end_time - start_time)


def test_robustness_number0(t):
    net = network()
    net.load_nnet('../../models/mnist_new_10x80/mnist_net_new_10x80.nnet')
    property_list = ["../../mnist_properties/mnist_properties_10x80/mnist_property_" + str(i) + ".txt" for i in
                     range(100)]

    num = 0
    for property_i in property_list:
        net.load_robustness(property_i, delta = t, TRIM=True)
        start_time = time.time()
        net.deeppoly()
        flag = True
        for neuron_i in net.layers[-1].neurons:  # 这么写表示和其他数字有交集？判断网络f可达集与不安全区域S- 是否有交集？
            # print("neuron_i.concrete_upper：", neuron_i.concrete_upper)
            if neuron_i.concrete_upper > 0:  # 这是unkown鲁棒不鲁棒!  表示 网络f可达集与不安全区域S- 有 交集！
                flag = False  # 所以这是写成1 ，-1的原因，两者相减
        if flag == True:  # 这是鲁棒safe! 网络f可达集与不安全区域S- 没有 交集！
            print('Verified')
            num += 1
        else:  # 不鲁棒!
            print('Unverified')

        end_time = time.time()
        print("time:", end_time - start_time)

    print("num:", num)


def test_robustness_number(d):
    net = network()
    net.load_nnet("../../models/mnist_new_10x80/mnist_net_new_10x80.nnet")
    property_list = ["../../mnist_properties/mnist_properties_10x80/mnist_property_" + str(i) + ".txt" for i in
                     range(100)]

    if not os.path.isdir('../../result/original_result'):
        os.makedirs('../../result/original_result')
    file = open("../../result/original_result/mnist_new_10x80_deeppoly_number_result_delta_"+ str(d) +".txt", mode="w+", encoding="utf-8")

    num_ans = 0
    time_ans = 0
    for property_i in property_list:
        start_time = time.time()
        num_single = net.find_robustness_number(property_i, d, TRIM=True)
        property_i = property_i[46:]
        property_i = property_i[:-4]
        if num_single == 1:
            print(f"{property_i} -- Verified")
        else:
            print(f"{property_i} -- UnVerified")

        num_ans += num_single
        end_time = time.time()
        time_single = end_time - start_time
        print("time:", time_single)
        time_ans += time_single

        save_number_result(property_i, num_single, time_single, file)
    file.write("delta : " + str(d) + "\n")
    file.write("number_sum : " + str(num_ans) + "\n")
    file.write("time_sum : " + str(time_ans) + "\n")


    file.close()
    print("delta:", d)
    print("number_sum:", num_ans)
    print("time_sum:", time_ans)



def test_robustness_number_deeppoly(d):
    # Number of inputs to verify
    amount = 100

    net = network()
    net.load_nnet("../../models/nnet/ACASXU_run2a_1_1_batch_2000.nnet")
    property_list = [f"../../acas_properties/acas_xu_p5_net_1_1/local_robustness_{i}.txt" for i in range(1, amount+1)]

    if not os.path.isdir('../../result/original_result'):
        os.makedirs('../../result/original_result')
    file = open(f"../../result/original_result/acas_xu_p5_net_1_1_deeppoly_number_result_delta_{d}_{style_time}.txt", mode="w+", encoding="utf-8")

    num_ans = 0
    time_ans = 0
    time_max = 0

    for property_i in property_list:
        start_time = time.time()
        num_single = net.find_robustness_number(property_i, d, TRIM=True)
        property_i = property_i[22:]
        property_i = property_i[:-4]
        if num_single == 1:
            print(f"{property_i} -- Verified")
        else:
            print(f"{property_i} -- UnVerified")

        num_ans += num_single
        end_time = time.time()
        time_single = end_time - start_time
        print("time:", time_single)
        time_ans += time_single
        if time_single > time_max:
            time_max = time_single

        save_number_result(property_i, num_single, time_single, file)
    file.write("delta : " + str(d) + "\n")
    file.write("number_sum : " + str(num_ans) + "\n")
    file.write("time_sum : " + str(time_ans) + "\n")
    file.write("time_average : " + str(time_ans/amount) + "\n")
    file.write("time_max : " + str(time_max) + "\n")




    file.close()
    print("delta:", d)
    print("number_sum:", num_ans)
    print("time_sum:", time_ans)
    print("time_average:", time_ans/amount)
    print("time_max:", time_max)







if __name__ == "__main__":
    test_robustness_number_deeppoly(2)

