import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import cvxpy as cp
import time
from copy import deepcopy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed,wait

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



def lpsolve(vars,cons,obj,solver=cp.GUROBI):
    prob=cp.Problem(obj,cons)
    prob.solve(solver=solver)
    return prob.value

class neuron(object):
    """
    Attributes:
        algebra_lower (numpy ndarray of float): neuron's algebra lower bound(coeffients of previous neurons and a constant)
        algebra_upper (numpy ndarray of float): neuron's algebra upper bound(coeffients of previous neurons and a constant)
        concrete_algebra_lower (numpy ndarray of float): neuron's algebra lower bound(coeffients of input neurons and a constant)
        concrete_algebra_upper (numpy ndarray of float): neuron's algebra upper bound(coeffients of input neurons and a constant)
        concrete_lower (float): neuron's concrete lower bound
        concrete_upper (float): neuron's concrete upper bound
        concrete_highest_lower (float): neuron's highest concrete lower bound
        concrete_lowest_upper (float): neuron's lowest concrete upper bound
        weight (numpy ndarray of float): neuron's weight
        bias (float): neuron's bias
        certain_flag (int): 0 uncertain 1 activated(>=0) 2 deactivated(<=0)
        prev_abs_mode (int): indicates abstract mode of relu nodes in previous iteration.0 use first,1 use second
    """

    def __init__(self):
        self.algebra_lower=None
        self.algebra_upper=None
        self.concrete_algebra_lower=None
        self.concrete_algebra_upper=None
        self.concrete_lower=None
        self.concrete_upper=None
        self.concrete_highest_lower=None
        self.concrete_lowest_upper=None
        self.weight=None
        self.bias=None
        self.prev_abs_mode=None
        self.certain_flag=0
    
    def clear(self):
        self.certain_flag=0
        self.concrete_highest_lower=None
        self.concrete_lowest_upper=None
        self.prev_abs_mode=None

    def print(self):
        print('algebra_lower:',self.algebra_lower)
        print('algebra_upper:',self.algebra_upper)
        print('concrete_algebra_lower:',self.concrete_algebra_lower)
        print('concrete_algebra_upper:',self.concrete_algebra_upper)
        print('concrete_lower:',self.concrete_lower)
        print('concrete_upper:',self.concrete_upper)
        print('weight:',self.weight)
        print('bias:',self.bias)
        print('certain_flag:',self.certain_flag)

class layer(object):
    """
    Attributes:
        neurons (list of neuron): Layer neurons
        size (int): Layer size
        layer_type (int) : Layer type 0 input 1 affine 2 relu
    """
    INPUT_LAYER=0
    AFFINE_LAYER=1
    RELU_LAYER=2    

    def __init__(self):
        self.size=None
        self.neurons=None
        self.layer_type=None
    
    def clear(self):
        for i in range(len(self.neurons)):
            self.neurons[i].clear()

    def print(self):
        print('Layer size:',self.size)
        print('Layer type:',self.layer_type)
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
        unsafe_region (list of ndarray):coeffient of output and a constant
        property_flag (bool) : indicates the network have verification layer or not
        property_region (float) : Area of the input box
        abs_mode_changed (int) : count of uncertain relu abstract mode changed
        self.MODE_ROBUSTNESS=1
        self.MODE_QUANTITIVE=0
    """

    def __init__(self):
        self.MODE_QUANTITIVE=0
        self.MODE_ROBUSTNESS=1

        self.numlayers=None
        self.layerSizes=None
        self.inputSize=None
        self.outputSize=None
        self.mins=None
        self.maxes=None
        self.ranges=None
        self.layers=None
        self.property_flag=None
        self.property_region=None
        self.abs_mode_changed=None
    
    def clear(self):
        for i in range(len(self.layers)):
            self.layers[i].clear()
    


    def verify_lp_split(self, PROPERTY, DELTA, MAX_ITER=5, SPLIT_NUM=0, WORKERS=28, TRIM=False, SOLVER=cp.GUROBI,
                        MODE=0, USE_OPT_2=False):
        if SPLIT_NUM > self.inputSize:
            SPLIT_NUM = self.inputSize
        if self.property_flag == True:
            self.layers.pop()
            self.property_flag = False
        self.load_robustness(PROPERTY, DELTA, TRIM=TRIM)
        delta_list = [
            self.layers[0].neurons[i].concrete_upper - self.layers[0].neurons[i].concrete_lower for
            i in range(self.inputSize)]
        self.clear()
        self.deeppoly()
        verify_layer = self.layers[-1]
        verify_neuron_upper = np.array([neur.concrete_upper for neur in verify_layer.neurons])
        verify_list = np.argsort(verify_neuron_upper)
        for i in range(len(verify_list)):
            if verify_neuron_upper[verify_list[i]] >= 0:
                verify_list = verify_list[i:]
                break
        if verify_neuron_upper[verify_list[0]] < 0:
            print("Property Verified")
            if MODE == self.MODE_ROBUSTNESS:
                return True
            return
        split_list = []
        for i in range(2 ** SPLIT_NUM):
            cur_split = []
            for j in range(SPLIT_NUM):
                if i & (2 ** j) == 0:
                    cur_split.append([self.layers[0].neurons[j].concrete_lower, (
                                self.layers[0].neurons[j].concrete_upper + self.layers[0].neurons[
                            j].concrete_lower) / 2])
                else:
                    cur_split.append([(self.layers[0].neurons[j].concrete_upper + self.layers[0].neurons[
                        j].concrete_lower) / 2, self.layers[0].neurons[j].concrete_upper])
            for j in range(SPLIT_NUM, self.inputSize):
                cur_split.append([self.layers[0].neurons[j].concrete_lower,
                                  self.layers[0].neurons[j].concrete_upper])
            split_list.append(cur_split)

        obj = None
        prob = None
        constraints = None
        variables = []
        for i in range(len(self.layers)):
            variables.append(cp.Variable(self.layers[i].size))

        refresh_count = 0
        refresh_time_ans = 0

        for i in verify_list:
            verification_neuron = self.layers[-1].neurons[i]
            total_area = 0
            for splits_num in range(len(split_list)):
                splits = split_list[splits_num]
                assert (len(splits) == self.inputSize)
                for j in range(self.inputSize):
                    self.layers[0].neurons[j].concrete_lower = splits[j][0]
                    self.layers[0].neurons[j].concrete_algebra_lower = np.array([splits[j][0]])
                    self.layers[0].neurons[j].algebra_lower = np.array([splits[j][0]])
                    self.layers[0].neurons[j].concrete_upper = splits[j][1]
                    self.layers[0].neurons[j].concrete_algebra_upper = np.array([splits[j][1]])
                    self.layers[0].neurons[j].algebra_upper = np.array([splits[j][1]])
                self.clear()

        verify_list = verify_list.tolist()
        for j in range(MAX_ITER):
            self.deeppoly()

            flag = True
            for neuron_i in self.layers[-1].neurons:
                # print("neuron_i.concrete_upper：", neuron_i.concrete_upper)
                if neuron_i.concrete_lowest_upper > 0:
                    flag = False
            if flag == True:
                print('Verified')
                return 1

                # num = 1
            else:  # Not robust!
                print('Unverified')
                # num = 0
                constraints = []
                # Build Constraints for each layer
                for k in range(len(self.layers)):
                    cur_layer = self.layers[k]
                    cur_neuron_list = cur_layer.neurons
                    if cur_layer.layer_type == layer.INPUT_LAYER:
                        for p in range(cur_layer.size):
                            constraints.append(variables[k][p] >= cur_neuron_list[p].concrete_lower)
                            constraints.append(variables[k][p] <= cur_neuron_list[p].concrete_upper)
                    elif cur_layer.layer_type == layer.AFFINE_LAYER:
                        assert (k > 0)
                        for p in range(cur_layer.size):
                            constraints.append(
                                variables[k][p] == cur_neuron_list[p].weight @ variables[k - 1] + cur_neuron_list[
                                    p].bias)
                    elif cur_layer.layer_type == layer.RELU_LAYER:
                        assert (cur_layer.size == self.layers[k - 1].size)
                        assert (k > 0)
                        for p in range(cur_layer.size):
                            constraints.append(
                                variables[k][p] <= cur_neuron_list[p].algebra_upper[:-1] @ variables[
                                    k - 1] + cur_neuron_list[p].algebra_upper[-1])
                            # constraints.append(variables[k][p]>=cur_neuron_list[p].algebra_lower[:-1]@variables[k-1]+cur_neuron_list[p].algebra_lower[-1])
                            # Modified:using two lower bounds
                            constraints.append(variables[k][p] >= 0)
                            constraints.append(variables[k][p] >= variables[k - 1][p])

                # Build the verification neuron constraint
                # Assume adding i binary variables
                len_lastlayer = len(verify_list)

                # Create i binary variables using list comprehension
                binary_vars = [cp.Variable(boolean=True) for _ in range(len_lastlayer)]

                # Big-M method
                M = 10000

                for i in verify_list:
                    # constraints.append(variables[-1][i] >= 0 - M * (1 - binary_vars[i]))
                    constraints.append(variables[-1][i] >= 0 - M * (1 - binary_vars[verify_list.index(i)]))


                # Construct disjunctive relations
                constraints.append(sum(binary_vars) >= 1)

                # Check the feasibility
                prob = cp.Problem(cp.Maximize(0), constraints)
                prob.solve(solver=SOLVER)
                if prob.status != cp.OPTIMAL:
                    print("Infeasible")
                    # print("Split:", splits_num, "Infeasible")
                    break

                refresh_start_time = time.time()

                # Refresh the uncertain ReLu's lowerbound
                mppool = mp.Pool(WORKERS)
                count_uncertain = 0
                count_uncertain_run = 0
                tasklist = []
                for k in range(len(self.layers) - 1):
                    cur_layer = self.layers[k]
                    next_layer = self.layers[k + 1]
                    if cur_layer.layer_type == layer.AFFINE_LAYER and next_layer.layer_type == layer.RELU_LAYER:
                        assert (cur_layer.size == next_layer.size)
                        for p in range(cur_layer.size):
                            if next_layer.neurons[p].certain_flag == 0:
                                obj = cp.Minimize(variables[k][p])
                                # Below using mp Pool
                                tasklist.append((variables, constraints, obj, SOLVER))
                # Below using mp Pool
                resultlist = mppool.starmap(lpsolve, tasklist)
                mppool.terminate()
                index = 0
                for k in range(len(self.layers) - 1):
                    cur_layer = self.layers[k]
                    next_layer = self.layers[k + 1]
                    if cur_layer.layer_type == layer.AFFINE_LAYER and next_layer.layer_type == layer.RELU_LAYER:
                        assert (cur_layer.size == next_layer.size)
                        for p in range(cur_layer.size):
                            if next_layer.neurons[p].certain_flag == 0:
                                if resultlist[index] > cur_layer.neurons[p].concrete_highest_lower:
                                    cur_layer.neurons[p].concrete_highest_lower = resultlist[index]
                                    count_uncertain_run += 1
                                if resultlist[index] >= 0:
                                    next_layer.neurons[p].certain_flag = 1
                                    count_uncertain += 1
                                index += 1

                # Refresh the uncertain ReLu's upperbound
                mppool = mp.Pool(WORKERS)
                tasklist = []
                for k in range(len(self.layers) - 1):
                    cur_layer = self.layers[k]
                    next_layer = self.layers[k + 1]
                    if cur_layer.layer_type == layer.AFFINE_LAYER and next_layer.layer_type == layer.RELU_LAYER:
                        assert (cur_layer.size == next_layer.size)
                        for p in range(cur_layer.size):
                            if next_layer.neurons[p].certain_flag == 0:
                                obj = cp.Maximize(variables[k][p])
                                # Below using mp Pool
                                tasklist.append((variables, constraints, obj, SOLVER))
                # Below using mp Pool
                resultlist = mppool.starmap(lpsolve, tasklist)
                mppool.terminate()
                index = 0
                for k in range(len(self.layers) - 1):
                    cur_layer = self.layers[k]
                    next_layer = self.layers[k + 1]
                    if cur_layer.layer_type == layer.AFFINE_LAYER and next_layer.layer_type == layer.RELU_LAYER:
                        assert (cur_layer.size == next_layer.size)
                        for p in range(cur_layer.size):
                            if next_layer.neurons[p].certain_flag == 0:
                                if resultlist[index] < cur_layer.neurons[p].concrete_lowest_upper:
                                    cur_layer.neurons[p].concrete_lowest_upper = resultlist[index]
                                    count_uncertain_run += 1
                                if resultlist[index] <= 0:
                                    next_layer.neurons[p].certain_flag = 2
                                    count_uncertain += 1
                                index += 1
                print('Refreshed uncertain ReLu nodes:', count_uncertain)
                print('Run uncertain ReLu nodes:', count_uncertain_run)



                refresh_end_time = time.time()
                refresh_time_once = refresh_end_time - refresh_start_time
                print("refresh_time_once:", refresh_time_once)


                refresh_time_ans += refresh_time_once
                refresh_count += 1

        print("refresh_count:", refresh_count)
        if refresh_count != 0:
            print("refresh_time_average:", refresh_time_ans / refresh_count)


        if prob.status != cp.OPTIMAL:
            if MODE == self.MODE_ROBUSTNESS:
                return True
        else:
            if MODE==self.MODE_ROBUSTNESS:
                return False



    def deeppoly(self):

        def pre(cur_neuron,i):
            if i==0:
                cur_neuron.concrete_algebra_lower=deepcopy(cur_neuron.algebra_lower)
                cur_neuron.concrete_algebra_upper=deepcopy(cur_neuron.algebra_upper)
            lower_bound=deepcopy(cur_neuron.algebra_lower)
            upper_bound=deepcopy(cur_neuron.algebra_upper)
            for k in range(i+1)[::-1]:
                # print(k)
                tmp_lower=np.zeros(len(self.layers[k].neurons[0].algebra_lower))
                tmp_upper=np.zeros(len(self.layers[k].neurons[0].algebra_lower))
                assert(self.layers[k].size+1==len(lower_bound))
                assert(self.layers[k].size+1==len(upper_bound))
                for p in range(self.layers[k].size):
                    if lower_bound[p]>=0:  
                        # print(lower_bound[p]*self.layers[k].neurons[p].algebra_lower)                                 
                        tmp_lower+=lower_bound[p]*self.layers[k].neurons[p].algebra_lower
                    else:
                        # print(lower_bound[p]*self.layers[k].neurons[p].algebra_upper)
                        tmp_lower+=lower_bound[p]*self.layers[k].neurons[p].algebra_upper

                    if upper_bound[p]>=0:                        
                        tmp_upper+=upper_bound[p]*self.layers[k].neurons[p].algebra_upper
                    else:
                        tmp_upper+=upper_bound[p]*self.layers[k].neurons[p].algebra_lower                
                # print(tmp_lower)
                tmp_lower[-1]+=lower_bound[-1]
                tmp_upper[-1]+=upper_bound[-1]
                lower_bound=deepcopy(tmp_lower)
                upper_bound=deepcopy(tmp_upper)
                if k==1:
                    cur_neuron.concrete_algebra_upper=deepcopy(upper_bound)
                    cur_neuron.concrete_algebra_lower=deepcopy(lower_bound)
            assert(len(lower_bound)==1)
            assert(len(upper_bound)==1)
            cur_neuron.concrete_lower=lower_bound[0]
            cur_neuron.concrete_upper=upper_bound[0]
            #add lowest and uppest history
            if (cur_neuron.concrete_highest_lower==None) or (cur_neuron.concrete_highest_lower<cur_neuron.concrete_lower):
                cur_neuron.concrete_highest_lower=cur_neuron.concrete_lower
            if (cur_neuron.concrete_lowest_upper==None) or (cur_neuron.concrete_lowest_upper>cur_neuron.concrete_upper):
                cur_neuron.concrete_lowest_upper=cur_neuron.concrete_upper            


        self.abs_mode_changed=0
        for i in range(len(self.layers)-1):
            # print('i=',i)
            pre_layer=self.layers[i]
            cur_layer=self.layers[i+1]
            pre_neuron_list=pre_layer.neurons
            cur_neuron_list=cur_layer.neurons
            if cur_layer.layer_type==layer.AFFINE_LAYER:
                for j in range(cur_layer.size):
                    cur_neuron=cur_neuron_list[j]
                    cur_neuron.algebra_lower=np.append(cur_neuron.weight,[cur_neuron.bias])
                    cur_neuron.algebra_upper=np.append(cur_neuron.weight,[cur_neuron.bias])
                    pre(cur_neuron,i)
            elif cur_layer.layer_type==layer.RELU_LAYER:
                for j in range(cur_layer.size):
                    cur_neuron=cur_neuron_list[j]
                    pre_neuron=pre_neuron_list[j]
                    #modified using lowest and uppest bound
                    if pre_neuron.concrete_highest_lower>=0 or cur_neuron.certain_flag==1:
                        cur_neuron.algebra_lower=np.zeros(cur_layer.size+1)
                        cur_neuron.algebra_upper=np.zeros(cur_layer.size+1)
                        cur_neuron.algebra_lower[j]=1
                        cur_neuron.algebra_upper[j]=1
                        cur_neuron.concrete_algebra_lower=deepcopy(pre_neuron.concrete_algebra_lower)
                        cur_neuron.concrete_algebra_upper=deepcopy(pre_neuron.concrete_algebra_upper)
                        cur_neuron.concrete_lower=pre_neuron.concrete_lower
                        cur_neuron.concrete_upper=pre_neuron.concrete_upper
                        #added
                        cur_neuron.concrete_highest_lower=pre_neuron.concrete_highest_lower
                        cur_neuron.concrete_lowest_upper=pre_neuron.concrete_lowest_upper
                        cur_neuron.certain_flag=1
                    elif pre_neuron.concrete_lowest_upper<=0 or cur_neuron.certain_flag==2:
                        cur_neuron.algebra_lower=np.zeros(cur_layer.size+1)
                        cur_neuron.algebra_upper=np.zeros(cur_layer.size+1)                        
                        cur_neuron.concrete_algebra_lower=np.zeros(self.inputSize)
                        cur_neuron.concrete_algebra_upper=np.zeros(self.inputSize)
                        cur_neuron.concrete_lower=0
                        cur_neuron.concrete_upper=0
                        #added
                        cur_neuron.concrete_highest_lower=0
                        cur_neuron.concrete_lowest_upper=0
                        cur_neuron.certain_flag=2
                    elif pre_neuron.concrete_highest_lower+pre_neuron.concrete_lowest_upper<=0:
                        #Relu abs mode changed
                        if (cur_neuron.prev_abs_mode!=None) and (cur_neuron.prev_abs_mode!=0):
                            self.abs_mode_changed+=1
                        cur_neuron.prev_abs_mode=0

                        cur_neuron.algebra_lower=np.zeros(cur_layer.size+1)
                        aux=pre_neuron.concrete_lowest_upper/(pre_neuron.concrete_lowest_upper-pre_neuron.concrete_highest_lower)
                        cur_neuron.algebra_upper=np.zeros(cur_layer.size+1)
                        cur_neuron.algebra_upper[j]=aux
                        cur_neuron.algebra_upper[-1]=-aux*pre_neuron.concrete_highest_lower
                        pre(cur_neuron,i)
                    else:
                        #Relu abs mode changed
                        if (cur_neuron.prev_abs_mode!=None) and (cur_neuron.prev_abs_mode!=1):
                            self.abs_mode_changed+=1
                        cur_neuron.prev_abs_mode=1

                        cur_neuron.algebra_lower=np.zeros(cur_layer.size+1)
                        cur_neuron.algebra_lower[j]=1
                        aux=pre_neuron.concrete_lowest_upper/(pre_neuron.concrete_lowest_upper-pre_neuron.concrete_highest_lower)
                        cur_neuron.algebra_upper=np.zeros(cur_layer.size+1)
                        cur_neuron.algebra_upper[j]=aux
                        cur_neuron.algebra_upper[-1]=-aux*pre_neuron.concrete_highest_lower
                        pre(cur_neuron,i)

    def print(self):
        print('numlayers:%d' % (self.numLayers))
        print('layerSizes:',self.layerSizes)        
        print('inputSize:%d' % (self.inputSize))
        print('outputSize:%d' % (self.outputSize))
        print('mins:',self.mins)        
        print('maxes:',self.maxes)        
        print('ranges:',self.ranges)
        print('Layers:')
        for l in self.layers:
            l.print()
            print('\n')
    
    def load_property(self, filename):
        self.property_flag=True
        self.property_region=1
        with open(filename) as f:
            for i in range(self.layerSizes[0]):
                line = f.readline()
                linedata=[float(x) for x in line.strip().split(' ')]
                self.layers[0].neurons[i].concrete_lower=linedata[0]
                self.layers[0].neurons[i].concrete_upper=linedata[1]
                self.property_region*=linedata[1]-linedata[0]
                self.layers[0].neurons[i].concrete_algebra_lower=np.array([linedata[0]])
                self.layers[0].neurons[i].concrete_algebra_upper=np.array([linedata[1]])
                self.layers[0].neurons[i].algebra_lower=np.array([linedata[0]])
                self.layers[0].neurons[i].algebra_upper=np.array([linedata[1]])
                # print(linedata)
            self.unsafe_region=[]
            line=f.readline()
            verify_layer=layer()
            verify_layer.neurons=[]
            while line:                                
                linedata=[float(x) for x in line.strip().split(' ')]
                assert(len(linedata)==self.outputSize+1)
                verify_neuron=neuron()
                verify_neuron.weight=np.array(linedata[:-1])
                verify_neuron.bias=linedata[-1]
                verify_layer.neurons.append(verify_neuron)
                linedata=np.array(linedata)
                # print(linedata)
                self.unsafe_region.append(linedata)
                assert(len(linedata)==self.outputSize+1)
                line=f.readline()
            verify_layer.size=len(verify_layer.neurons)
            verify_layer.layer_type=layer.AFFINE_LAYER
            if len(verify_layer.neurons)>0:
                self.layers.append(verify_layer)

    def load_robustness(self, filename,delta, TRIM=False):
        if self.property_flag==True:
                self.layers.pop()
                # self.clear()
        self.property_flag=True        
        with open(filename) as f:
            self.property_region=1
            for i in range(self.layerSizes[0]):
                line = f.readline()
                linedata=[float(line.strip())-delta,float(line.strip())+delta]
                if TRIM:
                    if linedata[0]<0:
                        linedata[0]=0
                    if linedata[1]>1:
                        linedata[1]=1
                self.layers[0].neurons[i].concrete_lower=linedata[0]
                self.layers[0].neurons[i].concrete_upper=linedata[1]
                self.property_region*=linedata[1]-linedata[0]
                self.layers[0].neurons[i].concrete_algebra_lower=np.array([linedata[0]])
                self.layers[0].neurons[i].concrete_algebra_upper=np.array([linedata[1]])
                self.layers[0].neurons[i].algebra_lower=np.array([linedata[0]])
                self.layers[0].neurons[i].algebra_upper=np.array([linedata[1]])
                # print(linedata)
            self.unsafe_region=[]
            line=f.readline()
            verify_layer=layer()
            verify_layer.neurons=[]
            while line:                                
                linedata=[float(x) for x in line.strip().split(' ')]
                assert(len(linedata)==self.outputSize+1)
                verify_neuron=neuron()
                verify_neuron.weight=np.array(linedata[:-1])
                verify_neuron.bias=linedata[-1]
                verify_layer.neurons.append(verify_neuron)
                linedata=np.array(linedata)
                # print(linedata)
                self.unsafe_region.append(linedata)
                assert(len(linedata)==self.outputSize+1)
                line=f.readline()
            verify_layer.size=len(verify_layer.neurons)
            verify_layer.layer_type=layer.AFFINE_LAYER
            if len(verify_layer.neurons)>0:
                self.layers.append(verify_layer)
    
    def load_nnet(self, filename):
        with open(filename) as f:
            line = f.readline()
            cnt = 1
            while line[0:2] == "//":
                line=f.readline() 
                cnt+= 1
            #numLayers does't include the input layer!
            numLayers, inputSize, outputSize, _ = [int(x) for x in line.strip().split(",")[:-1]]
            line=f.readline()

            #input layer size, layer1size, layer2size...
            layerSizes = [int(x) for x in line.strip().split(",")[:-1]]

            line=f.readline()
            symmetric = int(line.strip().split(",")[0])

            line = f.readline()
            inputMinimums = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            inputMaximums = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            inputMeans = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            inputRanges = [float(x) for x in line.strip().split(",")[:-1]]

            #process the input layer
            self.layers=[]
            new_layer=layer()
            new_layer.layer_type=layer.INPUT_LAYER
            new_layer.size=layerSizes[0]
            new_layer.neurons=[]
            for i in range(layerSizes[0]):
                new_neuron=neuron()                
                new_layer.neurons.append(new_neuron)
            self.layers.append(new_layer)

            for layernum in range(numLayers):

                previousLayerSize = layerSizes[layernum]
                currentLayerSize = layerSizes[layernum+1]
                new_layer=layer()
                new_layer.size=currentLayerSize
                new_layer.layer_type=layer.AFFINE_LAYER
                new_layer.neurons=[]
                for i in range(currentLayerSize):
                    line=f.readline()
                    new_neuron=neuron()
                    aux = [float(x) for x in line.strip().split(",")[:-1]]
                    assert(len(aux)==previousLayerSize)
                    new_neuron.weight=np.array(aux)
                    new_layer.neurons.append(new_neuron)

                #biases                
                for i in range(currentLayerSize):
                    line=f.readline()
                    x = float(line.strip().split(",")[0])
                    new_layer.neurons[i].bias=x
                
                self.layers.append(new_layer)                

                #add relu layer
                if layernum+1==numLayers:
                    break
                new_layer=layer()
                new_layer.size=currentLayerSize
                new_layer.layer_type=layer.RELU_LAYER
                new_layer.neurons=[]
                for i in range(currentLayerSize):
                    new_neuron=neuron()
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
            self.property_flag=False        

    def load_rlv(self,filename):        
        layersize=[]
        dicts=[]        
        self.layers=[]
        with open(filename,'r') as f:
            line=f.readline()
            while(line):            
                if(line.startswith('#')):
                    linedata=line.replace('\n','').split(' ')
                    layersize.append(int(linedata[3]))
                    layerdict={}
                    if(linedata[4]=='Input'):
                        new_layer=layer()
                        new_layer.layer_type=layer.INPUT_LAYER
                        new_layer.size=layersize[-1]
                        new_layer.neurons=[]
                        for i in range(layersize[-1]):
                            new_neuron=neuron()
                            new_layer.neurons.append(new_neuron)
                            line=f.readline()
                            linedata=line.split(' ')
                            layerdict[linedata[1].replace('\n','')]=i
                        dicts.append(layerdict)
                        self.layers.append(new_layer)
                    elif (linedata[4]=='ReLU'):
                        new_layer=layer()
                        new_layer.layer_type=layer.AFFINE_LAYER
                        new_layer.size=layersize[-1]
                        new_layer.neurons=[]                        
                        for i in range(layersize[-1]):
                            new_neuron=neuron()
                            new_neuron.weight=np.zeros(layersize[-2])
                            line=f.readline()
                            linedata=line.replace('\n','').split(' ')                        
                            layerdict[linedata[1]]=i
                            new_neuron.bias=float(linedata[2])                            
                            nodeweight=linedata[3::2]
                            nodename=linedata[4::2]
                            assert(len(nodeweight)==len(nodename))
                            for j in range(len(nodeweight)):
                                new_neuron.weight[dicts[-1][nodename[j]]]=float(nodeweight[j])
                            new_layer.neurons.append(new_neuron)
                        self.layers.append(new_layer)                                     
                        dicts.append(layerdict)
                        #add relu layer
                        new_layer=layer()
                        new_layer.layer_type=layer.RELU_LAYER
                        new_layer.size=layersize[-1]
                        new_layer.neurons=[]
                        for i in range(layersize[-1]):
                            new_neuron=neuron()
                            new_layer.neurons.append(new_neuron)
                        self.layers.append(new_layer)
                    elif (linedata[4]=='Linear') and (linedata[5]!='Accuracy'):
                        new_layer=layer()
                        new_layer.layer_type=layer.AFFINE_LAYER
                        new_layer.size=layersize[-1]
                        new_layer.neurons=[]                        
                        for i in range(layersize[-1]):
                            new_neuron=neuron()
                            new_neuron.weight=np.zeros(layersize[-2])
                            line=f.readline()
                            linedata=line.replace('\n','').split(' ')                        
                            layerdict[linedata[1]]=i
                            new_neuron.bias=float(linedata[2])                            
                            nodeweight=linedata[3::2]
                            nodename=linedata[4::2]
                            assert(len(nodeweight)==len(nodename))
                            for j in range(len(nodeweight)):
                                new_neuron.weight[dicts[-1][nodename[j]]]=float(nodeweight[j])
                            new_layer.neurons.append(new_neuron)
                        self.layers.append(new_layer)                                     
                        dicts.append(layerdict)
                line=f.readline()
        self.layerSizes=layersize
        self.inputSize=layersize[0]
        self.outputSize=layersize[-1]
        self.numLayers=len(layersize)-1
        pass

    def find_max_disturbance(self,PROPERTY,L=0,R=1000,TRIM=False):
        ans=0
        while L<=R:
            # print(L,R)
            mid=int((L+R)/2)
            self.load_robustness(PROPERTY,mid/1000,TRIM=TRIM)
            self.clear()
            self.deeppoly()
            flag=True
            for neuron_i in self.layers[-1].neurons:
                # print(neuron_i.concrete_upper)
                if neuron_i.concrete_upper>0:
                    flag=False
            if flag==True:
                ans=mid/1000
                L=mid+1
            else:
                R=mid-1
        return ans
    
    def find_max_disturbance_lp(self,PROPERTY,L,R,TRIM,WORKERS=28,SOLVER=cp.GUROBI):
        ans=L
        while L<=R:
            mid=int((L+R)/2)
            if self.verify_lp_split(PROPERTY=PROPERTY,DELTA=mid/1000,MAX_ITER=5,SPLIT_NUM=0,WORKERS=WORKERS,TRIM=TRIM,SOLVER=SOLVER,MODE=1):
                print("Disturbance:",mid/1000,"Success!")
                ans=mid/1000
                L=mid+1
            else:
                print("Disturbance:",mid/1000,"Failed!")
                R=mid-1
        return ans


    def find_robustness_number_mrlp(self, PROPERTY, t, TRIM, WORKERS=28, SOLVER=cp.GUROBI):
        if self.verify_lp_split(PROPERTY=PROPERTY, DELTA=t, MAX_ITER=5, SPLIT_NUM=0, WORKERS=WORKERS,
                                TRIM=TRIM, SOLVER=SOLVER, MODE=1):
            print("Disturbance:", t, "Success!")
            num = 1
        else:
            print("Disturbance:", t, "Failed!")
            num = 0
        return num



def mnist_robustness_radius_deeppoly():
    net = network()
    net.load_nnet("../../models/mnist_new_5x50/mnist_net_new_5x50.nnet")
    property_list = ["../../mnist_properties/mnist_properties_5x50/mnist_property_" + str(i) + ".txt" for i in range(100)]

    if not os.path.isdir('../../result/original_result'):
        os.makedirs('../../result/original_result')
    style_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    file = open("../../result/original_result/mnist_new_5x50_deeppoly_radius_result_" + str(style_time) + ".txt", mode="w+", encoding="utf-8")

    for property_i in property_list:
        start_time = time.time()
        delta_base = net.find_max_disturbance(PROPERTY=property_i, TRIM=True)
        end_time = time.time()
        property_i = property_i[46:]
        property_i = property_i[:-4]
        print(f"{property_i} -- delta_base : {delta_base}")
        print(f"{property_i} -- time : {end_time - start_time}")

        save_radius_result(property_i, delta_base, end_time - start_time, file)
    file.close()


def mnist_robustness_radius_lp():
    net = network()
    net.load_nnet("../../models/mnist_new_5x50/mnist_net_new_5x50.nnet")
    property_list = ["../../mnist_properties/mnist_properties_5x50/mnist_property_" + str(i) + ".txt" for i in range(100)]

    if not os.path.isdir('../../result/original_result'):
        os.makedirs('../../result/original_result')
    style_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    file = open("../../result/original_result/mnist_new_5x50_deepsrgr_radius_result_" + str(style_time) + ".txt", mode="w+", encoding="utf-8")

    for property_i in property_list:
        print(f"{property_i}")
        start_time = time.time()
        delta_base = net.find_max_disturbance(PROPERTY=property_i, TRIM=True)
        print("DeepPoly Max Verified Distrubance:", delta_base)
        delta_base = net.find_max_disturbance_lp(PROPERTY=property_i,L=int(delta_base*1000),R=int(delta_base*1000+63),TRIM=True,WORKERS=28,SOLVER=cp.GUROBI)
        end_time = time.time()
        property_i = property_i[46:]
        property_i = property_i[:-4]
        print(f"{property_i} -- delta_base : {delta_base}")
        print(f"{property_i} -- time : {end_time - start_time}")

        save_radius_result(property_i, delta_base, end_time - start_time, file)
    file.close()



def test_robustness_number_mrlp(d):
    # Number of inputs to verify
    amount = 100

    net = network()
    net.load_nnet("../../models/mnist_new_5x50/mnist_net_new_5x50.nnet")
    property_list = ["../../mnist_properties/mnist_properties_5x50/mnist_property_" + str(i) + ".txt" for i in
                     range(amount)]

    if not os.path.isdir('../../result/original_result'):
        os.makedirs('../../result/original_result')
    file = open("../../result/original_result/mnist_new_5x50_deepmr_3_number_result_delta_" + str(d) + "_" + str(style_time) + ".txt", mode="w+", encoding="utf-8")

    num_ans = 0
    time_ans = 0
    time_max = 0
    for property_i in property_list:
        start_time = time.time()
        num_single = net.find_robustness_number_mrlp(property_i, d, TRIM=True)
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
    test_robustness_number_mrlp(0.018)


