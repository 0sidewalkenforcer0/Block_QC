import networkx as nx
import matplotlib.pyplot as plt
import sys
import time
import numpy as np
import pandas as pd
import seaborn as sns
from dwave_qbsolv import QBSolv
from matplotlib import pyplot as plt
import copy
import progressbar
import os
import os.path
from scipy.optimize import minimize
import matplotlib.pylab as plt
import plotly.express as px
import seaborn as sns
import random

#Quantum Annealing
from dwave_qbsolv import QBSolv
import dimod
import neal
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave.system import LeapHybridSampler
import dimod.utilities

#Quantum Gate Model
from qiskit import IBMQ
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import Aer, execute
from qiskit.circuit import Parameter
from qiskit.visualization import plot_histogram
from qiskit import Aer
from qiskit.algorithms.optimizers import SPSA, ADAM, COBYLA
from qiskit.aqua.operators import EvolvedOp
from qiskit.opflow import CircuitOp
from qiskit.optimization.algorithms import MinimumEigenOptimizer, CplexOptimizer
from qiskit.optimization.converters.quadratic_program_to_qubo import QuadraticProgramToQubo
from qiskit.optimization import QuadraticProgram
from qiskit.algorithms.minimum_eigen_solvers.qaoa import QAOAAnsatz
from qiskit.aqua.algorithms import QAOA
from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller

import utils


'''
Alle möglichen Variablen-tupeln x_(i,j,k)
'''
def get_all_vars(row_num, column_num, k_max):
    alle_vars = []
    for i in range(row_num):
        for j in range(column_num):
            for k in range(k_max):
                alle_vars.append((i,j,k))
    return alle_vars

'''
Alle unbekannten Variablen-tupeln x_(i,j,k)
'''
def get_unbekannte_vars(row_num, column_num, k_max, known_vars):
    var_list = []

    for i in range(row_num):
        for j in range(column_num):
            for k in range(k_max):
                if (i, j) not in known_vars.keys():
                    var_list.append((i,j,k))
    return var_list


'''
Um die Anzahl von Variablen zu reduzieren, löschen wir alle absurden (unmöglichen) Variablen x_(i,j,k)
Prunning in ROW
Prunning in COLUMN
Prunning in SUBBLOCK
'''
# row_num, column_num - matrix size
# num_sub_row, num_sub_column - anzahl von subblocks
def get_prunned_vars(row_num, column_num, k_max, num_sub_row, num_sub_column, block_size, known_vars):
    all_vars = get_all_vars(row_num, column_num, k_max)
    var_list = get_unbekannte_vars(row_num, column_num, k_max, known_vars)
    prunned_list = var_list
    all_blocks = get_all_blocks(num_sub_row, num_sub_column)
    
    #prunning rows and columns
    for known_var in known_vars:
        for var in var_list:
            if (var[0] == known_var[0] or var[1] == known_var[1]):
                if (var[2] == known_vars[known_var]):
                    prunned_list.remove(var)
    
    #prunning blocks
    for block in all_blocks:
        variables_in_block = get_vars_in_block(all_vars, block[0], block[1], block_size)
        for var_1 in variables_in_block:
            for var_2 in variables_in_block:
                if (((var_1[0], var_1[1]) in known_vars) and (var_2 in prunned_list)):
                    if (var_2[2] == known_vars[(var_1[0],var_1[1])]):
                        prunned_list.remove(var_2)
                    
    return prunned_list


'''
Return liste mit all Subblocks-Koordinaten ()
'''
def get_all_blocks(num_sub_row, num_sub_column):
    all_blocks = []
    for i in range(num_sub_column):
        for j in range(num_sub_row):
            all_blocks.append((i,j))
    return all_blocks


'''
Return liste mit all Variablen-tupeln x_(i,j,k) in einem Bestimmten block
'''
def get_vars_in_block(all_vars, sub_row, sub_column, block_size):
    variables_in_block = []
    for var in all_vars:
        if(var[0] >= sub_row*block_size and var[0] < (sub_row+1)*block_size and 
            var[1] >= sub_column*block_size and var[1] < (sub_column+1)*block_size):
            variables_in_block.append(var)
    return variables_in_block


#make heatmap of QUBO
def get_qubo_heatmap(qubo, operations, output):
    line_width_qubo = 1/len(operations)
    df = pd.DataFrame(qubo, columns=operations, index=operations)
    sns.set_style("white")
    mask = np.tril(np.zeros_like(df)).astype(np.bool)
    mask[np.tril_indices_from(mask)] = True

    # Keep diagonal elements
    mask[np.diag_indices_from(mask)] = False

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(7, 6))

    # Generate a custom diverging colormap in Trumpf Color 0033BA
    color_map = sns.diverging_palette(255, 200, sep=10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio

    sns_plot = sns.heatmap(df, mask=mask, cmap=color_map, center=0,
                           square=True,
                            linewidths=line_width_qubo,
                           # cbar_kws={"shrink": .5}
                           )
    # save to file
    fig = sns_plot.get_figure()
    output += ".png"
    fig.savefig(output)

 


'''
H1 == Nur eine Zahl in jedes Kästchen
'''
def h1_penalty(weight, QUBO, prunned_list):
    for variable in prunned_list:
        i = prunned_list.index(variable)
        QUBO[i][i] += -weight
    
        for variable_2 in prunned_list:
            j = prunned_list.index(variable_2)
            if (variable_2[0] == variable[0] and variable_2[1] == variable[1]):
                if (variable_2[2] > variable[2]):
                    QUBO[i][j] += 2*weight
    return QUBO




'''
H2 == Each column-j cannot have any duplicate number
'''
def h2_penalty(weight, QUBO, prunned_list, all_vars, known_vars):
    #diags
    for variable in prunned_list:
        i = prunned_list.index(variable)
        QUBO[i][i] += -weight
    
    for first_var in all_vars:
        for second_var in all_vars:
            
            #wenn mind einer von ihnen unbekannt ist
            if ((not (first_var[0], first_var[1]) in known_vars) or (not (second_var[0],second_var[1]) in known_vars)):
                #und zwei indizies gleich
                if (first_var[1] == second_var[1] and first_var[2] == second_var[2]):
                    #summierte index j>i
                    if (first_var[0] > second_var[0]):
                        #erste unbekannt 
                        if ((not (first_var[0],first_var[1]) in known_vars) and ((second_var[0],second_var[1]) in known_vars)):
                            if (first_var in prunned_list):
                                i = prunned_list.index(first_var)
                                x_known = (second_var[2] == known_vars[(second_var[0], second_var[1])])
                                QUBO[i][i] += weight*(2*x_known)
                            
                        #zweite unbekannt 
                        if (((first_var[0],first_var[1]) in known_vars) and (not (second_var[0],second_var[1]) in known_vars)):
                            if (second_var in prunned_list):
                                i = prunned_list.index(second_var)
                                x_known = (first_var[2] == known_vars[(first_var[0],first_var[1])])
                                QUBO[i][i] += weight*(2*x_known)
                            
                        #erste und zweite unbekannt 
                        if ((not (first_var[0],first_var[1]) in known_vars) and (not (second_var[0],second_var[1]) in known_vars)):
                            if (second_var in prunned_list and first_var in prunned_list):
                                j = prunned_list.index(first_var)
                                i = prunned_list.index(second_var)
                                QUBO[i][j] += 2*weight
    return QUBO




'''
H3 == Each row-i cannot have any duplicate number
'''
def h3_penalty(weight, QUBO, prunned_list, all_vars, known_vars):
    for variable in prunned_list:
        i = prunned_list.index(variable)
        QUBO[i][i] += -weight
    
    for first_var in all_vars:
        for second_var in all_vars:
            #wenn mind einer von ihnen unbekannt ist
            if ((not (first_var[0],first_var[1]) in known_vars) or (not (second_var[0],second_var[1]) in known_vars)):
                #zwei indizies gleich
                if (first_var[0] == second_var[0] and first_var[2] == second_var[2]):
                    if (first_var[1] > second_var[1]):
                        
                        if ((not (first_var[0],first_var[1]) in known_vars) and ((second_var[0],second_var[1]) in known_vars)):
                            if (first_var in prunned_list):
                                i = prunned_list.index(first_var)
                                x_known = (second_var[2] == known_vars[(second_var[0],second_var[1])])
                                QUBO[i][i] += weight*(2*x_known)
                                #print(str(first_var) + '---' + str(second_var))
                    
                        if (((first_var[0],first_var[1]) in known_vars) and (not (second_var[0],second_var[1]) in known_vars)):
                            if (second_var in prunned_list):
                                i = prunned_list.index(second_var)
                                x_known = (first_var[2] == known_vars[(first_var[0],first_var[1])])
                                QUBO[i][i] += weight*(2*x_known)
                            
                        if ((not (first_var[0],first_var[1]) in known_vars) and (not (second_var[0],second_var[1]) in known_vars)):
                            if (second_var in prunned_list and first_var in prunned_list):
                                j = prunned_list.index(first_var)
                                i = prunned_list.index(second_var)
                        
                                QUBO[i][j] += 2*weight
    return QUBO




'''
H4 == Each of the nine 2x2 subgrids cannot have any duplicate number
'''
def h4_penalty(weight, QUBO, prunned_list, all_vars, known_vars, block_size, all_blocks):
    for block in all_blocks:
        variables_in_block = get_vars_in_block(all_vars, block[0], block[1], block_size)
        
        for var_1 in variables_in_block:
            #lin term
            if (var_1 in prunned_list):
                i = prunned_list.index(var_1)
                QUBO[i][i] += -weight
            
            #
            for var_2 in variables_in_block:
                if (variables_in_block.index(var_2)>variables_in_block.index(var_1)):
                    if (var_1[2] == var_2[2]):
                        #print(str(var_1) + '---' + str(var_2))
                        
                        #wenn mind einer var in den unbekannt ist
                        if ((not (var_1[0], var_1[1]) in known_vars) or (not (var_2[0], var_2[1]) in known_vars)):
                            
                            #zweite bekannt
                            if ((not (var_1[0],var_1[1]) in known_vars) and ((var_2[0],var_2[1]) in known_vars)):
                                if (var_1 in prunned_list):
                                    i = prunned_list.index(var_1)
                                    x_known = (var_2[2] == known_vars[(var_2[0], var_2[1])])
                                    #print(delta*(2*x_known))
                                    QUBO[i][i] += weight*(2*x_known)
                            
                            #erste bekannt
                            if (((var_1[0],var_1[1]) in known_vars) and (not (var_2[0],var_2[1]) in known_vars)):
                                if (var_2 in prunned_list):
                                    i = prunned_list.index(var_2)
                                    x_known = (var_1[2] == known_vars[(var_1[0],var_1[1])])
                                    QUBO[i][i] += weight*(2*x_known)
                                    #print(QUBO[0][0])
                                    #print(str(var_1) + '---' + str(var_2))
                                
                            #beide unbekannt
                            if ((not (var_1[0],var_1[1]) in known_vars) and (not (var_2[0],var_2[1]) in known_vars)):
                                if (var_1 in prunned_list and var_2 in prunned_list):
                                    j = prunned_list.index(var_2)
                                    i = prunned_list.index(var_1)
                                    QUBO[i][j] += 2*weight
    return QUBO
                

'''
für ein Sampleset-Vektor result return Sudoku-Array
'''
def visualize_solution(row_num, column_num, prunned_list, known_vars, result):
    for i, var in enumerate(prunned_list):
        if (int(result[i])==1):
            known_vars[(var[0],var[1])] = var[2]
    sudoku = np.zeros((row_num, column_num))
    for i in range(row_num):
        for j in range(column_num):
            sudoku[i][j] = known_vars[(i,j)] 
    return np.asmatrix(sudoku)


'''
little QBsolv needs a dictionary, not a matrix <3
'''
def matrix_to_dictionary(QUBO):
    n = len(QUBO)

    ## convert QUBO to dictionary 
    qubo_d = {}
    x1 = 0
    while x1 < n:
        x2 = x1
        while x2 < n:
            qubo_d[(x1, x2)] = int(QUBO[x1][x2])
            x2 += 1
        x1 += 1
    return qubo_d


#QBSolve
def on_QBsolve(qubo_dictionary):
    result = QBSolv().sample_qubo(qubo_dictionary)
    return result

#echtes HW
def on_DWave(QUBO, numr): 
    bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(QUBO)
    sampler = EmbeddingComposite(DWaveSampler())
       
    # reding num_reads responses from the sampler
    sampleset = sampler.sample(bqm, chain_strength=find_chstr(QUBO),num_reads=numr)
    return sampleset

#Find the Chain Strength following D Waves Problem Solving Handbook
def find_chstr(QUBO):
    chstr = QUBO.max() # Implementation parameter on the DWave QPU
    return chstr;


'''
For qiskit
'''

def QUBO_to_QuadraticProgram(QUBO):
    qubo = QuadraticProgram()
            
    ## übersetzen QUBO in Quadratic Program
    N = len(QUBO)
    for i in range(N):
       l='x'+str(i)
       qubo.binary_var(l)

    quad2={}
    for i in range(N):
        li='x'+str(i)
        for j in range(N):
            lj='x'+str(j)
            quad2.update({(li, lj): QUBO[i,j]})
    qubo.minimize(quadratic=quad2)
    return qubo


def on_qiskit(QuadraticProgram, backend, with_graphic = False):
    #convert QP zu Ising
    H, offset = QuadraticProgram.to_ising()
    # plot diagonal of matrix
    if (with_graphic):
        H_matrix = np.real(H.to_matrix())
        # plot diagonal of matrix
        opt_indices = list(np.where(H_matrix.diagonal() == min(H_matrix.diagonal())))[0]
        backend = Aer.get_backend('statevector_simulator')
        
    optimizer = COBYLA(maxiter=250)

    qaoa_mes = QAOA(optimizer=optimizer,  
                    quantum_instance=backend)
    result = qaoa_mes.compute_minimum_eigenvalue(H)

    print('optimal params:      ', result.optimal_parameters)
    print('optimal value:       ', result.optimal_value)
    
    qc = qaoa_mes.get_optimal_circuit()
    n = qc.num_qubits
    
    if (with_graphic):
        result.eigenstate
        probabilities = np.abs(result.eigenstate)**2
        plt.figure(figsize=(12, 5))
        plt.bar(range(2**n), probabilities)
        plt.bar(opt_indices, probabilities[opt_indices], color='g')
        plt.xticks(range(2**n), ['{0:04b}'.format(i) for i in range(2**n)], rotation=90, fontsize=12)
        plt.yticks(fontsize=14)
        plt.show()
        print('('+str(i)+') {0:05b}'.format(i) for i in range(2**n))#, probabilities)
    #np.abs(result.eigenstate)**2
    return result.optimal_value, qc


def measure_ciruit(qc, backend, shots):
    gate = qc.to_gate() 
    n = qc.num_qubits
    wires = range(n)

    qc_qaoa = QuantumCircuit(n, n)
    qc_qaoa.append(gate, wires)
    qc_qaoa.measure(wires, wires)
    display(qc_qaoa.draw('mpl'))
    
    job = execute(qc_qaoa, backend, shots = shots)
    plot_histogram(job.result().get_counts(qc_qaoa))
    return job.result().get_counts(qc_qaoa)



def get_costs(qc):
    pass_ = Unroller(['u3', 'cx'])
    pm = PassManager(pass_)
    new_circuit = pm.run(qc)

    ops = new_circuit.count_ops()
    print(ops)
    cost = ops['u3'] + 10 * ops['cx'] 
    print("Cost: " + str(cost))
    return cost
