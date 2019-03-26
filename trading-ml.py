# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 13:17:12 2017

@author: argonalyst
"""

import random
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LinearRegression
from sklearn import ensemble
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
import time

def define_best_strategy(n, volume, transaction_fee, strategies):
    
    #preços para treinar o algoritmo
    prices = np.array([])
    with open(r'prices-trandingML.txt', 'r') as f:
        lines = f.readlines()

        for i in lines:
            prices = np.append(prices, float(i))
    
    file_result_read = open('final_result.txt', 'r') 
    final_result = file_result_read.read() 
    final_result = float(final_result)    

    final_profitability = np.array([])
    with open('final_profitability.txt', 'r') as f:
        lines = f.readlines()        
        for i in lines:            
            final_profitability = np.append(final_profitability, float(i))

    final_target = np.array([])

    #roda varias vezes o mesmo negocio gerando numeros aleatorios para maximizar o rendimento dada a lista de preços    
    for x in range(0,200):

        target = np.array([])
        profitability = np.array([])

        #gera operações aleatórias de compra e venda
        for i in prices:
            count_buy = 0
            count_sell = 0
            a = (random.choice(strategies))
            count_buy = np.count_nonzero(target == 1)
            count_sell = np.count_nonzero(target == 2)
    
            check = False
            if a == 0:
                check = True
                target = np.append(target, float(a))
        
            if a == 1 and (count_buy == count_sell or count_buy == 0):
                check = True
                target = np.append(target, float(a))
        
            
            if a == 2 and count_buy > count_sell:
                check = True
                target = np.append(target, float(a))
                
            if check == False:
                strategy = 0
                target = np.append(target, float(strategy))

        b_price = 0
        s_price = 0
        result_total = 0
        
        operation_check_buy = False
        operation_check_sell = False
        
        #cacula o rendimento das operações de compra e venda
        for x, y in zip(prices, target):

            should_break = False
            if y == 1:
                operation_check_buy = True
                b_price = (x * volume) + transaction_fee
            if y == 2:
                operation_check_sell = True
                s_price = (x * volume) - transaction_fee
                
            if operation_check_buy and operation_check_sell == True:
                result = s_price - b_price
                result_total = result_total + result
                operation_check_buy = False
                operation_check_sell = False
                profitability = np.append(profitability, float(result_total))

        #marca a combinação que tem maior rendimento
        if result_total > final_result:   
            final_result = result_total
            final_result_file = open('final_result.txt', 'w')
            final_result_file.write("%s" % final_result)
            final_result_file.close()
            
            final_target = target
            thefile = open('final_target.txt', 'w')
            for item in final_target:
                thefile.write("%s\n" % item)
            thefile.close()
            
            final_profitability = profitability
            profitability_file = open('final_profitability.txt', 'w')
            for item in final_profitability:
                profitability_file.write("%s\n" % item)
            profitability_file.close()    
    
    final_target = np.genfromtxt("final_target.txt")
    print final_target
    final_target = final_target[n:]
    
    #printa gráficos
    x1 = final_profitability
    x = np.array(x1)
    plt.plot(x)
    plt.show()
    print x1[-1]-x1[0]
    
    x1 = prices
    x = np.array(x1)
    plt.plot(x)
    plt.show()
    print x1[-1]-x1[0]

#################    

    index = 0
    p_array = np.array([0])
    
    for i in prices:    
        z = prices[index-n:index]    
        index_z = 0    
        for x in z:
           p = (z[index_z]*100)/z[0]       
           p_array = np.append(p_array, p)
           index_z+=1
        index+=1
        
    p_array = np.delete(p_array, 0)
    sliced_prices = p_array.reshape(-1, n)

###################

    #------------------------------------------------
    X = sliced_prices
    y = final_target
    
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    #model = KNeighborsClassifier()
    #model = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2, learning_rate = 0.1, loss = 'ls')
    model.fit(X, y)    
    
###################    

#novos dados para analisar ------

    prices_new = np.array([])
    with open(r'prices-trandingML.txt', 'r') as f:
        lines = f.readlines()        
        for i in lines:            
            prices_new = np.append(prices_new, float(i))
    
    
    x1 = prices_new
    x = np.array(x1)
    plt.plot(x)
    plt.show()
    print x1[-1]-x1[0]
    
    #transforma os preços em base 100

    index = 0
    p_array = np.array([0])
    
    for i in prices:    
        z = prices_new[index-n:index]    
        index_z = 0    
        for x in z:
           p = (z[index_z]*100)/z[0]       
           p_array = np.append(p_array, p)
           index_z+=1
        index+=1
    p_array = np.delete(p_array, 0)
    sliced_prices_parsed_new = p_array.reshape(-1, n)

######################

    #simula trading usando o resultado da decisão do machine learning
    predictions_vector = np.array([])
    for x in sliced_prices_parsed_new:
        X_test = [x]
    
        predictions = model.predict(X_test)
        
        count_buy = 0
        count_sell = 0
        a = predictions[0]
        count_buy = np.count_nonzero(predictions_vector == 1)
        count_sell = np.count_nonzero(predictions_vector == 2)
    
        check = False
        if a == 0:
            check = True
            predictions_vector = np.append(predictions_vector, a)
    
        if a == 1 and (count_buy == count_sell or count_buy == 0):
            check = True
            predictions_vector = np.append(predictions_vector, a)
    
        
        if a == 2 and count_buy > count_sell:
            check = True
            predictions_vector = np.append(predictions_vector, a)
            
        if check == False:
            strategy = 0
            predictions_vector = np.append(predictions_vector, strategy)
    
###############################
    
    #calcula o resultado do trading
    operation_check_buy = False
    operation_check_sell = False
    
    profitability_testdata = np.array([])
    result_total_testdata = 0
    
    b_price = 0
    s_price = 0
    for x, y in zip(prices_new, predictions_vector):
    
        if y == 1:
            operation_check_buy = True
            b_price = x
        if y == 2:
            operation_check_sell = True
            s_price = x
            
        if operation_check_buy and operation_check_sell == True:
            result_testdata = s_price - b_price
            result_total_testdata = result_total_testdata + result_testdata            
            operation_check_buy = False
            operation_check_sell = False
            profitability_testdata = np.append(profitability_testdata, result_total_testdata)
          
        final_result_testdata = result_total_testdata
        final_target_testdata = profitability_testdata
        final_profitability_testdata = profitability_testdata
    
    x1 = final_profitability_testdata
    
    x = np.array(x1)
    
    plt.plot(x)
    plt.show()
    print x1[-1]-x1[0]

def main():
    n = 15
    volume = 1
    transaction_fee = 0
    strategies = [0,1,2]
    
    define_best_strategy(n, volume, transaction_fee, strategies)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))