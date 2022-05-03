from itertools import combinations_with_replacement 
from scipy.special import gamma, factorial
import numpy as np
import pandas as pd

def prams_count(dims, order, mode):
    if mode == 'sum':
        out = gamma(1 + dims + order)/(gamma(1 + dims) * gamma(1 + order))
    if mode == 'one':
        out = factorial(dims + order - 1)/(factorial(dims - 1)*factorial(order))
    return int(out)
    

def kmc(df, order, dt, lag, mode):
    assert len(df.shape) == 2, ValueError('time series must have (n, dims) shape')
    assert df.shape[0] > 0, ValueError('no data in time series')
    
    n, dims = df.shape
    
    n_free_params = prams_count(dims=dims, order=order, mode='sum')
    
    if mode == 'drift':
        ts = {}
        for i in range(dims):
            ts['x{}'.format(i+1)] = np.array(df[:, i])

        for i in range(1, dims+1):
            ts['y{0}'.format(i)] = ts['x{0}'.format(i)][lag:] - ts['x{0}'.format(i)][:(len(ts['x{0}'.format(i)])-lag)]
            ts['x{0}'.format(i)] = ts['x{0}'.format(i)][:(len(ts['x{0}'.format(i)])-lag)]

        A_matrix, A_const, B_vector = {}, {}, {}
        A, B = np.zeros((n_free_params, n_free_params)), np.zeros((dims, n_free_params))

        if order == 0:########################################################################################################################
            ########################################
            #############   A-Matrix   #############
            ########################################
            A = np.identity(dims)
            
            ########################################
            #############   B-Vector   #############
            ########################################
            for p in range(1, dims+1):
                B_vector['<y{0}>'.format(int(p))] = np.mean(ts['y{0}'.format(int(p))])
                
            ########################################
            ############   Generating   ############
            ########################################

            ########################################
            B_vector_values, B_vector_keys = np.array(list(B_vector.values())), np.array(list(B_vector))
            B = B_vector_values
            
            ########################################
            coeff_keys = np.array(['1'])
        
        ######################################################################################################################################
        if order == 1:########################################################################################################################
            comb_1 = np.array(list(combinations_with_replacement(np.arange(1, dims+1), 1)))
            n_terms_1 = comb_1.shape[0]
            ########################################
            #############   A-Const.   #############
            ########################################
            for i in range(1, dims+1):
                A_const['<x{0}>_o01'.format(i)] = np.mean(ts['x{0}'.format(i)])

            ########################################
            #############   A-Matrix   #############
            ########################################
            for i in range(1, dims+1):
                for j in range(1, dims+1):
                    A_matrix['<x{0}x{1}>_o11'.format(i, j)] = np.mean(ts['x{0}'.format(i)]*\
                                                                      ts['x{0}'.format(j)])

            ########################################
            #############   B-Vector   #############
            ########################################
            for p in range(1, dims+1):
                B_vector['<y{0}>'.format(int(p))] = np.mean(ts['y{0}'.format(int(p))])
                for n in range(n_terms_1):
                    i = comb_1[n]
                    B_vector['<y{0}x{1}>'.format(int(p), int(i))] = np.mean(ts['y{0}'.format(int(p))]*\
                                                                            ts['x{0}'.format(int(i))])

            ########################################
            ############   Generating   ############
            ########################################
            n_o1 = prams_count(dims=dims, order=1, mode='one')

            A_const_values, A_const_keys = np.array(list(A_const.values())), np.array(list(A_const))
            A_matrix_values, A_matrix_keys = np.array(list(A_matrix.values())), np.array(list(A_matrix))

            o11 = A_matrix_values[0 : n_o1*n_o1]
            o11 = o11.reshape(n_o1, n_o1)

            A[0, 0] = 1
            A[0, 1:] = A_const_values
            A[1:, 0] = A_const_values

            A[1:1+n_o1, 1:1+n_o1] = o11

            ########################################
            B_vector_values, B_vector_keys = np.array(list(B_vector.values())), np.array(list(B_vector))
            B = np.split(B_vector_values, dims)

            ########################################
            coeff_keys = np.array(['1'])

            for n in range(n_terms_1):
                i = comb_1[n]
                coeff_keys = np.append(coeff_keys, 'x{0}'.format(int(i)))

        ########################################################################################################################################
        elif order == 2:########################################################################################################################
            comb_1 = np.array(list(combinations_with_replacement(np.arange(1, dims+1), 1)))
            n_terms_1 = comb_1.shape[0]

            comb_2 = np.array(list(combinations_with_replacement(np.arange(1, dims+1), 2)))
            n_terms_2 = comb_2.shape[0]
            ########################################
            #############   A-Const.   #############
            ########################################
            for i in range(1, dims+1):
                A_const['<x{0}>_o01'.format(i)] = np.mean(ts['x{0}'.format(i)])

            for n in range(n_terms_2):
                i, j = comb_2[n]
                A_const['<x{0}x{1}>_o02'.format(int(i), int(j))] = np.mean(ts['x{0}'.format(int(i))]*\
                                                                           ts['x{0}'.format(int(j))])

            ########################################
            #############   A-Matrix   #############
            ########################################
            for i in range(1, dims+1):
                for j in range(1, dims+1):
                    A_matrix['<x{0}x{1}>_o11'.format(i, j)] = np.mean(ts['x{0}'.format(i)]*\
                                                                      ts['x{0}'.format(j)])

            for i in range(1, dims+1):
                for n in range(n_terms_2):
                    p, q = comb_2[n]
                    A_matrix['<x{0}x{1}x{2}>_o12'.format(i, int(p), int(q))] = np.mean(ts['x{0}'.format(i)]*\
                                                                                       ts['x{0}'.format(int(p))]*\
                                                                                       ts['x{0}'.format(int(q))])

            for n in range(n_terms_2):
                i, j = comb_2[n]
                for m in range(n_terms_2):
                    p, q = comb_2[m]
                    A_matrix['<x{0}x{1}x{2}x{3}>_o22'.format(int(i), int(j), int(p), int(q))] = np.mean(ts['x{0}'.format(int(i))]*\
                                                                                                        ts['x{0}'.format(int(j))]*\
                                                                                                        ts['x{0}'.format(int(p))]*\
                                                                                                        ts['x{0}'.format(int(q))])

            ########################################
            #############   B-Vector   #############
            ########################################
            for p in range(1, dims+1):
                B_vector['<y{0}>'.format(int(p))] = np.mean(ts['y{0}'.format(int(p))])
                for n in range(n_terms_1):
                    i = comb_1[n]
                    B_vector['<y{0}x{1}>'.format(int(p), int(i))] = np.mean(ts['y{0}'.format(int(p))]*\
                                                                            ts['x{0}'.format(int(i))])

                for n in range(n_terms_2):
                    i, j = comb_2[n]
                    B_vector['<y{0}x{1}x{2}>'.format(int(p), int(i), int(j))] = np.mean(ts['y{0}'.format(int(p))]*\
                                                                                        ts['x{0}'.format(int(i))]*\
                                                                                        ts['x{0}'.format(int(j))])

            ########################################
            ############   Generating   ############
            ########################################
            n_o1 = prams_count(dims=dims, order=1, mode='one')
            n_o2 = prams_count(dims=dims, order=2, mode='one')

            A_const_values, A_const_keys = np.array(list(A_const.values())), np.array(list(A_const))
            A_matrix_values, A_matrix_keys = np.array(list(A_matrix.values())), np.array(list(A_matrix))

            o11 = A_matrix_values[0 : n_o1*n_o1]
            o11 = o11.reshape(n_o1, n_o1)

            o12 = A_matrix_values[n_o1*n_o1 : n_o1*n_o1 + n_o1*n_o2]
            o12 = o12.reshape(n_o1, n_o2)

            o22 = A_matrix_values[n_o1*n_o1 + n_o1*n_o2 : n_o1*n_o1 + n_o1*n_o2 + n_o2*n_o2]
            o22 = o22.reshape(n_o2, n_o2)

            A[0, 0] = 1
            A[0, 1:] = A_const_values
            A[1:, 0] = A_const_values

            A[1:1+n_o1, 1:1+n_o1] = o11

            A[1:1+n_o1, 1+n_o1:1+n_o1+n_o2] = o12
            A[1+n_o1:1+n_o1+n_o2, 1:1+n_o1] = o12.T

            A[1+n_o1:1+n_o1+n_o2, 1+n_o1:1+n_o1+n_o2] = o22

            ########################################
            B_vector_values, B_vector_keys = np.array(list(B_vector.values())), np.array(list(B_vector))
            B = np.split(B_vector_values, dims)

            ########################################
            coeff_keys = np.array(['1'])

            for n in range(n_terms_1):
                i = comb_1[n]
                coeff_keys = np.append(coeff_keys, 'x{0}'.format(int(i)))

            for n in range(n_terms_2):
                i, j = comb_2[n]
                coeff_keys = np.append(coeff_keys, 'x{0}x{1}'.format(int(i), int(j)))

        #########################################################################################################################################
        elif order == 3: ########################################################################################################################
            comb_1 = np.array(list(combinations_with_replacement(np.arange(1, dims+1), 1)))
            n_terms_1 = comb_1.shape[0]

            comb_2 = np.array(list(combinations_with_replacement(np.arange(1, dims+1), 2)))
            n_terms_2 = comb_2.shape[0]

            comb_3 = np.array(list(combinations_with_replacement(np.arange(1, dims+1), 3)))
            n_terms_3 = comb_3.shape[0]     
            ########################################
            #############   A-Const.   #############
            ########################################
            for i in range(1, dims+1):
                A_const['<x{0}>_o01'.format(i)] = np.mean(ts['x{0}'.format(i)])

            for n in range(n_terms_2):
                i, j = comb_2[n]
                A_const['<x{0}x{1}>_o02'.format(int(i), int(j))] = np.mean(ts['x{0}'.format(int(i))]*\
                                                                           ts['x{0}'.format(int(j))])

            for n in range(n_terms_3):
                i, j, k = comb_3[n]
                A_const['<x{0}x{1}x{2}>_o03'.format(int(i), int(j), int(k))] = np.mean(ts['x{0}'.format(int(i))]*\
                                                                                       ts['x{0}'.format(int(j))]*\
                                                                                       ts['x{0}'.format(int(k))])
            ########################################
            #############   A-Matrix   #############
            ########################################
            for i in range(1, dims+1):
                for j in range(1, dims+1):
                    A_matrix['<x{0}x{1}>_o11'.format(i, j)] = np.mean(ts['x{0}'.format(i)]*\
                                                                      ts['x{0}'.format(j)])

            for i in range(1, dims+1):
                for n in range(n_terms_2):
                    p, q = comb_2[n]
                    A_matrix['<x{0}x{1}x{2}>_o12'.format(i, int(p), int(q))] = np.mean(ts['x{0}'.format(i)]*\
                                                                                       ts['x{0}'.format(int(p))]*\
                                                                                       ts['x{0}'.format(int(q))])

            for i in range(1, dims+1):
                for n in range(n_terms_3):
                    p, q, r = comb_3[n]
                    A_matrix['<x{0}x{1}x{2}x{3}>_o13'.format(i, int(p), int(q), int(r))] = np.mean(ts['x{0}'.format(i)]*\
                                                                                                   ts['x{0}'.format(int(p))]*\
                                                                                                   ts['x{0}'.format(int(q))]*\
                                                                                                   ts['x{0}'.format(int(r))])

            for n in range(n_terms_2):
                i, j = comb_2[n]
                for m in range(n_terms_2):
                    p, q = comb_2[m]
                    A_matrix['<x{0}x{1}x{2}x{3}>_o22'.format(int(i), int(j), int(p), int(q))] = np.mean(ts['x{0}'.format(int(i))]*\
                                                                                                        ts['x{0}'.format(int(j))]*\
                                                                                                        ts['x{0}'.format(int(p))]*\
                                                                                                        ts['x{0}'.format(int(q))])

            for n in range(n_terms_2):
                i, j = comb_2[n]
                for m in range(n_terms_3):
                    p, q, r = comb_3[m]
                    A_matrix['<x{0}x{1}x{2}x{3}x{4}>_o23'.format(int(i), int(j), int(p), int(q), int(r))] = np.mean(ts['x{0}'.format(int(i))]*\
                                                                                                                    ts['x{0}'.format(int(j))]*\
                                                                                                                    ts['x{0}'.format(int(p))]*\
                                                                                                                    ts['x{0}'.format(int(q))]*\
                                                                                                                    ts['x{0}'.format(int(r))])

            for n in range(n_terms_3):
                i, j, k = comb_3[n]
                for m in range(n_terms_3):
                    p, q, r = comb_3[m]
                    A_matrix['<x{0}x{1}x{2}x{3}x{4}x{5}>_o33'.format(int(i), int(j), int(k), int(p), int(q), int(r))] = np.mean(ts['x{0}'.format(int(i))]*\
                                                                                                                                ts['x{0}'.format(int(j))]*\
                                                                                                                                ts['x{0}'.format(int(k))]*\
                                                                                                                                ts['x{0}'.format(int(p))]*\
                                                                                                                                ts['x{0}'.format(int(q))]*\
                                                                                                                                ts['x{0}'.format(int(r))])
            ########################################
            #############   B-Vector   #############
            ########################################
            for p in range(1, dims+1):
                B_vector['<y{0}>'.format(int(p))] = np.mean(ts['y{0}'.format(int(p))])
                for n in range(n_terms_1):
                    i = comb_1[n]
                    B_vector['<y{0}x{1}>'.format(int(p), int(i))] = np.mean(ts['y{0}'.format(int(p))]*\
                                                                            ts['x{0}'.format(int(i))])

                for n in range(n_terms_2):
                    i, j = comb_2[n]
                    B_vector['<y{0}x{1}x{2}>'.format(int(p), int(i), int(j))] = np.mean(ts['y{0}'.format(int(p))]*\
                                                                                        ts['x{0}'.format(int(i))]*\
                                                                                        ts['x{0}'.format(int(j))])
                for n in range(n_terms_3):
                    i, j, k = comb_3[n]
                    B_vector['<y{0}x{1}x{2}x{3}>'.format(int(p), int(i), int(j), int(k))] = np.mean(ts['y{0}'.format(int(p))]*\
                                                                                                    ts['x{0}'.format(int(i))]*\
                                                                                                    ts['x{0}'.format(int(j))]*\
                                                                                                    ts['x{0}'.format(int(k))])

            ########################################
            ############   Generating   ############
            ########################################
            n_o1 = prams_count(dims=dims, order=1, mode='one')
            n_o2 = prams_count(dims=dims, order=2, mode='one')
            n_o3 = prams_count(dims=dims, order=3, mode='one')

            A_const_values, A_const_keys = np.array(list(A_const.values())), np.array(list(A_const))
            A_matrix_values, A_matrix_keys = np.array(list(A_matrix.values())), np.array(list(A_matrix))

            o11 = A_matrix_values[0 : n_o1*n_o1]
            o11 = o11.reshape(n_o1, n_o1)

            o12 = A_matrix_values[n_o1*n_o1 : n_o1*n_o1 + n_o1*n_o2]
            o12 = o12.reshape(n_o1, n_o2)

            o13 = A_matrix_values[n_o1*n_o1 + n_o1*n_o2 : n_o1*n_o1 + n_o1*n_o2 + n_o1*n_o3]
            o13 = o13.reshape(n_o1, n_o3)

            o22 = A_matrix_values[n_o1*n_o1 + n_o1*n_o2 + n_o1*n_o3 : n_o1*n_o1 + n_o1*n_o2 + n_o1*n_o3 + n_o2*n_o2]
            o22 = o22.reshape(n_o2, n_o2)

            o23 = A_matrix_values[n_o1*n_o1 + n_o1*n_o2 + n_o1*n_o3 + n_o2*n_o2 : n_o1*n_o1 + n_o1*n_o2 + n_o1*n_o3 + n_o2*n_o2 + n_o2*n_o3]
            o23 = o23.reshape(n_o2, n_o3)        

            o33 = A_matrix_values[n_o1*n_o1 + n_o1*n_o2 + n_o1*n_o3 + n_o2*n_o2 + n_o2*n_o3 : n_o1*n_o1 + n_o1*n_o2 + n_o1*n_o3 + n_o2*n_o2 + n_o2*n_o3 + n_o3*n_o3]
            o33 = o33.reshape(n_o3, n_o3)

            A[0, 0] = 1
            A[0, 1:] = A_const_values
            A[1:, 0] = A_const_values

            A[1 : 1+n_o1, 1 : 1+n_o1] = o11

            A[1 : 1+n_o1, 1+n_o1 : 1+n_o1+n_o2] = o12
            A[1+n_o1 : 1+n_o1+n_o2, 1 : 1+n_o1] = o12.T

            A[1 : 1+n_o1, 1+n_o1+n_o2 : 1+n_o1+n_o2+n_o3] = o13
            A[1+n_o1+n_o2 : 1+n_o1+n_o2+n_o3, 1 : 1+n_o1] = o13.T

            A[1+n_o1 : 1+n_o1+n_o2, 1+n_o1 : 1+n_o1+n_o2] = o22

            A[1+n_o1 : 1+n_o1+n_o2, 1+n_o1+n_o2 : 1+n_o1+n_o2+n_o3] = o23
            A[1+n_o1+n_o2 : 1+n_o1+n_o2+n_o3, 1+n_o1 : 1+n_o1+n_o2] = o23.T

            A[1+n_o1+n_o2 : 1+n_o1+n_o2+n_o3, 1+n_o1+n_o2 : 1+n_o1+n_o2+n_o3] = o33

            ########################################
            B_vector_values, B_vector_keys = np.array(list(B_vector.values())), np.array(list(B_vector))
            B = np.split(B_vector_values, dims)

            ########################################
            coeff_keys = np.array(['1'])

            for n in range(n_terms_1):
                i = comb_1[n]
                coeff_keys = np.append(coeff_keys, 'x{0}'.format(int(i)))

            for n in range(n_terms_2):
                i, j = comb_2[n]
                coeff_keys = np.append(coeff_keys, 'x{0}x{1}'.format(int(i), int(j)))

            for n in range(n_terms_3):
                i, j, k = comb_3[n]
                coeff_keys = np.append(coeff_keys, 'x{0}x{1}x{2}'.format(int(i), int(j), int(k)))

        else:
            raise ValueError('order must be >= 0 and the maximum order is 3')

        A, B = np.array(A), np.array(B)
        coeff = np.linalg.solve(A, B.T) * 1/(lag*dt)
    
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################

    elif mode == 'diffusion':
        ts = {}
        for i in range(1, dims+1):
            ts['x{0}'.format(i)] = np.array(df[:, i-1])

        for i in range(1, dims+1):
            ts['y{0}'.format(i)] = ts['x{0}'.format(i)][lag:] - ts['x{0}'.format(i)][:(len(ts['x{0}'.format(i)])-lag)]
            ts['x{0}'.format(i)] = ts['x{0}'.format(i)][:(len(ts['x{0}'.format(i)])-lag)]

        A_matrix, A_const, B_vector = {}, {}, {}
        A, B = np.zeros((n_free_params, n_free_params)), np.zeros((dims*dims, n_free_params))

        if order == 0:########################################################################################################################
            ########################################
            #############   A-Matrix   #############
            ########################################
            A = np.identity(dims)
            
            ########################################
            #############   B-Vector   #############
            ########################################
            for p in range(1, dims+1):
                B_vector['<y{0}>'.format(int(p))] = np.mean(ts['y{0}'.format(int(p))])
                
            ########################################
            ############   Generating   ############
            ########################################

            ########################################
            B_vector_values, B_vector_keys = np.array(list(B_vector.values())), np.array(list(B_vector))
            B = B_vector_values
            
            ########################################
            coeff_keys = np.array(['1'])
        
        ######################################################################################################################################
        if order == 1:########################################################################################################################
            comb_1 = np.array(list(combinations_with_replacement(np.arange(1, dims+1), 1)))
            n_terms_1 = comb_1.shape[0]
            ########################################
            #############   A-Const.   #############
            ########################################
            for i in range(1, dims+1):
                A_const['<x{0}>_o01'.format(i)] = np.mean(ts['x{0}'.format(i)])

            ########################################
            #############   A-Matrix   #############
            ########################################
            for i in range(1, dims+1):
                for j in range(1, dims+1):
                    A_matrix['<x{0}x{1}>_o11'.format(i, j)] = np.mean(ts['x{0}'.format(i)]*\
                                                                      ts['x{0}'.format(j)])

            ########################################
            #############   B-Vector   #############
            ########################################
            for p in range(1, dims+1):
                for q in range(1, dims+1):
                    B_vector['<y{0}{1}>'.format(int(p), int(q))] = np.mean(ts['y{0}'.format(int(p))]*\
                                                                           ts['y{0}'.format(int(q))])
                    for n in range(n_terms_1):
                        i = comb_1[n]
                        B_vector['<y{0}{1}x{2}>'.format(int(p), int(q), int(i))] = np.mean(ts['y{0}'.format(int(p))]*\
                                                                                           ts['y{0}'.format(int(q))]*\
                                                                                           ts['x{0}'.format(int(i))])

            ########################################
            ############   Generating   ############
            ########################################
            n_o1 = prams_count(dims=dims, order=1, mode='one')

            A_const_values, A_const_keys = np.array(list(A_const.values())), np.array(list(A_const))
            A_matrix_values, A_matrix_keys = np.array(list(A_matrix.values())), np.array(list(A_matrix))

            o11 = A_matrix_values[0 : n_o1*n_o1]
            o11 = o11.reshape(n_o1, n_o1)

            A[0, 0] = 1
            A[0, 1:] = A_const_values
            A[1:, 0] = A_const_values

            A[1:1+n_o1, 1:1+n_o1] = o11

            ########################################
            B_vector_values, B_vector_keys = np.array(list(B_vector.values())), np.array(list(B_vector))
            B = np.split(B_vector_values, dims*dims)

            ########################################
            coeff_keys = np.array(['1'])

            for n in range(n_terms_1):
                i = comb_1[n]
                coeff_keys = np.append(coeff_keys, 'x{0}'.format(int(i)))

        ########################################################################################################################################
        elif order == 2:########################################################################################################################
            comb_1 = np.array(list(combinations_with_replacement(np.arange(1, dims+1), 1)))
            n_terms_1 = comb_1.shape[0]
            comb_2 = np.array(list(combinations_with_replacement(np.arange(1, dims+1), 2)))
            n_terms_2 = comb_2.shape[0]
            ########################################
            #############   A-Const.   #############
            ########################################
            for i in range(1, dims+1):
                A_const['<x{0}>_o01'.format(i)] = np.mean(ts['x{0}'.format(i)])

            for n in range(n_terms_2):
                i, j = comb_2[n]
                A_const['<x{0}x{1}>_o02'.format(int(i), int(j))] = np.mean(ts['x{0}'.format(int(i))]*\
                                                                           ts['x{0}'.format(int(j))])

            ########################################
            #############   A-Matrix   #############
            ########################################
            for i in range(1, dims+1):
                for j in range(1, dims+1):
                    A_matrix['<x{0}x{1}>_o11'.format(i, j)] = np.mean(ts['x{0}'.format(i)]*\
                                                                      ts['x{0}'.format(j)])

            for i in range(1, dims+1):
                for n in range(n_terms_2):
                    p, q = comb_2[n]
                    A_matrix['<x{0}x{1}x{2}>_o12'.format(i, int(p), int(q))] = np.mean(ts['x{0}'.format(i)]*\
                                                                                       ts['x{0}'.format(int(p))]*\
                                                                                       ts['x{0}'.format(int(q))])

            for n in range(n_terms_2):
                i, j = comb_2[n]
                for m in range(n_terms_2):
                    p, q = comb_2[m]
                    A_matrix['<x{0}x{1}x{2}x{3}>_o22'.format(int(i), int(j), int(p), int(q))] = np.mean(ts['x{0}'.format(int(i))]*\
                                                                                                        ts['x{0}'.format(int(j))]*\
                                                                                                        ts['x{0}'.format(int(p))]*\
                                                                                                        ts['x{0}'.format(int(q))])

            ########################################
            #############   B-Vector   #############
            ########################################
            for p in range(1, dims+1):
                for q in range(1, dims+1):
                    B_vector['<y{0}{1}>'.format(int(p), int(q))] = np.mean(ts['y{0}'.format(int(p))]*\
                                                                           ts['y{0}'.format(int(q))])
                    for n in range(n_terms_1):
                        i = comb_1[n]
                        B_vector['<y{0}{1}x{2}>'.format(int(p), int(q), int(i))] = np.mean(ts['y{0}'.format(int(p))]*\
                                                                                           ts['y{0}'.format(int(q))]*\
                                                                                           ts['x{0}'.format(int(i))])

                    for n in range(n_terms_2):
                        i, j = comb_2[n]
                        B_vector['<y{0}{1}x{2}x{3}>'.format(int(p), int(q), int(i), int(j))] = np.mean(ts['y{0}'.format(int(p))]*\
                                                                                                       ts['y{0}'.format(int(q))]*\
                                                                                                       ts['x{0}'.format(int(i))]*\
                                                                                                       ts['x{0}'.format(int(j))])

            ########################################
            ############   Generating   ############
            ########################################
            n_o1 = prams_count(dims=dims, order=1, mode='one')
            n_o2 = prams_count(dims=dims, order=2, mode='one')

            A_const_values, A_const_keys = np.array(list(A_const.values())), np.array(list(A_const))
            A_matrix_values, A_matrix_keys = np.array(list(A_matrix.values())), np.array(list(A_matrix))

            o11 = A_matrix_values[0 : n_o1*n_o1]
            o11 = o11.reshape(n_o1, n_o1)

            o12 = A_matrix_values[n_o1*n_o1 : n_o1*n_o1 + n_o1*n_o2]
            o12 = o12.reshape(n_o1, n_o2)

            o22 = A_matrix_values[n_o1*n_o1 + n_o1*n_o2 : n_o1*n_o1 + n_o1*n_o2 + n_o2*n_o2]
            o22 = o22.reshape(n_o2, n_o2)

            A[0, 0] = 1
            A[0, 1:] = A_const_values
            A[1:, 0] = A_const_values

            A[1:1+n_o1, 1:1+n_o1] = o11

            A[1:1+n_o1, 1+n_o1:1+n_o1+n_o2] = o12
            A[1+n_o1:1+n_o1+n_o2, 1:1+n_o1] = o12.T

            A[1+n_o1:1+n_o1+n_o2, 1+n_o1:1+n_o1+n_o2] = o22

            ########################################
            B_vector_values, B_vector_keys = np.array(list(B_vector.values())), np.array(list(B_vector))
            B = np.split(B_vector_values, dims*dims)

            ########################################
            coeff_keys = np.array(['1'])

            for n in range(n_terms_1):
                i = comb_1[n]
                coeff_keys = np.append(coeff_keys, 'x{0}'.format(int(i)))

            for n in range(n_terms_2):
                i, j = comb_2[n]
                coeff_keys = np.append(coeff_keys, 'x{0}x{1}'.format(int(i), int(j)))

        #########################################################################################################################################
        elif order == 3: ########################################################################################################################
            comb_1 = np.array(list(combinations_with_replacement(np.arange(1, dims+1), 1)))
            n_terms_1 = comb_1.shape[0]

            comb_2 = np.array(list(combinations_with_replacement(np.arange(1, dims+1), 2)))
            n_terms_2 = comb_2.shape[0]

            comb_3 = np.array(list(combinations_with_replacement(np.arange(1, dims+1), 3)))
            n_terms_3 = comb_3.shape[0]     
            ########################################
            #############   A-Const.   #############
            ########################################
            for i in range(1, dims+1):
                A_const['<x{0}>_o01'.format(i)] = np.mean(ts['x{0}'.format(i)])

            for n in range(n_terms_2):
                i, j = comb_2[n]
                A_const['<x{0}x{1}>_o02'.format(int(i), int(j))] = np.mean(ts['x{0}'.format(int(i))]*\
                                                                           ts['x{0}'.format(int(j))])

            for n in range(n_terms_3):
                i, j, k = comb_3[n]
                A_const['<x{0}x{1}x{2}>_o03'.format(int(i), int(j), int(k))] = np.mean(ts['x{0}'.format(int(i))]*\
                                                                                       ts['x{0}'.format(int(j))]*\
                                                                                       ts['x{0}'.format(int(k))])
            ########################################
            #############   A-Matrix   #############
            ########################################
            for i in range(1, dims+1):
                for j in range(1, dims+1):
                    A_matrix['<x{0}x{1}>_o11'.format(i, j)] = np.mean(ts['x{0}'.format(i)]*\
                                                                      ts['x{0}'.format(j)])

            for i in range(1, dims+1):
                for n in range(n_terms_2):
                    p, q = comb_2[n]
                    A_matrix['<x{0}x{1}x{2}>_o12'.format(i, int(p), int(q))] = np.mean(ts['x{0}'.format(i)]*\
                                                                                       ts['x{0}'.format(int(p))]*\
                                                                                       ts['x{0}'.format(int(q))])

            for i in range(1, dims+1):
                for n in range(n_terms_3):
                    p, q, r = comb_3[n]
                    A_matrix['<x{0}x{1}x{2}x{3}>_o13'.format(i, int(p), int(q), int(r))] = np.mean(ts['x{0}'.format(i)]*\
                                                                                                   ts['x{0}'.format(int(p))]*\
                                                                                                   ts['x{0}'.format(int(q))]*\
                                                                                                   ts['x{0}'.format(int(r))])

            for n in range(n_terms_2):
                i, j = comb_2[n]
                for m in range(n_terms_2):
                    p, q = comb_2[m]
                    A_matrix['<x{0}x{1}x{2}x{3}>_o22'.format(int(i), int(j), int(p), int(q))] = np.mean(ts['x{0}'.format(int(i))]*\
                                                                                                        ts['x{0}'.format(int(j))]*\
                                                                                                        ts['x{0}'.format(int(p))]*\
                                                                                                        ts['x{0}'.format(int(q))])

            for n in range(n_terms_2):
                i, j = comb_2[n]
                for m in range(n_terms_3):
                    p, q, r = comb_3[m]
                    A_matrix['<x{0}x{1}x{2}x{3}x{4}>_o23'.format(int(i), int(j), int(p), int(q), int(r))] = np.mean(ts['x{0}'.format(int(i))]*\
                                                                                                                    ts['x{0}'.format(int(j))]*\
                                                                                                                    ts['x{0}'.format(int(p))]*\
                                                                                                                    ts['x{0}'.format(int(q))]*\
                                                                                                                    ts['x{0}'.format(int(r))])

            for n in range(n_terms_3):
                i, j, k = comb_3[n]
                for m in range(n_terms_3):
                    p, q, r = comb_3[m]
                    A_matrix['<x{0}x{1}x{2}x{3}x{4}x{5}>_o33'.format(int(i), int(j), int(k), int(p), int(q), int(r))] = np.mean(ts['x{0}'.format(int(i))]*\
                                                                                                                                ts['x{0}'.format(int(j))]*\
                                                                                                                                ts['x{0}'.format(int(k))]*\
                                                                                                                                ts['x{0}'.format(int(p))]*\
                                                                                                                                ts['x{0}'.format(int(q))]*\
                                                                                                                                ts['x{0}'.format(int(r))])
            ########################################
            #############   B-Vector   #############
            ########################################
            for p in range(1, dims+1):
                for q in range(1, dims+1):
                    B_vector['<y{0}{1}>'.format(int(p), int(q))] = np.mean(ts['y{0}'.format(int(p))]*\
                                                                           ts['y{0}'.format(int(q))])
                    for n in range(n_terms_1):
                        i = comb_1[n]
                        B_vector['<y{0}{1}x{2}>'.format(int(p), int(q), int(i))] = np.mean(ts['y{0}'.format(int(p))]*\
                                                                                           ts['y{0}'.format(int(q))]*\
                                                                                           ts['x{0}'.format(int(i))])

                    for n in range(n_terms_2):
                        i, j = comb_2[n]
                        B_vector['<y{0}{1}x{2}x{3}>'.format(int(p), int(q), int(i), int(j))] = np.mean(ts['y{0}'.format(int(p))]*\
                                                                                                      ts['y{0}'.format(int(q))]*\
                                                                                                      ts['x{0}'.format(int(i))]*\
                                                                                                      ts['x{0}'.format(int(j))])
                    for n in range(n_terms_3):
                        i, j, k = comb_3[n]
                        B_vector['<y{0}{1}x{2}x{3}x{4}>'.format(int(p), int(q), int(i), int(j), int(k))] = np.mean(ts['y{0}'.format(int(p))]*\
                                                                                                                   ts['y{0}'.format(int(q))]*\
                                                                                                                   ts['x{0}'.format(int(i))]*\
                                                                                                                   ts['x{0}'.format(int(j))]*\
                                                                                                                   ts['x{0}'.format(int(k))])

            ########################################
            ############   Generating   ############
            ########################################
            n_o1 = prams_count(dims=dims, order=1, mode='one')
            n_o2 = prams_count(dims=dims, order=2, mode='one')
            n_o3 = prams_count(dims=dims, order=3, mode='one')

            A_const_values, A_const_keys = np.array(list(A_const.values())), np.array(list(A_const))
            A_matrix_values, A_matrix_keys = np.array(list(A_matrix.values())), np.array(list(A_matrix))

            o11 = A_matrix_values[0 : n_o1*n_o1]
            o11 = o11.reshape(n_o1, n_o1)

            o12 = A_matrix_values[n_o1*n_o1 : n_o1*n_o1 + n_o1*n_o2]
            o12 = o12.reshape(n_o1, n_o2)

            o13 = A_matrix_values[n_o1*n_o1 + n_o1*n_o2 : n_o1*n_o1 + n_o1*n_o2 + n_o1*n_o3]
            o13 = o13.reshape(n_o1, n_o3)

            o22 = A_matrix_values[n_o1*n_o1 + n_o1*n_o2 + n_o1*n_o3 : n_o1*n_o1 + n_o1*n_o2 + n_o1*n_o3 + n_o2*n_o2]
            o22 = o22.reshape(n_o2, n_o2)

            o23 = A_matrix_values[n_o1*n_o1 + n_o1*n_o2 + n_o1*n_o3 + n_o2*n_o2 : n_o1*n_o1 + n_o1*n_o2 + n_o1*n_o3 + n_o2*n_o2 + n_o2*n_o3]
            o23 = o23.reshape(n_o2, n_o3)        

            o33 = A_matrix_values[n_o1*n_o1 + n_o1*n_o2 + n_o1*n_o3 + n_o2*n_o2 + n_o2*n_o3 : n_o1*n_o1 + n_o1*n_o2 + n_o1*n_o3 + n_o2*n_o2 + n_o2*n_o3 + n_o3*n_o3]
            o33 = o33.reshape(n_o3, n_o3)

            A[0, 0] = 1
            A[0, 1:] = A_const_values
            A[1:, 0] = A_const_values

            A[1 : 1+n_o1, 1 : 1+n_o1] = o11

            A[1 : 1+n_o1, 1+n_o1 : 1+n_o1+n_o2] = o12
            A[1+n_o1 : 1+n_o1+n_o2, 1 : 1+n_o1] = o12.T

            A[1 : 1+n_o1, 1+n_o1+n_o2 : 1+n_o1+n_o2+n_o3] = o13
            A[1+n_o1+n_o2 : 1+n_o1+n_o2+n_o3, 1 : 1+n_o1] = o13.T

            A[1+n_o1 : 1+n_o1+n_o2, 1+n_o1 : 1+n_o1+n_o2] = o22

            A[1+n_o1 : 1+n_o1+n_o2, 1+n_o1+n_o2 : 1+n_o1+n_o2+n_o3] = o23
            A[1+n_o1+n_o2 : 1+n_o1+n_o2+n_o3, 1+n_o1 : 1+n_o1+n_o2] = o23.T

            A[1+n_o1+n_o2 : 1+n_o1+n_o2+n_o3, 1+n_o1+n_o2 : 1+n_o1+n_o2+n_o3] = o33

            ########################################
            B_vector_values, B_vector_keys = np.array(list(B_vector.values())), np.array(list(B_vector))
            B = np.split(B_vector_values, dims*dims)
            
            ########################################
            coeff_keys = np.array(['1'])

            for n in range(n_terms_1):
                i = comb_1[n]
                coeff_keys = np.append(coeff_keys, 'x{0}'.format(int(i)))

            for n in range(n_terms_2):
                i, j = comb_2[n]
                coeff_keys = np.append(coeff_keys, 'x{0}x{1}'.format(int(i), int(j)))

            for n in range(n_terms_3):
                i, j, k = comb_3[n]
                coeff_keys = np.append(coeff_keys, 'x{0}x{1}x{2}'.format(int(i), int(j), int(k)))

        else:
            raise ValueError('order must be >= 0 and the maximum order is 3')

        A, B = np.array(A), np.array(B)
        coeff = np.linalg.solve(A, B.T) * 1/(lag*dt)
        
    else:
        raise ValueError('mode must be drift or diffusion')
    return(coeff, coeff_keys)

