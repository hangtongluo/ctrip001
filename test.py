# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 19:26:31 2017

@author: Administrator
"""

#import sys
#if __name__ == "__main__":
#    # 读取第一行的n
#    n = int(sys.stdin.readline().strip())
#    # 读取每一行
#    line = sys.stdin.readline().strip()
#    # 把每一行的数字分隔后转化成int列表
#    values = list(map(int, line.split()))
#    #排序数据
#    sort_values = sorted(values)
#    #找到距离
#    d = sort_values[0] - sort_values[1]
#    #判断每个元素
#    for i in range(n):
#        if abs(sort_values[i + 1] - sort_values[i]) != abs(d):
#            print('Impossible')
#            #return 'Impossible'
#    else:
#        print('possible')
#        #return 'possible


#n = input()
#ran = list(map(int,input().split(' ')))
#ran = sorted(set(ran))
#ran = list(map(str,ran))
#length = len(ran)
#print(length)
#print(" ".join(ran))



#import sys
#  
#for line in sys.stdin:
#    n = int(line.readline().strip())
#    ran = list(map(int,line.readline().strip().split(' ')))
#    ran = sorted(set(ran))
#    ran = list(map(str,ran))
#    length = len(ran)
#    print(length)
#    print(" ".join(ran))
#
#
#
#
#import sys
# 
#for line in sys.stdin:
#    input1 = list(map(int,line.readline().strip().split(' ')))
#    n = input[0]
#    r = input[1]
#    avg = input[2]
#    values = [[] for i in range(n)]
#    for i in range(n):
#        ran = list(map(int,line.readline().strip().split(' ')))
#        values[i].append(ran[1])
#        values[i].append(ran[0])
#
#
#
#def printMatrix(mat, n, m):
##    # write code here
##    temp = []
##    for i in range(n):
##        if i % 2 == 0:
##            temp = temp + mat[i]
##        else:
##            temp = temp + mat[i][::-1]
##    
##    return temp
#    for i in range(n):
#        if i % 2 == 1:
#            mat[i] = mat[i][::-1]
#    return [x for y in mat for x in y]
#
#
#mat = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
#n = 4
#m = 3
#
#
#print(printMatrix(mat, n, m))
#
#
#
#def rotateMatrix(mat, n):
#    # write code here
##    temp1 = []
##    for i in range(n):
##        temp2 = []
##        for j in range(n):
##            temp2.append(mat[j][i])
##        temp1.append(temp2[::-1])
##    
##    return temp1
#    return [list(x[::-1]) for x in zip(*mat)]
#
#mat = [[1,2,3],[4,5,6],[7,8,9]]
#n = 3
#print(rotateMatrix(mat, n))








# -*- coding: UTF-8 -*-.
#!/bin/python
#import sys
#import os
#import math

# /*请完成下面这个函数，实现题目要求的功能*/
# /*当然，你也可以不按照下面这个模板来作答，完全按照自己的想法来 ^-^ */
#/******************************开始写代码******************************/

#def minTravelTime(N,intersections,M,roads,s,t):
#    
#    pass
#/******************************结束写代码******************************/

#_N = int(raw_input())
#
#_intersections = []
#for i in range(0,_N):
#    x,y = raw_input().split(',')
#    _intersection= [int(x),int(y)]
#    _intersections.append(_intersection)
#
#_M = int(raw_input())
#_roads= []
#for j in range(0,_M):
#    u,v,w = raw_input().split(',')
#    _road= [int(u),int(v),int(w)]
#    _roads.append(_road)
#
#_s,_t = raw_input().split(',')
#_s,_t = int(_s),int(_t)
#
#minTime = minTravelTime(_N,_intersections,_M,_roads,_s,_t)
#
#print(str(minTime)+"\n")



##输入格子数k
#k = int(input().strip())
#
##表示数组
#for i in range(k+1):
#    k = k - (i+1)
#    if k < 0:
#        L = list(range(i+2))[1:]
#        value = L[i+k]
#        break
#    if k == 0:
#        L = list(range(i+2))
#        value = L[-1]
#        break
#
#print(value)





#n, m, k = map(int, input().split(' '))
#
#multiplication_table = []
#for i in range(n):
#    for j in range(m):
#        multiplication_table.append((i+1)*(j+1))
# 
#print(sorted(multiplication_table)[k-1])


#n = int(input().strip())
#array = list(map(int, input().split(' ')))
#k = int(input().strip())
#ans = 0
#l = 0
#lengh = len(array)
#for i in range(lengh): #表示数组长度
#    for j in range(lengh): #表示数组第几个元素
#        temp = array[j:(i+1)]
#        if (sum(temp) % k) == 0:
#            l = len(temp)
#        if ans < l:
#            ans = l
#print(ans)           
            
        

#n = int(input().strip())
#s = list(map(int, input().split(' ')))
#if n == 2:
#    print("No")
#if n == 4:
#    print("Yes")

#from itertools import combinations
#print(list(combinations([1,2,3,4,5], 3)))





#
#k, m, n = list(map(int, input().split(' ')))
#value = []
#for i in range(n):
#    temp = list(map(int, input().split(' ')))
#    value += temp
#
#print(sorted(value)[k-1])



#N = int(input().strip())
#value = []
#for i in range(N):
#    value.append(list(map(int, input().split(' '))))
#
#K = int(input().strip())
#ans_value = []
#for i in range(K):
#    ans_value.append(list(map(int, input().split(' '))))
#




#N = 10
#value = [[6, 1, 2, 3, 4, 5, 6], 
#         [2, 4, 10], 
#         [3, 1, 2, 3], 
#         [4, 1, 2, 2, 3], 
#         [4, 6, 3, 3, 9], 
#         [4, 7, 8, 2, 5], 
#         [4, 5, 7, 8, 3], 
#         [4, 7, 6, 5, 9], 
#         [5, 1, 5, 8, 7, 0], 
#         [5, 10, 9, 8, 4, 3]]
#K = 3
#ans_value = [[0], 
#             [3, 1, 2, 3], 
#             [3, 10, 4, 10]]
#
#
#ans = []
#yn = 0
#for i in range(K):
#    temp = []
#    for j in range(N):
#        if (ans_value[i][0] - value[j][0] == 0) or abs((ans_value[i][0] - value[j][0] == 1)):
#            if set(ans_value[i][1:]) == set(value[j][1:]):
#                yn += 1
#                temp.append(j)
#        print(yn," ".join(list(map(str,temp))))
#        break
#    else:
#        print(yn)
#        
#        

#n = int(input().strip())     
#values = {}  
#for i in range(n):
#    temp = list(input().split(' '))
#    values[temp[0]] = temp[1]
#
#
#print(values)
#
#sorted_values = sorted(values.keys(), key=lambda item: item[1])
#for key,values in temp.items():
    
    


#N = int(input().strip())    
#ku_values = []
#for i in range(N):
#    temp = input().split(' ')
#    ku_values.append(temp)
#    
#M = int(input().strip()) 
#xw_values = []   
#for i in range(M):
#    temp = input().split(' ')
#    xw_values.append(temp)    
#
#
#for val1 in xw_values:
#    count = 0
#    for val2 in ku_values:
#        if val2[0].find(val1[0]) >= 0: 
#            count += 1
#    print(count)
#    count = 0


#n = int(input().strip())
#n = 169  
#for i in range(1,169):
#    n = n - i
#    if n <= 0:
#        print(i)
#        break
        


#n = int(input().strip()) +1
#count = 0
#for a in range(1,n):
#    for b in range(1,n):
#        for c in range(1,n):
#            for d in range(1,n):
#                if a**b == c**d:
#                    count += 1
#    
#print(count)




#n, K = list(map(int, input().split(' ')))
#array = list(map(int, input().split(' ')))

#n = 14
#K = 4
#array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 1, 1, 1]
#ans = []
#for i in range(n+1):
#    if i >= K:
#        temp = sorted(array[:i])
#        ans.append(sum(temp[-K:]))
#
#ans = list(map(str,ans))
#print(" ".join(ans))

'''今天小M要举办一个宴席，在她家有一张圆桌，桌边均匀地放置了L个座位，
座位的编号1从L到。现在她要安排她的朋友入座，她不希望桌子显得太空，
所以她希望任意的连续S个座位中，都至少有一个座位有人坐。即不存在连续的S个座位，
这些座位都是空的。注意1号座位和L号座位也是相邻的。你可以假设小M的朋友有任意多个，
你不需要将它们全部安排入座，所有的朋友都看作是相同的。
现在请你帮忙计算一下一共有多少种合法的安排方法？由于答案可能非常大，你只需要输出答案除以123456789后的余数。

第一行包含两个整数L和S。
3 2
输出对应的答案。
4

在测试样例1中，共有如下4种安排座位方法：{1，2}，{2，3}，{1，3}，{1，2，3}。
Input Sample 2­­­­
2500 2000

Output Sample 2
27511813

'''




#L, S = list(map(int, input().split(' ')))

#L, S = 3, 2



#n = int(input().strip())
#array = list(map(int, input().split(' ')))
#
#n = 3
#array = [127, 1996, 12]
#count = 0
#for x1 in array:
#    for x2 in array:
#        if x1 != x2:
#            temp = int(str(x1)+str(x2))
##            print(temp)
#            if temp % 7 == 0:
#                print(temp)
#                count += 1
#print(count)


#n = int(input().strip())
#array = list(map(int, input().split(' ')))

#n = 3
#array = [0, 1, 1]

#n = 5
#array = [1, 1, 1, 0, 0]
#
#count = 0
#for i in range(n):
#    if array[i] == 1:
#        for j in range(i,n):
#            if array[j] == 0:
#                array[j] = 1
#            if array[j] == 1:
#                array[j] = 0
#    count += 1
#    temp = sum(array)
#    print(array)
#    if (temp == 0) and (count % 2 == 0):
#        print('Alice')
#        break
#    if (temp == 0) and (count % 2 == 1):
#        print('Bob')
#        break
        
      

#n = int(raw_input().strip())
#array = list(map(int, raw_input().split(' ')))
#count = 0
#for i in range(n):
#    if array[i] == 1:
#        for j in range(i,n):
#            if array[j] == 0:
#                array[j] = 1
#            if array[j] == 1:
#                array[j] = 0
#        count += 1
#    temp = sum(array)
#    if (temp == 0) and (count % 2 == 1):
#        print('Alice')
#        break
#    if (temp == 0) and (count % 2 == 0):
#        print('Bob')
#        break


#k, n = list(map(int, input().split(' ')))
#values = []
#k, n = 3, 15
#array = range(1,10)
#for x1 in array:
#    for x2 in array:
#        for x3 in array:
#            if x1 + x2 + x3 == 15:
#                temp = sorted([x1, x2, x3])
#                if (temp not in values) and (len(set(temp)) == k):
#                    values.append(temp)
#                    temp = list(map(str,temp))
#                    print(" ".join(temp))
#                


#http://www.cnblogs.com/Finley/p/5946000.html (BP)























