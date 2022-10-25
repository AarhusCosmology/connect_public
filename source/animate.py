from time import sleep
import numpy as np


p1="""           __ 
    ______|  |
 __/     __  |
|       /  ) |
|      (o)   |
|__  (__/    |
   \______   |
          |__|"""
p2=""" __           
|  |______    
|         \__ 
|     / o    |
|   o - o    |
|     \ o  __|
|   ______/   
|__|          """
p0="""        
        
  ______
 (______
  ______
 (______
        
        """
k1="""         
\\ \\\\ \\   
 \\ \\\\ \\  
  \\ \\\\ \\ 
   \\ \\== 
    \\--- 
         
         """

k2="""         
  / // / 
 / // /  
/ // /   
==/ /    
---/     
         
          """

l="""
 ██████╗ ██████╗ ███╗   ██╗███╗   ██╗███████╗ ██████╗████████╗
██╔════╝██╔═══██╗████╗  ██║████╗  ██║██╔════╝██╔════╝╚══██╔══╝
██║     ██║   ██║██╔██╗ ██║██╔██╗ ██║█████╗  ██║        ██║   
██║     ██║   ██║██║╚██╗██║██║╚██╗██║██╔══╝  ██║        ██║   
╚██████╗╚██████╔╝██║ ╚████║██║ ╚████║███████╗╚██████╗   ██║   
 ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═══╝╚══════╝ ╚═════╝   ╚═╝    """


LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

N=62

pp0=p0.split('\n')
pp1=p1.split('\n')
pp2=p2.split('\n')
kk1=k1.split('\n')
kk2=k2.split('\n')
ll=l.split('\n')

def play():
    for i in range(54):
        frame=''
        for j in range(8):
            if i < 8:
                frame += ' '*(N-i) + pp0[j][:i]
                frame += '\n'
            elif i < 23:
                frame += pp1[j][22-i:] + ' '*(N-2*i+8) + pp0[j] + pp2[j][:i-8]
                frame += '\n'
            elif i < 36:
                if j in [3,5]:
                    k = '-'
                elif j == 4:
                    k = '='
                else:
                    k = ' '
                frame += (i-22)*k + pp1[j] + ' '*(N-2*i+8) + pp0[j] + pp2[j] + (i-22)*k
                frame += '\n'
            elif i < 40:
                if j in [3,5]:
                    k = '-'
                elif j == 4:
                    k = '='
                else:
                    k = ' '
                frame += (i-22)*k + pp1[j] + pp0[j][2*(i-35):] + pp2[j] + (i-22)*k
                frame += '\n'
            elif i < 48:
                if j == 0:
                    for line in ll[47-i:]:
                        frame += line + '\n'
                if j in [3,5]:
                    k = '-'
                elif j == 4:
                    k = '='
                else:
                    k = ' '
                frame += kk1[j][47-i:8] + (56-i)*k + pp1[j] + pp2[j] + (56-i)*k + kk2[j][:i-39]
                frame += '\n'
            elif i < 52:
                if j == 0:
                    for line in ll:
                        frame += line + '\n'
                if j in [3,5]:
                    k = '-'
                elif j == 4:
                    k = '='
                else:
                    k = ' '
                frame += kk1[j][:-1] + (9-(i-47))*k + pp1[j] + pp0[j][8-2*(i-47):] + pp2[j] + (9-(i-47))*k + kk2[j]
                frame += '\n'
            elif i < 54:
                if j == 0:
                    for line in ll:
                        frame += line + '\n'
                if j in [3,5]:
                    k = '-'
                elif j == 4:
                    k = '='
                else:
                    k = ' '
                frame += kk1[j][:-1] + (9-(i-47))*k + pp1[j] + ' '*2*(i-51) + pp0[j] + pp2[j] + (9-(i-47))*k + kk2[j]
                frame += '\n'

        if i > 0:        
            for _ in range(frame_count):
                print(LINE_UP, end=LINE_CLEAR)
        frame_count = frame.count('\n')+1

        sleep(0.05)
        print(frame)
