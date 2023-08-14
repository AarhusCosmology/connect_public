from time import sleep

import numpy as np


col = {'grey':'\033[40;90m',
       'white':'\033[40;97m',
       'purple':'\033[40;35m',
       'normal':'\033[0m'}

p1="""           __ 
    ______|WW|
 __/WWWWW__WW|
|WWWWWWW/WW)W|
|WWWWWW(o)WWW|
|__WW(__/WWWW|
   \______WWW|
          |__|"""
p2=""" __           
|WW|______    
|WWWWWWWWW\__ 
|WWWWW/WoWWWW|
|WWWoW-WoWWWW|
|WWWWW\WoWW__|
|WWW______/   
|__|          """
p0="""        
        
  ______
 (______
  ______
 (______
        
        """

k1="""         
\\W\\\\W\\   
 \\W\\\\W\\  
  \\W\\\\W\\ 
   \\W\\== 
    \\--- 
         
         """

k2="""         
  /W//W/ 
 /W//W/  
/W//W/   
==/W/    
---/     
         
         """

p1_idx = {0:[],
          1:[],
          2:[9,10],
          3:[8,9,10,11],
          4:[7,8,9],
          5:[5,6,7,8],
          6:[],
          7:[]}

p2_idx = {0:[],
          1:[],
          2:[],
          3:[6,7,8],
          4:[4,5,6,7,8],
          5:[6,7,8],
          6:[],
          7:[]}


l="""                                                              
 ██████╗ ██████╗ ███╗   ██╗███╗   ██╗███████╗ ██████╗████████╗
██╔════╝██╔═══██╗████╗  ██║████╗  ██║██╔════╝██╔════╝╚══██╔══╝
██║     ██║   ██║██╔██╗ ██║██╔██╗ ██║█████╗  ██║        ██║   
██║     ██║   ██║██║╚██╗██║██║╚██╗██║██╔══╝  ██║        ██║   
╚██████╗╚██████╔╝██║ ╚████║██║ ╚████║███████╗╚██████╗   ██║   
 ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═══╝╚══════╝ ╚═════╝   ╚═╝   """

stars = ''
for i in range(16):
    for j in range(63):
        if np.random.uniform() < 0.12:
            stars+='.'
        else:
            stars+=' '
    stars+='\n'


#stars="""  .      . .    .              .                      .        
#        .      .  .         .   .        ..           .        
#              .   .  .     . .                                 
#              .           .    .           .               .   
#    . . .        .          . .  .                 . .        .
#.       . .. .  .. . .       .     ..     .     .        .  .  
#.            .              .                   .             .
#.     .             ..       .                             .   
#                  ..  . .       .               .        .   . 
#      .              .      .  .             .    .            
#            .  .       . ..          .     . . .           .   
#.  .     .  .                     .                   .        
#       ..                      .       . .       ..          . 
#.                                  .   .     .         . ..    
# .         .        .                .              .          
#..           ..    ..          .            . .                """



LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

N=62

pp0=p0.split('\n')
pp1=p1.split('\n')
pp2=p2.split('\n')
kk1=k1.split('\n')
kk2=k2.split('\n')
ll=l.split('\n')
stars = stars.split('\n')

ppp0=[]
for i, pp in enumerate(pp0):
    ppp0.append('')
    for j, char in enumerate(pp):
        ppp0[-1] += col['grey']+char+col['normal']
pp0=ppp0

ppp1=[]
for i, pp in enumerate(pp1):
    ppp1.append('')
    for j, char in enumerate(pp):
        if j in p1_idx[i]:
            ppp1[-1] += col['purple']+char+col['normal']
        else:
            ppp1[-1] += col['grey']+char+col['normal']
pp1=ppp1

ppp2=[]
for i, pp in enumerate(pp2):
    ppp2.append('')
    for j, char in enumerate(pp):
        if j in p2_idx[i]:
            ppp2[-1] += col['purple']+char+col['normal']
        else:
            ppp2[-1] += col['grey']+char+col['normal']
pp2=ppp2

kkk1=[]
for i, kk in enumerate(kk1):
    kkk1.append('')
    for j, char in enumerate(kk):
        kkk1[-1] += col['grey']+char+col['normal']
kk1=kkk1

kkk2=[]
for i, kk in enumerate(kk2):
    kkk2.append('')
    for j, char in enumerate(kk):
        kkk2[-1] += col['grey']+char+col['normal']
kk2=kkk2

lll=[]
for i, lx in enumerate(ll):
    lll.append('')
    for j, char in enumerate(lx):
        lll[-1] += col['purple']+char+col['normal']
ll=lll

col_factor = 13

def play():
    for i in range(59):
        frame=''
        space = col['grey']+' '+col['normal']
        if i < 40:
            for line in ll[(47-i):]:
                frame += line + '\n'
            frame += (space*(N)+'\n')*4
        elif i < 45:
            for line in ll[(47-i):]:
                frame += line + '\n'
            frame += (space*(N)+'\n')*(44-i)
        elif i < 48:
            for line in ll[(47-i):]:
                frame += line + '\n'
        elif i < 60:
            for line in ll:
                frame += line + '\n'


        for j in range(8):
            if i < 8:
                frame += space*(N-i) + pp0[j][:i*col_factor]
                frame += '\n'
            elif i < 23:
                frame += pp1[j][(22-i)*col_factor:] + space*(N-2*i+8) + pp0[j] + pp2[j][:(i-8)*col_factor]
                frame += '\n'
            elif i < 36:
                if j in [3,5]:
                    k = col['grey']+'-'+col['normal']
                elif j == 4:
                    k = col['grey']+'='+col['normal']
                else:
                    k = col['grey']+' '+col['normal']
                frame += (i-22)*k + pp1[j] + space*(N-2*i+8) + pp0[j] + pp2[j] + (i-22)*k
                frame += '\n'
            elif i < 40:
                if j in [3,5]:
                    k = col['grey']+'-'+col['normal']
                elif j == 4:
                    k = col['grey']+'='+col['normal']
                else:
                    k = col['grey']+' '+col['normal']
                frame += (i-22)*k + pp1[j] + pp0[j][(2*(i-35))*col_factor:] + pp2[j] + (i-22)*k
                frame += '\n'
            elif i < 45:
                if j in [3,5]:
                    k = col['grey']+'-'+col['normal']
                elif j == 4:
                    k = col['grey']+'='+col['normal']
                else:
                    k = col['grey']+' '+col['normal']
                frame += 17*k + pp1[j] + pp0[j][8*col_factor:] + pp2[j] + 17*k
                frame += '\n'
            elif i < 53:
                if j in [3,5]:
                    k = col['grey']+'-'+col['normal']
                elif j == 4:
                    k = col['grey']+'='+col['normal']
                else:
                    k = col['grey']+' '+col['normal']
                frame += kk1[j][(52-i)*col_factor:8*col_factor] + (61-i)*k + pp1[j] + pp2[j] + (61-i)*k + kk2[j][:(i-44)*col_factor]
                frame += '\n'
            elif i < 56:
                if j in [3,5]:
                    k = col['grey']+'-'+col['normal']
                elif j == 4:
                    k = col['grey']+'='+col['normal']
                else:
                    k = col['grey']+' '+col['normal']
                frame += kk1[j][:-1*col_factor] + (9-(i-52))*k + pp1[j] + pp0[j][(8-2*(i-52))*col_factor:] + pp2[j] + (9-(i-52))*k + kk2[j][:-1*col_factor]
                frame += '\n'
            elif i < 60:
                if j in [3,5]:
                    k = col['grey']+'-'+col['normal']
                elif j == 4:
                    k = col['grey']+'='+col['normal']
                else:
                    k = col['grey']+' '+col['normal']
                frame += kk1[j][:-1*col_factor] + (9-(i-52))*k + pp1[j] + space*2*(i-56) + pp0[j] + pp2[j] + (9-(i-52))*k + kk2[j][:-1*col_factor]
                frame += '\n'


        if i < 44:
            frame += (space*(N)+'\n')*4
        elif i < 48:
            frame += (space*(N)+'\n')*(48-i)
        elif i < 60:
            frame += (space*(N)+'\n')



        if i > 0:        
            for _ in range(frame_count):
                print(LINE_UP, end=LINE_CLEAR)
        frame_count = frame.count('\n')+1

        sleep(0.05)
        

        frame2 = ''
        for ii, frame_line in enumerate(frame.split('\n')):
            new_frame_line = ''
            for j in range(int(len(frame_line)/col_factor)):
                if frame_line[j*col_factor+8] == ' ':
                    try:
                        new_frame_line += col['white'] + stars[ii][j] + col['normal']
                    except:
                        print(i)
                        print(frame)
                        for f in frame.split('\n'):
                            print(int(len(f)/12))
                        raise ValueError()
                elif frame_line[j*col_factor+8] == 'W':
                    new_frame_line += col['white'] + ' ' + col['normal']
                else:
                    new_frame_line += frame_line[j*col_factor:(j+1)*col_factor]
            frame2 += new_frame_line + '\n'

        frame2 = frame2[:-1]

        print(frame2)
