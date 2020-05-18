import matplotlib.pyplot as plt
import numpy as np
import os

classes1 = ('applauding', 'blowing_bubbles', 'brushing_teeth',
            'cleaning_the_floor', 'climbing', 'cooking', 'cutting_trees',
            'cutting_vegetables', 'drinking', 'feeding_a_horse',
            'fishing', 'fixing_a_bike', 'fixing_a_car', 'gardening',
            'holding_an_umbrella', 'jumping', 'looking_through_a_microscope',
            'looking_through_a_telescope', 'playing_guitar', 'playing_violin',
            'pouring_liquid', 'pushing_a_cart', 'reading', 'phoning',
            'riding_a_bike', 'riding_a_horse', 'rowing_a_boat', 'running',
            'shooting_an_arrow', 'smoking', 'taking_photos', 'texting_message',
            'throwing_frisby', 'using_a_computer', 'walking_the_dog',
            'washing_dishes', 'watching_TV', 'waving_hands', 'writing_on_a_board', 'writing_on_a_book')
classes2 = ('jumping', 'phoning', 'playinginstrument', 'reading', 'ridingbike', 'ridinghorse', 'running', 'takingphoto', 'usingcomputer', 'walking', 'others')

import xlrd
data = xlrd.open_workbook('testre.xls')
table = data.sheet_by_name('sheet1')
#print(type(table.cell_value(0,0)))
data1=[]
data2=[]
data3=[]
data4=[]
for i in range(len(classes1)):
    data1.append(100*table.cell_value(0,i))
    data2.append(100*table.cell_value(1,i))
for i in range(len(classes2)):
    data3.append(100*table.cell_value(2,i))
    data4.append(100*table.cell_value(3,i))

import pandas as pd

dd=np.array([data1,data2])
dd=dd.T
data=pd.DataFrame(dd,
                  index=classes1,
                  columns=['DenseNet','DenseNet+CBOW'],
                  )
a=data.plot(kind='bar',
          figsize=(15,10),
          title='AP of every action type in stanford 40 action dataset',fontsize=18)
plt.show()
fig = a.get_figure()
fig.savefig('stanford40.png')

ddq=np.array([data3,data4])
ddq=ddq.T
data=pd.DataFrame(ddq,
                  index=classes2,
                  columns=['DenseNet','DenseNet+CBOW'],
                  )
b=data.plot(kind='bar',
          figsize=(15,10),
          title='AP of every action type in PASCAL VOC 2012 action dataset',fontsize=18)
plt.show()
fig = b.get_figure()
fig.savefig('voc.png')