import pandas as pd
from matplotlib import pyplot as plt

onegunskip = 43
twogunskip = 74
eof = 557

onegunskip_cc = 55
twogunskip_cc = 86
eof_cc = 569

columns_one_gun = ['droparray1', 'dropgun1', 'gunvolume1', 'AvgdB', 'MaxdB', 'MaxPhase']
stat_one_gun = pd.read_csv('st_4130T__060_2000_080_25.dan', names=columns_one_gun, skiprows=onegunskip, delim_whitespace=True, nrows=twogunskip-onegunskip)

columns_two_gun = ['droparray1', 'dropgun1', 'droparray2', 'dropgun2', 'gunvolume1', 'gunvolume2', 'AvgdB', 'MaxdB', 'MaxPhase']
stat_two_gun = pd.read_csv('st_4130T__060_2000_080_25.dan', names=columns_two_gun, skiprows=twogunskip, delim_whitespace=True, nrows=eof-twogunskip)

columns_one_gun_cc = ['droparray1', 'dropgun1', 'gunvolume1', 'Peak', 'Peakch','PtoB','PtoBch', 'x-corr', 'AvgdB', 'MaxdB']
stat_one_gun_cc = pd.read_csv('cc_4130T__060_2000_080_25.dan', names=columns_one_gun_cc, skiprows=onegunskip_cc, delim_whitespace=True, nrows=twogunskip_cc-onegunskip_cc)

columns_two_gun_cc = ['droparray1', 'dropgun1', 'droparray2', 'dropgun2', 'gunvolume1', 'gunvolume2', 'Peak', 'Peakch', 'PtoB', 'PtoBch', 'x-corr', 'AvgdB', 'MaxdB']
stat_two_gun_cc = pd.read_csv('cc_4130T__060_2000_080_25.dan', names=columns_two_gun_cc, skiprows=twogunskip_cc, delim_whitespace=True, nrows=eof_cc - twogunskip_cc)

print(stat_one_gun_cc.head())
print(stat_two_gun_cc.head())
stat_one_gun_all = stat_one_gun
stat_two_gun_all = stat_two_gun
stat_one_gun_cc_all = stat_one_gun_cc
stat_two_gun_cc_all = stat_two_gun_cc


illegal_one_gun = [[1, 1], 
                   [1, 2], 
				   [1, 13],
				   [1, 14],
				   [3, 1],
				   [3, 2], 
                   [3, 11], 
				   [3, 12],
				   [3, 13],
				   [3, 14]]
for i in range(len(illegal_one_gun)):
	stat_one_gun = stat_one_gun.drop(stat_one_gun[(stat_one_gun.droparray1 == illegal_one_gun[i][0]) & (stat_one_gun.dropgun1 == illegal_one_gun[i][1])].index) 
	stat_one_gun_cc = stat_one_gun_cc.drop(stat_one_gun_cc[(stat_one_gun_cc.droparray1 == illegal_one_gun[i][0]) & (stat_one_gun_cc.dropgun1 == illegal_one_gun[i][1])].index)

#print(stat_one_gun.head())

"""illegal_two_gun = [[1, 1, 1, 2], 
				   [1, 3, 1, 4]]
				   
for i in range(len(illegal_two_gun)):
	stat_two_gun = stat_two_gun.drop(stat_two_gun[(stat_two_gun.droparray1 == illegal_two_gun[i][0]) & (stat_two_gun.dropgun1 == illegal_two_gun[i][1]) & (stat_two_gun.droparray2 == illegal_two_gun[i][2]) & (stat_two_gun.dropgun2 == illegal_two_gun[i][3])].index) 
"""
for i in range(len(illegal_one_gun)):
	stat_two_gun = stat_two_gun.drop(stat_two_gun[(stat_two_gun.droparray1 == illegal_one_gun[i][0]) & (stat_two_gun.dropgun1 == illegal_one_gun[i][1]) | (stat_two_gun.droparray2 == illegal_one_gun[i][0]) & (stat_two_gun.dropgun2 == illegal_one_gun[i][1])].index)
	stat_two_gun_cc = stat_two_gun_cc.drop(stat_two_gun_cc[(stat_two_gun_cc.droparray1 == illegal_one_gun[i][0]) & (stat_two_gun_cc.dropgun1 == illegal_one_gun[i][1]) | (stat_two_gun_cc.droparray2 == illegal_one_gun[i][0]) & (stat_two_gun_cc.dropgun2 == illegal_one_gun[i][1])].index)
print(stat_two_gun_all.head())
print(stat_two_gun.head())

plt.figure(1)
plt.subplot(321)
plt.hist(stat_two_gun_all['MaxPhase'], bins=30, range=(0, 30), alpha=0.4, label='all')
plt.hist(stat_two_gun['MaxPhase'], bins=30, range=(0, 30), alpha=0.4, label='legal')
plt.legend()
plt.title('MaxPhase')

plt.subplot(323)
plt.hist(stat_two_gun_all['MaxdB'], bins=20, range=(0, 5), alpha=0.4, label='all')
plt.hist(stat_two_gun['MaxdB'], bins=20, range=(0, 5), alpha=0.4, label='legal')
plt.legend()
plt.title('MaxdB')

plt.subplot(325)
plt.hist(stat_two_gun_all['AvgdB'], bins=20, range=(0, 1), alpha=0.4, label='all')
plt.hist(stat_two_gun['AvgdB'], bins=20, range=(0, 1), alpha=0.4, label='legal')
plt.legend()
plt.title('AvgdB')

plt.subplot(322)
plt.hist(stat_two_gun_cc_all['x-corr'], bins=20, range=(0.99, 1), alpha=0.4, label='all')
plt.hist(stat_two_gun_cc['x-corr'], bins=20, range=(0.99, 1), alpha=0.4, label='legal')
plt.legend()
plt.title('X-corr')

plt.subplot(324)
plt.hist(stat_two_gun_cc_all['Peakch'], bins=12, range=(-10, 2), alpha=0.4, label='all')
plt.hist(stat_two_gun_cc['Peakch'], bins=12, range=(-10, 2), alpha=0.4, label='legal')
plt.legend()
plt.title('Peakch')

plt.subplot(326)
plt.hist(stat_two_gun_cc_all['PtoBch'], bins=30, range=(-20, 10), alpha=0.4, label='all')
plt.hist(stat_two_gun_cc['PtoBch'], bins=30, range=(-20, 10), alpha=0.4, label='legal')
plt.legend()
plt.title('PtoBch')

plt.show()

plt.figure(2)
plt.subplot(321)
plt.hist(stat_one_gun_all['MaxPhase'], bins=30, range=(0, 30), alpha=0.4, label='all')
plt.hist(stat_one_gun['MaxPhase'], bins=30, range=(0, 30), alpha=0.4, label='legal')
plt.legend()
plt.title('MaxPhase')

plt.subplot(322)
plt.hist(stat_one_gun_all['MaxdB'], bins=20, range=(0, 5), alpha=0.4, label='all')
plt.hist(stat_one_gun['MaxdB'], bins=20, range=(0, 5), alpha=0.4, label='legal')
plt.legend()
plt.title('MaxdB')

plt.subplot(323)
plt.hist(stat_one_gun_all['AvgdB'], bins=20, range=(0, 1), alpha=0.4, label='all')
plt.hist(stat_one_gun['AvgdB'], bins=20, range=(0, 1), alpha=0.4, label='legal')
plt.legend()
plt.title('AvgdB')

plt.subplot(324)
plt.hist(stat_one_gun_cc_all['x-corr'], bins=20, range=(0.98, 1), alpha=0.4, label='all')
plt.hist(stat_one_gun_cc['x-corr'], bins=20, range=(0.98, 1), alpha=0.4, label='legal')
plt.legend()
plt.title('X-corr')

plt.subplot(325)
plt.hist(stat_one_gun_cc_all['Peakch'], bins=12, range=(-10, 2), alpha=0.4, label='all')
plt.hist(stat_one_gun_cc['Peakch'], bins=12, range=(-10, 2), alpha=0.4, label='legal')
plt.legend()
plt.title('Peakch')

plt.show()

