import pandas as pd
from matplotlib import pyplot as plt

#bb statistics
onegunskip = 43
twogunskip = 74
spareskip = 539
eof = 557

#conventional statistics
onegunskipcc = 55
twogunskipcc = 86
spareskipcc = 551
eofcc = 569

arrayvol = '4130T'
arraydepth = [4, 5, 6, 7, 8, 9]
subsep = [8, 10]
temp = [5, 10, 15, 20, 25]
prefix = ['bb', 'cc']

filenames = []
for a in arraydepth:
	for s in subsep:
		for t in temp:
			if s < 10: 
				filenames.append(arrayvol + '__0' + str(a) + '0_2000_0' + str(s) + '0_' + str(t) + '.dan')
			else:
				filenames.append(arrayvol + '__0' + str(a) + '0_2000_' + str(s) + '0_' + str(t) + '.dan')
				

columns_one_gun = ['droparray1', 'dropgun1', 'gunvolume1', 'AvgdB', 'MaxdB', 'MaxPhase']
columns_two_gun = ['droparray1', 'dropgun1', 'droparray2', 'dropgun2', 'gunvolume1', 'gunvolume2', 'AvgdB', 'MaxdB', 'MaxPhase']
columns_spare_gun = ['droparray1', 'dropgun1', 'droparray2', 'dropgun2', 'gunvolume1', 'gunvolume2', 'AvgdB', 'MaxdB', 'MaxPhase']	
columns_one_gun_cc = ['droparray1', 'dropgun1', 'gunvolume1', 'Peak', 'Peakch','PtoB','PtoBch', 'x-corr', 'AvgdB', 'MaxdB']
columns_two_gun_cc = ['droparray1', 'dropgun1', 'droparray2', 'dropgun2', 'gunvolume1', 'gunvolume2', 'Peak', 'Peakch', 'PtoB', 'PtoBch', 'x-corr', 'AvgdB', 'MaxdB']
columns_spare_gun_cc = ['droparray1', 'dropgun1', 'droparray2', 'dropgun2', 'gunvolume1', 'gunvolume2', 'Peak', 'Peakch', 'PtoB', 'PtoBch', 'x-corr', 'AvgdB', 'MaxdB']

stat_one_gun = pd.DataFrame()
stat_two_gun = pd.DataFrame()
stat_spare_gun = pd.DataFrame()
stat_one_gun_cc = pd.DataFrame()
stat_two_gun_cc = pd.DataFrame()
stat_spare_gun_cc = pd.DataFrame()
			
for filename in filenames:
	stat_one_gun = stat_one_gun.append(pd.read_csv('bb_' + filename, names=columns_one_gun, skiprows=onegunskip, delim_whitespace=True, nrows=twogunskip-onegunskip))
	stat_two_gun = stat_two_gun.append(pd.read_csv('bb_' + filename, names=columns_two_gun, skiprows=twogunskip, delim_whitespace=True, nrows=spareskip-twogunskip))
	stat_spare_gun = stat_spare_gun.append(pd.read_csv('bb_' + filename, names=columns_spare_gun, skiprows=spareskip, delim_whitespace=True, nrows=eof-spareskip))
	stat_one_gun_cc = stat_one_gun_cc.append(pd.read_csv('cc_' + filename, names=columns_one_gun_cc, skiprows=onegunskipcc, delim_whitespace=True, nrows=twogunskipcc-onegunskipcc))
	stat_two_gun_cc = stat_two_gun_cc.append(pd.read_csv('cc_' + filename, names=columns_two_gun_cc, skiprows=twogunskipcc, delim_whitespace=True, nrows=spareskipcc-twogunskipcc))
	stat_spare_gun_cc = stat_spare_gun_cc.append(pd.read_csv('cc_' + filename, names=columns_spare_gun_cc, skiprows=spareskipcc, delim_whitespace=True, nrows=eofcc-spareskipcc))

# print('stat_one_gun: ')
# print(stat_one_gun.head())
# print('stat_two_gun: ')
# print(stat_two_gun.head())
# print(stat_two_gun.info())
# print('stat_spare_gun')
# print(stat_spare_gun.head())
# print(stat_spare_gun.info())
# print('stat_one_gun_cc: ')
# print(stat_one_gun_cc.head())
# print('stat_two_gun_cc: ')
# print(stat_two_gun_cc.head())
# print(stat_two_gun_cc.info())
# print('stat_spare_gun_cc')
# print(stat_spare_gun_cc.head())
# print(stat_spare_gun_cc.info())

stat_one_gun_all = stat_one_gun
stat_two_gun_all = stat_two_gun
stat_spare_gun_all = stat_spare_gun
stat_one_gun_cc_all = stat_one_gun_cc
stat_two_gun_cc_all = stat_two_gun_cc
stat_spare_gun_cc_all = stat_spare_gun_cc



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

#Below it is assumed that any two gun combination with illegal one gun is also illegal.

for i in range(len(illegal_one_gun)):
	stat_two_gun = stat_two_gun.drop(stat_two_gun[(stat_two_gun.droparray1 == illegal_one_gun[i][0]) & (stat_two_gun.dropgun1 == illegal_one_gun[i][1]) | (stat_two_gun.droparray2 == illegal_one_gun[i][0]) & (stat_two_gun.dropgun2 == illegal_one_gun[i][1])].index)
	stat_two_gun_cc = stat_two_gun_cc.drop(stat_two_gun_cc[(stat_two_gun_cc.droparray1 == illegal_one_gun[i][0]) & (stat_two_gun_cc.dropgun1 == illegal_one_gun[i][1]) | (stat_two_gun_cc.droparray2 == illegal_one_gun[i][0]) & (stat_two_gun_cc.dropgun2 == illegal_one_gun[i][1])].index)

# print('stat_one_gun_all: ')
# print(stat_one_gun_all.info())
# print('stat_one_gun: ')
# print(stat_one_gun.info())
# print('stat_one_gun_cc_all: ')
# print(stat_one_gun_cc_all.info())
# print('stat_one_gun_cc: ')
# print(stat_one_gun_cc.info())

# print('stat_two_gun_all: ')
# print(stat_two_gun_all.info())
# print('stat_two_gun: ')
# print(stat_two_gun.info())
# print('stat_two_gun_cc_all: ')
# print(stat_two_gun_cc_all.info())
# print('stat_two_gun_cc: ')
# print(stat_two_gun_cc.info())

def figplot(figno, dataset_bb_all, dataset_bb, dataset_cc_all, dataset_cc, figtitle):
	plt.figure(figno, figsize=(12, 10))
	plt.suptitle(figtitle, fontsize=16)
	plt.subplot(321)
	plt.hist(dataset_bb_all['MaxPhase'], bins=40, range=(0, 40), alpha=0.4, label='all')
	plt.hist(dataset_bb['MaxPhase'], bins=40, range=(0, 40), alpha=0.4, label='legal')
	plt.axvline(x=20)
	plt.legend()
	plt.title('MaxPhase')

	plt.subplot(323)
	plt.hist(dataset_bb_all['MaxdB'], bins=30, range=(0, 6), alpha=0.4, label='all')
	plt.hist(dataset_bb['MaxdB'], bins=30, range=(0, 6), alpha=0.4, label='legal')
	plt.axvline(x=3)
	plt.legend()
	plt.title('MaxdB')

	plt.subplot(325)
	plt.hist(dataset_bb_all['AvgdB'], bins=40, range=(0, 1), alpha=0.4, label='all')
	plt.hist(dataset_bb['AvgdB'], bins=40, range=(0, 1), alpha=0.4, label='legal')
	plt.axvline(x=0.85)
	plt.legend()
	plt.title('AvgdB')

	plt.subplot(322)
	plt.hist(dataset_cc_all['x-corr'], bins=40, range=(0.99, 1), alpha=0.4, label='all')
	plt.hist(dataset_cc['x-corr'], bins=40, range=(0.99, 1), alpha=0.4, label='legal')
	plt.axvline(x=0.998)
	plt.legend()
	plt.title('X-corr')

	plt.subplot(324)
	plt.hist(dataset_cc_all['Peakch'], bins=24, range=(-10, 2), alpha=0.4, label='all')
	plt.hist(dataset_cc['Peakch'], bins=24, range=(-10, 2), alpha=0.4, label='legal')
	plt.axvline(x=-10)
	plt.legend()
	plt.title('Peakch %')

	plt.subplot(326)
	plt.hist(dataset_cc_all['PtoBch'], bins=25, range=(-40, 10), alpha=0.4, label='all')
	plt.hist(dataset_cc['PtoBch'], bins=25, range=(-40, 10), alpha=0.4, label='legal')
	plt.axvline(x=-10)
	plt.legend()
	plt.title('PtoBch %')

	plt.show()

figplot(1, stat_two_gun_all, stat_two_gun, stat_two_gun_cc_all, stat_two_gun_cc, 'Two Gun dropout statistics 4130T')
figplot(2, stat_one_gun_all, stat_one_gun, stat_one_gun_cc_all, stat_one_gun_cc, 'One Gun dropout statistics 4130T')
figplot(3, stat_spare_gun_all, stat_spare_gun, stat_spare_gun_cc_all, stat_spare_gun_cc, 'Spare Gun dropout statistics 4130T')


