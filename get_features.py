#coding=utf-8
import numpy as np
import pandas as pd
import os
import glob 

def draw_data(times_list,volt_list):
	time_list_part = []
	volt_list_part = []
	for i in range(len(times_list)):
		if times_list[i] >500.0 and times_list[i] < 1500.0:
			time_list_part.append(times_list[i])
			volt_list_part.append(volt_list[i])
	return time_list_part,volt_list_part
	

def Get_Feature_discharge(dict_discharge,dots_list):


	times_list = list(dict_discharge['Time'])
	volt_list = list(dict_discharge['Voltage_measured'])
	temp_list = list(dict_discharge['Temperature_measured'])
	cap = list(dict_discharge['Capacity'])[0]
	##########################
	time_list_part,volt_list_part = draw_data(times_list,volt_list)
	fix_ = np.polyfit(time_list_part,volt_list_part,4)
	fix_equation = np.poly1d(fix_)
	volt_predict_corr = fix_equation(time_list_part)
	volt_predict = fix_equation(dots_list)
	############################
	
	volt_min = min(volt_list)
	volt_index_min = volt_list.index(volt_min)
	a = (volt_list[0]-volt_min)
	b = float(times_list[volt_index_min]-times_list[0])/float(3600)
	teq_vd = float(b)/float(a)
	veq_td = float(a)/float(b)
	
	
	
	#corr = Corr_coeff(volt_list_part,volt_predict_corr)
	MVF = sum(4.2-np.array(volt_predict))/float(len(volt_predict))
	Tst = temp_list[0]
	Tavg = np.mean(temp_list)
	Vavg = np.mean(volt_list)
	
	
	
	return cap,MVF,Tst,Tavg,Vavg,teq_vd,veq_td
	
def Get_Feature_charge(df_charge):
	times_list = list(df_charge['Time'])
	volt_list = list(df_charge['Voltage_measured'])
	current_list = list(df_charge['Current_measured'])
	
	for ii in range(len(volt_list)):
		if volt_list[ii] >=4.2:
			key_index = ii
			break
	for jj in range(len(volt_list)-1,-1,-1):
		if current_list[jj] >=0.02:
			key_index_down = jj
			break
	thv = float(times_list[key_index_down]-times_list[key_index])/float(3600)
	
	time_up_volt = float(times_list[key_index]-times_list[0])/float(3600)
	up_volt = volt_list[key_index]-volt_list[0]
	#print(volt_list[key_index])
	#print(up_volt)
	teq_vc = time_up_volt/up_volt
	veq_vc = up_volt/time_up_volt

	c = current_list[key_index]-current_list[key_index_down]
	d = float(times_list[key_index_down]-times_list[key_index])/float(3600)
	leq_tc = c/d
	
	return thv,teq_vc,veq_vc,leq_tc
	


os.chdir(os.getcwd())
lines = '{}	'*12+'\n'
feature_txt = open(str(os.getcwd().split('\\')[-1])+'_feature.txt','w+')
feature_txt.write(lines.format('cycle','cap','MVF','corr','Tst','Tavg','Vavg','teq_vd','veq_td','thv','teq_vc','veq_vc','leq_tc'))
xlsd_files_discharge = glob.glob('discharge_*.xls')
xlsd_files_discharge = sorted(xlsd_files_discharge,key = lambda x:int(x.split('_')[1][:-4]))

xlsd_files_charge = glob.glob('charge_*.xls')
xlsd_files_charge = sorted(xlsd_files_charge,key = lambda x:int(x.split('_')[1][:-4]))

xlsd_files_charge.pop(32)
xlsd_files_charge.pop(-1)
#print(xlsd_files_charge)

dots_list = [i for i in range(500,1500,10)]
dict_feature = {}



to_df = [[] for _ in range(11)]

for discharge_file,charge_file in zip(xlsd_files_discharge,xlsd_files_charge):
	df_discharge = pd.read_excel(discharge_file)
	df_charge = pd.read_excel(charge_file)
	cap,MVF,Tst,Tavg,Vavg,teq_vd,veq_td = Get_Feature_discharge(df_discharge,dots_list)
	thv,teq_vc,veq_vc,leq_tc = Get_Feature_charge(df_charge)
	feature_list = [cap,MVF,Tst,Tavg,Vavg,teq_vd,veq_td,thv,teq_vc,veq_vc,leq_tc]
	
	for n,to_df_cell in enumerate(to_df):
		to_df_cell.append(feature_list[n])
	
	#feature_txt.write(str(discharge_file.split('_')[1][:-4])+'	'+str(cap)+'	'+str(MVF)+'	'+str(corr)+'	'+str(Tst)+'	'+str(Tavg)+
	#'	'+str(Vavg)+'	'+str(teq_vd)+'	'+str(veq_td)+'	'+str(thv)+'\n')
	
	#feature_txt.write(lines.format(str(discharge_file.split('_')[1][:-4]),str(cap),str(MVF),str(corr),str(Tst),str(Tavg),str(Vavg),str(teq_vd),str(veq_td),str(thv)
	#,str(teq_vc),str(veq_vc),str(leq_tc)))
	feature_txt.write(lines.format(str(discharge_file.split('_')[1][:-4]),*feature_list))
#print(np.array(feature_list))
df_feature = pd.DataFrame(np.array(to_df).T,columns=['capacity','MVF','Tst','Tavg','Vavg','teq_vd','veq_td','thv','teq_vc','veq_vc','leq_tc'])
df_feature.to_excel('dataset.xls')	
feature_txt.close()
#
#
#
#
#
#
#

#mvf = Get_MVF(df,dots_list)
#print(mvf)
