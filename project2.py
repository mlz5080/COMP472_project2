import sys,re,math,os
import pandas as pd
import string
import operator
import subprocess


def parse_file(training_file,testing_file):
	if testing_file==None:
		testing_file = "./OriginalDataSet/test-tweets-given.txt"
	if training_file==None:
		training_file = "./OriginalDataSet/training-tweets.txt"

	test_string = ""
	test_list=[]
	training_string = ""
	training_list=[]
	with open(testing_file, "r") as file:
		for line in file:
			test_list.append(line)

	with open(training_file, "r") as file:
		for line in file:
			training_list.append(line)

	return [training_list,test_list]

def initialize_score(df,total):
	Score={}
	for i in df.keys():
		#print(df[i],total)
		Score[i]=math.log10(df[i]/total)
		#print(i,df[i],total)
	#print(max(Score.items(), key=operator.itemgetter(1))[0])
	return Score

def initialize_score_byom(df,total):
	Score={}
	for i in df.keys():
		Score[i]=0
	return Score

def unigrams(V,smooth_value,debug,training,testing):
	Vocabulary_bank = {}
	key_list = ["eu","ca","gl","es","en","pt"]
	Vocabulary_bank['eu'] = {}
	Vocabulary_bank['ca'] = {}
	Vocabulary_bank['gl'] = {}
	Vocabulary_bank['es'] = {}
	Vocabulary_bank['en'] = {}
	Vocabulary_bank['pt'] = {}

	sentences_counter={}
	Precision = {}
	Recall = {}
	F1 = {}
	correct_result = 0
	total = 0
	macro_F1=0.0
	average_F1=0.0
	output = ""
	eval_model = ""

	for key in Vocabulary_bank:
		sentences_counter[key]=0
		Precision[key]=[0.0,0.0]
		Recall[key] = [0.0,0.0]
		F1[key] = 0

	if training==None and testing==None:
		result_list = parse_file(None,None)
	else:
		result_list = parse_file(training,testing)
	training_lists = result_list[0]
	test_lists = result_list[1]
	lowercase_set = set(string.ascii_lowercase)
	letter_set = set(string.ascii_letters)
	
	if V==0:

		################# Building Vocabulary#################
		for letter in lowercase_set:
			for key in Vocabulary_bank:
				Vocabulary_bank[key][letter] = smooth_value

		################# Training for V = 0 #################		
		for tr_str in training_lists:
			tr_list = tr_str.split("\t")
			filtered_str = re.sub(r"[^A-Za-z]+", '', tr_list[3]).lower()
			sentences_counter[tr_list[2]]+=1
			#print(filtered_str)
			for letter in filtered_str:
				#print(tr_list[2])
				Vocabulary_bank[tr_list[2]][letter]+=1

		################# Result #################
		df = pd.DataFrame.from_dict(Vocabulary_bank).T
		sum_column = df.sum(axis=1)
		sum_row = df.sum(axis=0)
		#print(df)
		total_sentences = len(training_lists)
		
		for test in test_lists:
			if len(test.split("\t"))==4:
				output+= test.split("\t")[0]+ "  "
				actual = test.split("\t")[2]
				Score = initialize_score(sentences_counter,total_sentences)

				str_test = test.split("\t")[3]
				filtered_str = re.sub(r"[^A-Za-z]+", '', str_test).lower()
				for letter in filtered_str:
					for key in Score:
						try:
							Score[key]+= math.log10(df[letter][key]/sum_column[key])
						except:
							if smooth_value!=0:
								Score[key]+= math.log10(smooth_value/sum_column[key])
							else:
								Score[key]+= math.log10(0.000000000000000000000000000000000000000000000000000000000001)
				#print(Score)
				################# Building evaluation #################
				test_result = max(Score.items(), key=operator.itemgetter(1))[0]
				output+= test_result + "  " + str(Score[test_result]) + "  " +actual
				Recall[actual][1]+=1
				Precision[test_result][1]+=1
				#True positive
				if(test_result in actual):
					#print(actual)
					correct_result+=1
					output+="  correct"
					Precision[test_result][0]+=1
					Recall[test_result][0]+=1
				else:
					output+="  wrong"
				total+=1
				output+="\n"

			#print("Test result is:",max(Score),"with a score of",Score[max(Score)],"\nActual result is:",actual,"\n")
		with open("trace_0_1_"+str(smooth_value)+".txt","w") as file:
			file.write(output)

		if debug:
			################# Test on Training set #####################
			total_test=0
			correct_test=0
			for test in training_lists:
				if len(test.split("\t"))==4:
					actual = test.split("\t")[2]
					Score_test = initialize_score(sentences_counter,total_sentences)
					str_test = test.split("\t")[3]
					filtered_str = re.sub(r"[^A-Za-z]+", '', str_test).lower()
					for letter in filtered_str:
						try:
							for key in Score_test:
								#print(key,df[letter][key],sum_row[letter])
								Score_test[key]+= math.log10(df[letter][key]/sum_column[key])
						except:
							pass
					test_result = max(Score_test.items(), key=operator.itemgetter(1))[0]
					#True positive
					if(test_result in actual):
						correct_test+=1				
					total_test+=1
			print("Test on training set. The acuracy is ",correct_test/total_test)


		################# Evaluation of model #################
		eval_model+="{:.4f}".format(correct_result/total)+"\n"
		for key in key_list:
			temp_result = 0
			if Precision[key][1] != 0:
				temp_result=Precision[key][0]/Precision[key][1]
			eval_model+="{:.4f}".format(temp_result)+"  "

		eval_model+="\n"
		for key in key_list:
			temp_result = 0
			if Recall[key][1]!=0:
				temp_result = Recall[key][0]/Recall[key][1]
			eval_model+="{:.4f}".format(temp_result)+"  "

		eval_model+="\n"
		
		for key in key_list:
			pre_temp=0
			re_temp=0
			if Precision[key][1] != 0:
				pre_temp = Precision[key][0]/Precision[key][1]
			if Recall[key][1]!=0:
				re_temp = Recall[key][0]/Recall[key][1]
			if pre_temp==0 and re_temp==0:
				eval_model+="{:.4f}".format(0)+"  "
			else:
				macro_F1+=2 * pre_temp*re_temp/(pre_temp+re_temp)
				average_F1+= Recall[key][1] * 2 * pre_temp*re_temp/(pre_temp+re_temp)
				eval_model+="{:.4f}".format(2 * pre_temp*re_temp/(pre_temp+re_temp))+"  "
		eval_model+="\n"+ "{:.4f}".format(macro_F1/len(key_list)) +"  " + "{:.4f}".format(average_F1/total)
		with open("eval_"+str(V)+"_1_"+str(smooth_value)+".txt","w") as file:
			file.write(eval_model)
		print("The accuracy is",correct_result/total)

	elif V==1:
		################# Building Vocabulary#################
		for letter in letter_set:
			for key in Vocabulary_bank:
				Vocabulary_bank[key][letter] = smooth_value

		################# Training for V = 1 #################	
		for tr_str in training_lists:
			tr_list = tr_str.split("\t")
			filtered_str = re.sub(r"[^A-Za-z]+", '', tr_list[3])
			sentences_counter[tr_list[2]]+=1
			for letter in filtered_str:
				Vocabulary_bank[tr_list[2]][letter]+=1

		################# Result #################
		df = pd.DataFrame.from_dict(Vocabulary_bank).T
		sum_column = df.sum(axis=1)
		sum_row = df.sum(axis=0)
		#print(df)
		total_sentences = len(training_lists)
		correct_result = 0
		total = 0
		for test in test_lists:
			if len(test.split("\t"))==4:
				output+= test.split("\t")[0]+ "  "
				actual = test.split("\t")[2]
				Score = initialize_score(sentences_counter,total_sentences)

				str_test = test.split("\t")[3]
				filtered_str = re.sub(r"[^A-Za-z]+", '', str_test)
				for letter in filtered_str:
					try:
						for key in Score:
							#print(key,df[letter][key],sum_row[letter])
							Score[key]+= math.log10(df[letter][key]/sum_column[key])
					except:
						if smooth_value!=0:
							Score[key]+= math.log10(smooth_value/sum_column[key])
						else:
							Score[key]+= math.log10(0.000000000000000000000000000000000000000000000000000000000001)
						#print("exception")
				#print(Score)
				################# Building evaluation #################
				test_result = max(Score.items(), key=operator.itemgetter(1))[0]
				output+= test_result + "  " + str(Score[test_result]) + "  " +actual
				Recall[actual][1]+=1
				Precision[test_result][1]+=1
				#True positive
				if(test_result in actual):
					#print(actual)
					correct_result+=1
					output+="  correct"
					Precision[test_result][0]+=1
					
					Recall[test_result][0]+=1
				else:
					output+="  wrong"
				total+=1
				output+="\n"
			#print("Test result is:",max(Score),"with a score of",Score[max(Score)],"\nActual result is:",actual,"\n")
		with open("trace_1_1_"+str(smooth_value)+".txt","w") as file:
			file.write(output)


		if debug:
			################# Test on Training set #####################
			total_test=0
			correct_test=0
			for test in training_lists:
				if len(test.split("\t"))==4:
					actual = test.split("\t")[2]
					Score_test = initialize_score(sentences_counter,total_sentences)
					str_test = test.split("\t")[3]
					filtered_str = re.sub(r"[^A-Za-z]+", '', str_test)
					for letter in filtered_str:
						try:
							for key in Score_test:
								#print(key,df[letter][key],sum_row[letter])
								Score_test[key]+= math.log10(df[letter][key]/sum_column[key])
						except:
							pass
							#print("exception")
							#print("exception")
					test_result = max(Score_test.items(), key=operator.itemgetter(1))[0]
					#True positive
					if(test_result in actual):
						correct_test+=1				
					total_test+=1
			print("Test on training set. The acuracy is ",correct_test/total_test)

		################# Evaluation of model #################
		eval_model+="{:.4f}".format(correct_result/total)+"\n"
		for key in key_list:
			temp_result = 0
			if Precision[key][1] != 0:
				temp_result=Precision[key][0]/Precision[key][1]
			eval_model+="{:.4f}".format(temp_result)+"  "

		eval_model+="\n"
		for key in key_list:
			temp_result = 0
			if Recall[key][1]!=0:
				temp_result = Recall[key][0]/Recall[key][1]
			eval_model+="{:.4f}".format(temp_result)+"  "

		eval_model+="\n"
		
		for key in key_list:
			pre_temp=0
			re_temp=0
			if Precision[key][1] != 0:
				pre_temp = Precision[key][0]/Precision[key][1]
			if Recall[key][1]!=0:
				re_temp = Recall[key][0]/Recall[key][1]
			if pre_temp==0 and re_temp==0:
				eval_model+="{:.4f}".format(0)+"  "
			else:
				macro_F1+=2 * pre_temp*re_temp/(pre_temp+re_temp)
				average_F1+= Recall[key][1] * 2 * pre_temp*re_temp/(pre_temp+re_temp)
				eval_model+="{:.4f}".format(2 * pre_temp*re_temp/(pre_temp+re_temp))+"  "
		eval_model+="\n"+ "{:.4f}".format(macro_F1/len(key_list)) +"  " + "{:.4f}".format(average_F1/total)
		with open("eval_"+str(V)+"_1_"+str(smooth_value)+".txt","w") as file:
			file.write(eval_model)

		print("The accuracy is",correct_result/total)

	else:
		################# Building Vocabulary#################
		for letter in letter_set:
			for key in Vocabulary_bank:
				Vocabulary_bank[key][letter] = smooth_value

		################# Training for V = 2 #################	
		for tr_str in training_lists:
			tr_list = tr_str.split("\t")
			sentences_counter[tr_list[2]]+=1
			for letter in tr_list[3]:
				if letter.isalpha():
					try:
						Vocabulary_bank[tr_list[2]][letter]+=1
					except:
						for key in Vocabulary_bank:
							Vocabulary_bank[key][letter] = smooth_value
						Vocabulary_bank[tr_list[2]][letter]+=1

		################# Result #################
		df = pd.DataFrame.from_dict(Vocabulary_bank).T
		sum_column = df.sum(axis=1)
		sum_row = df.sum(axis=0)
		#print(df)
		total_sentences = len(training_lists)
		correct_result = 0
		total = 0
		for test in test_lists:
			if len(test.split("\t"))==4:
				output+= test.split("\t")[0]+ "  "
				actual = test.split("\t")[2]
				Score = initialize_score(sentences_counter,total_sentences)

				str_test = test.split("\t")[3]
				for letter in str_test:
					if letter.isalpha():
						for key in Score:
							try:
								#print(key,df[letter][key],sum_row[letter])
								Score[key]+= math.log10(df[letter][key]/sum_column[key])
							except:
								if smooth_value!=0:
									Score[key]+= math.log10(smooth_value/sum_column[key])
								else:
									Score[key]+= math.log10(0.000000000000000000000000000000000000000000000000000000000001)
						#print("exception")

				################# Building evaluation #################
				test_result = max(Score.items(), key=operator.itemgetter(1))[0]
				output+= test_result + "  " + str(Score[test_result]) + "  " +actual
				Recall[actual][1]+=1
				Precision[test_result][1]+=1
				#True positive
				if(test_result in actual):
					#print(actual)
					correct_result+=1
					output+="  correct"
					Precision[test_result][0]+=1
					Recall[test_result][0]+=1
				else:
					output+="  wrong"
				total+=1
				output+="\n"
			#print("Test result is:",max(Score),"with a score of",Score[max(Score)],"\nActual result is:",actual,"\n")
		with open("trace_2_1_"+str(smooth_value)+".txt","w") as file:
			file.write(output)


		if debug:
			################# Test on Training set #####################
			total_test=0
			correct_test=0
			for test in training_lists:
				if len(test.split("\t"))==4:
					actual = test.split("\t")[2]
					Score_test = initialize_score(sentences_counter,total_sentences)
					str_test = test.split("\t")[3]
					for letter in str_test:
						if letter.isalpha():
							try:
								for key in Score_test:
									#print(key,df[letter][key],sum_row[letter])
									Score_test[key]+= math.log10(df[letter][key]/sum_column[key])
							except:
								pass
							#print("exception")
					test_result = max(Score_test.items(), key=operator.itemgetter(1))[0]
					#True positive
					if(test_result in actual):
						correct_test+=1				
					total_test+=1
			print("Test on training set. The acuracy is ",correct_test/total_test)

		################# Evaluation of model #################
		eval_model+="{:.4f}".format(correct_result/total)+"\n"
		for key in key_list:
			temp_result = 0
			if Precision[key][1] != 0:
				temp_result=Precision[key][0]/Precision[key][1]
			eval_model+="{:.4f}".format(temp_result)+"  "

		eval_model+="\n"
		for key in key_list:
			temp_result = 0
			if Recall[key][1]!=0:
				temp_result = Recall[key][0]/Recall[key][1]
			eval_model+="{:.4f}".format(temp_result)+"  "

		eval_model+="\n"
		
		for key in key_list:
			pre_temp=0
			re_temp=0
			if Precision[key][1] != 0:
				pre_temp = Precision[key][0]/Precision[key][1]
			if Recall[key][1]!=0:
				re_temp = Recall[key][0]/Recall[key][1]
			if pre_temp==0 and re_temp==0:
				eval_model+="{:.4f}".format(0)+"  "
			else:
				macro_F1+=2 * pre_temp*re_temp/(pre_temp+re_temp)
				average_F1+= Recall[key][1] * 2 * pre_temp*re_temp/(pre_temp+re_temp)
				eval_model+="{:.4f}".format(2 * pre_temp*re_temp/(pre_temp+re_temp))+"  "
		eval_model+="\n"+ "{:.4f}".format(macro_F1/len(key_list)) +"  " + "{:.4f}".format(average_F1/total)
		with open("eval_"+str(V)+"_1_"+str(smooth_value)+".txt","w") as file:
			file.write(eval_model)

		print("The accuracy is",correct_result/total)

def bigrams(V,smooth_value,debug,training,testing):
	Vocabulary_bank={}
	Vocabulary_bank['eu'] = {}
	Vocabulary_bank['ca'] = {}
	Vocabulary_bank['gl'] = {}
	Vocabulary_bank['es'] = {}
	Vocabulary_bank['en'] = {}
	Vocabulary_bank['pt'] = {}

	if training==None and testing==None:
		result_list = parse_file(None,None)
	else:
		result_list = parse_file(training,testing)
	training_lists = result_list[0]
	test_lists = result_list[1]
	lowercase_set = set(string.ascii_lowercase)
	letter_set = set(string.ascii_letters)
	isalpha_set = set(string.ascii_letters)
	output=""
	Precision = {}
	Recall = {}
	F1 = {}
	correct_result = 0
	total = 0
	macro_F1=0.0
	average_F1=0.0
	output = ""
	eval_model = ""
	key_list = ["eu","ca","gl","es","en","pt"]
	sentences_counter={}
	for key in Vocabulary_bank:
		sentences_counter[key]=0
		Precision[key]=[0.0,0.0]
		Recall[key] = [0.0,0.0]
		F1[key] = 0

	if V==0:
		################# Building Vocabulary#################
		for letter in lowercase_set:
			for letter2 in lowercase_set:
				for key in Vocabulary_bank:
					Vocabulary_bank[key][letter+letter2] = smooth_value

		################# Training for V = 0 #################		
		for tr_str in training_lists:
			tr_list = tr_str.split("\t")
			sentences_counter[tr_list[2]]+=1
			#print(filtered_str)
			for index,letter in enumerate(tr_list[3]):
				if letter.lower() in lowercase_set:
					if index<(len(tr_list[3])-1) and tr_list[3][index+1].lower() in lowercase_set:
						Vocabulary_bank[tr_list[2]][letter.lower()+tr_list[3][index+1].lower()]+=1

		################# Result #################
		df = pd.DataFrame.from_dict(Vocabulary_bank).T
		sum_column = df.sum(axis=1)
		sum_row = df.sum(axis=0)
		#print(df)
		total_sentences = len(training_lists)
		correct_result = 0
		total = 0
		for test in test_lists:
			if len(test.split("\t"))==4:
				output+= test.split("\t")[0]+ "  "
				actual = test.split("\t")[2]
				print(sentences_counter)
				Score = initialize_score(sentences_counter,total_sentences)

				str_test = test.split("\t")[3]
				for index,letter in enumerate(str_test):
					if letter.lower() in lowercase_set:
						if index<(len(str_test)-1) and str_test[index+1].lower() in lowercase_set:
							for key in Score:
								try:
								#print(key,df[letter][key],sum_row[letter])
									Score[key]+= math.log10(df[letter.lower()+str_test[index+1].lower()][key]/sum_column[key])
								except:
									if smooth_value!=0:
										Score[key]+= math.log10(smooth_value/sum_column[key])
									else:
										Score[key]+= math.log10(0.000000000000000000000000000000000000000000000000000000000001)
									#print("exception",letter.lower()+str_test[index+1].lower(),key)

				################# Building evaluation #################
				test_result = max(Score.items(), key=operator.itemgetter(1))[0]
				output+= test_result + "  " + str(Score[test_result]) + "  " +actual
				Recall[actual][1]+=1
				Precision[test_result][1]+=1
				#True positive
				if(test_result in actual):
					#print(actual)
					correct_result+=1
					output+="  correct"
					Precision[test_result][0]+=1
					Recall[test_result][0]+=1
				else:
					output+="  wrong"
				total+=1
				output+="\n"
			#print("Test result is:",max(Score),"with a score of",Score[max(Score)],"\nActual result is:",actual,"\n")
		with open("trace_0_2_"+str(smooth_value)+".txt","w") as file:
			file.write(output)

		if debug:
			################# Test on Training set #####################
			total_test=0
			correct_test=0
			for test in training_lists:
				if len(test.split("\t"))==4:
					actual = test.split("\t")[2]
					Score_test = initialize_score(sentences_counter,total_sentences)
					str_test = test.split("\t")[3]
					for index,letter in enumerate(str_test):
						if letter.lower() in lowercase_set:
							if index<(len(str_test)-1) and str_test[index+1].lower() in lowercase_set:
								for key in Score_test:
									try:
									#print(key,df[letter][key],sum_row[letter])
										Score_test[key]+= math.log10(df[letter.lower()+str_test[index+1].lower()][key]/sum_column[key])
									except:
										if smooth_value!=0:
											Score[key]+= math.log10(smooth_value/sum_column[key])
										else:
											Score[key]+= math.log10(0.000000000000000000000000000000000000000000000000000000000001)

					test_result = max(Score_test.items(), key=operator.itemgetter(1))[0]
					#True positive
					if(test_result in actual):
						correct_test+=1				
					total_test+=1
			print("Test on training set. The acuracy is ",correct_test/total_test)

		################# Evaluation of model #################
		eval_model+="{:.4f}".format(correct_result/total)+"\n"
		for key in key_list:
			temp_result = 0
			if Precision[key][1] != 0:
				temp_result=Precision[key][0]/Precision[key][1]
			eval_model+="{:.4f}".format(temp_result)+"  "

		eval_model+="\n"
		for key in key_list:
			temp_result = 0
			if Recall[key][1]!=0:
				temp_result = Recall[key][0]/Recall[key][1]
			eval_model+="{:.4f}".format(temp_result)+"  "

		eval_model+="\n"
		
		for key in key_list:
			pre_temp=0
			re_temp=0
			if Precision[key][1] != 0:
				pre_temp = Precision[key][0]/Precision[key][1]
			if Recall[key][1]!=0:
				re_temp = Recall[key][0]/Recall[key][1]
			if pre_temp==0 and re_temp==0:
				eval_model+="{:.4f}".format(0)+"  "
			else:
				macro_F1+=2 * pre_temp*re_temp/(pre_temp+re_temp)
				average_F1+= Recall[key][1] * 2 * pre_temp*re_temp/(pre_temp+re_temp)
				eval_model+="{:.4f}".format(2 * pre_temp*re_temp/(pre_temp+re_temp))+"  "
		eval_model+="\n"+ "{:.4f}".format(macro_F1/len(key_list)) +"  " + "{:.4f}".format(average_F1/total)
		with open("eval_"+str(V)+"_2_"+str(smooth_value)+".txt","w") as file:
			file.write(eval_model)

		print("The accuracy is",correct_result/total)

	elif V==1:
		################# Building Vocabulary#################
		for letter in letter_set:
			for letter2 in letter_set:
				for key in Vocabulary_bank:
					#set(string.ascii_letters) = [a-zA-Z]
					Vocabulary_bank[key][letter+letter2] = smooth_value


		################# Training for V = 0 #################		
		for tr_str in training_lists:
			tr_list = tr_str.split("\t")
			sentences_counter[tr_list[2]]+=1
			#print(filtered_str)
			for index,letter in enumerate(tr_list[3]):
				if letter.lower() in set(string.ascii_letters):
					if index<(len(tr_list[3])-1) and tr_list[3][index+1] in letter_set:
						Vocabulary_bank[tr_list[2]][letter+tr_list[3][index+1]]+=1

		################# Result #################
		df = pd.DataFrame.from_dict(Vocabulary_bank).T
		#df.T.to_csv("Bigram_Values.txt", sep='\t')
		sum_column = df.sum(axis=1)
		sum_row = df.sum(axis=0)
		#print(sum_column)
		#print(sentences_counter)
		#print(df)

		total_sentences = len(training_lists)
		correct_result = 0
		total = 0
		for test in test_lists:
			if len(test.split("\t"))==4:
				output+= test.split("\t")[0]+ "  "
				actual = test.split("\t")[2]
				Score = initialize_score(sentences_counter,total_sentences)

				str_test = test.split("\t")[3]
				for index,letter in enumerate(str_test):
					if letter in letter_set:
						if index<(len(str_test)-1) and str_test[index+1] in letter_set:
							#print("New round")
							for key in Score:
								try:
									#print(key,df[letter+str_test[index+1]][key],sum_column[key])
									#print(key,df[letter+str_test[index+1]][key],sum_column[key])
									Score[key] += math.log10(df[letter+str_test[index+1]][key]/sum_column[key])
									#print(key,sum_column[key])
									#print(Score)
								except:
									if smooth_value!=0:
										Score[key]+= math.log10(smooth_value/sum_column[key])
									else:
										Score[key]+= math.log10(0.000000000000000000000000000000000000000000000000000000000001)
									#print("exception",letter+str_test[index+1],key)
				################# Building evaluation #################
				test_result = max(Score.items(), key=operator.itemgetter(1))[0]
				output+= test_result + "  " + str(Score[test_result]) + "  " +actual
				Recall[actual][1]+=1
				Precision[test_result][1]+=1
				#True positive
				if(test_result in actual):
					#print(actual)
					correct_result+=1
					output+="  correct"
					Precision[test_result][0]+=1
					Recall[test_result][0]+=1
				else:
					output+="  wrong"
				total+=1
				output+="\n"
			#print("Test result is:",max(Score.items(), key=operator.itemgetter(1))[0],"with a score of",Score[max(Score.items(), key=operator.itemgetter(1))[0]],"\nActual result is:",actual,"\n")
		print("The accuracy is",correct_result/total)
		with open("trace_1_2_"+str(smooth_value)+".txt","w") as file:
			file.write(output)


		if debug:
			################# Test on Training set #####################
			total_test=0
			correct_test=0
			for test in training_lists:
				if len(test.split("\t"))==4:
					actual = test.split("\t")[2]
					Score_test = initialize_score(sentences_counter,total_sentences)
					str_test = test.split("\t")[3]
					for index,letter in enumerate(str_test):
						if letter in letter_set:
							if index<(len(str_test)-1) and str_test[index+1] in letter_set:
								for key in Score_test:
									try:
									#print(key,df[letter][key],sum_row[letter])
										Score_test[key]+= math.log10(df[letter+str_test[index+1]][key]/sum_column[key])
									except:
										if smooth_value!=0:
											Score[key]+= math.log10(smooth_value/sum_column[key])
										else:
											Score[key]+= math.log10(0.000000000000000000000000000000000000000000000000000000000001)
										#print("exception",letter+str_test[index+1],key)
					test_result = max(Score_test.items(), key=operator.itemgetter(1))[0]
					#True positive
					if(test_result in actual):
						correct_test+=1				
					total_test+=1
			print("Test on training set. The acuracy is ",correct_test/total_test)

		################# Evaluation of model #################
		eval_model+="{:.4f}".format(correct_result/total)+"\n"
		for key in key_list:
			temp_result = 0
			if Precision[key][1] != 0:
				temp_result=Precision[key][0]/Precision[key][1]
			eval_model+="{:.4f}".format(temp_result)+"  "

		eval_model+="\n"
		for key in key_list:
			temp_result = 0
			if Recall[key][1]!=0:
				temp_result = Recall[key][0]/Recall[key][1]
			eval_model+="{:.4f}".format(temp_result)+"  "

		eval_model+="\n"
		
		for key in key_list:
			pre_temp=0
			re_temp=0
			if Precision[key][1] != 0:
				pre_temp = Precision[key][0]/Precision[key][1]
			if Recall[key][1]!=0:
				re_temp = Recall[key][0]/Recall[key][1]
			if pre_temp==0 and re_temp==0:
				eval_model+="{:.4f}".format(0)+"  "
			else:
				macro_F1+=2 * pre_temp*re_temp/(pre_temp+re_temp)
				average_F1+= Recall[key][1] * 2 * pre_temp*re_temp/(pre_temp+re_temp)
				eval_model+="{:.4f}".format(2 * pre_temp*re_temp/(pre_temp+re_temp))+"  "
		eval_model+="\n"+ "{:.4f}".format(macro_F1/len(key_list)) +"  " + "{:.4f}".format(average_F1/total)
		with open("eval_"+str(V)+"_2_"+str(smooth_value)+".txt","w") as file:
			file.write(eval_model)

	else:
		################# Building Vocabulary#################
		for letter in letter_set:
			for letter2 in letter_set:
				for key in Vocabulary_bank:
					#set(string.ascii_letters) = [a-zA-Z]
					Vocabulary_bank[key][letter+letter2] = smooth_value

		################# Training for V = 2 #################	
		for tr_str in training_lists:
			tr_list = tr_str.split("\t")
			sentences_counter[tr_list[2]]+=1
			for index,letter in enumerate(tr_list[3]):
				if index<(len(tr_list[3])-1) and (letter.isalpha() and tr_list[3][index+1].isalpha()):
					try:
						Vocabulary_bank[tr_list[2]][letter+tr_list[3][index+1]]+=1

					except:
						if letter in isalpha_set:
							for key in Vocabulary_bank:
								for let in isalpha_set:
									Vocabulary_bank[key][tr_list[3][index+1]+let] = smooth_value
									Vocabulary_bank[key][let+tr_list[3][index+1]] = smooth_value
							isalpha_set.add(tr_list[3][index+1])

						elif tr_list[3][index+1] in isalpha_set:
							for key in Vocabulary_bank:
								for let in isalpha_set:
									Vocabulary_bank[key][letter+let] = smooth_value
									Vocabulary_bank[key][let+letter] = smooth_value
							isalpha_set.add(letter)

						else:
							for key in Vocabulary_bank:
								for let in isalpha_set:
									Vocabulary_bank[key][letter+let] = smooth_value
									Vocabulary_bank[key][let+letter] = smooth_value
									Vocabulary_bank[key][tr_list[3][index+1]+let] = smooth_value
									Vocabulary_bank[key][let+tr_list[3][index+1]] = smooth_value

							Vocabulary_bank[tr_list[2]][letter+letter] = smooth_value
							Vocabulary_bank[tr_list[2]][tr_list[3][index+1]+letter] = smooth_value
							Vocabulary_bank[tr_list[2]][tr_list[3][index+1]+tr_list[3][index+1]] = smooth_value
							Vocabulary_bank[tr_list[2]][letter+tr_list[3][index+1]] = smooth_value

							isalpha_set.add(tr_list[3][index+1])
							isalpha_set.add(letter)

						Vocabulary_bank[tr_list[2]][letter+tr_list[3][index+1]]+=1

		################# Result #################
		df = pd.DataFrame.from_dict(Vocabulary_bank).T
		sum_column = df.sum(axis=1)
		sum_row = df.sum(axis=0)
		#print(df)

		total_sentences = len(training_lists)
		correct_result = 0
		total = 0
		for test in test_lists:
			if len(test.split("\t"))==4:
				output+= test.split("\t")[0]+ "  "
				actual = test.split("\t")[2]
				#print(sentences_counter)
				Score = initialize_score(sentences_counter,total_sentences)
				str_test = test.split("\t")[3]
				for index,letter in enumerate(str_test):
					if letter.isalpha():
						if index<(len(str_test)-1) and (str_test[index+1].isalpha()):
							for key in Score:
								try:
								#print(key,df[letter][key],sum_row[letter])
									Score[key]+= math.log10(df[letter+str_test[index+1]][key]/sum_column[key])
								except:
									if smooth_value!=0:
										Score[key]+= math.log10(smooth_value/sum_column[key])
									else:
										Score[key]+= math.log10(0.000000000000000000000000000000000000000000000000000000000001)
									#print("exception",letter+str_test[index+1],key)
				
				################# Building evaluation #################
				test_result = max(Score.items(), key=operator.itemgetter(1))[0]
				output+= test_result + "  " + str(Score[test_result]) + "  " +actual
				Recall[actual][1]+=1
				Precision[test_result][1]+=1	
				if(test_result in actual):
					#print(actual)
					correct_result+=1
					output+="  correct"
					#True positive
					Precision[test_result][0]+=1
					Recall[test_result][0]+=1
				else:
					output+="  wrong"
				total+=1
				output+="\n"
			#print("Test result is:",max(Score.items(), key=operator.itemgetter(1))[0],"with a score of",Score[max(Score.items(), key=operator.itemgetter(1))[0]],"\nActual result is:",actual,"\n")
		print("The accuracy is",correct_result/total)
		with open("trace_2_2_"+str(smooth_value)+".txt","w") as file:
			file.write(output)

		if debug:
			################# Test on Training set #####################
			total_test=0
			correct_test=0
			for test in training_lists:
				if len(test.split("\t"))==4:
					actual = test.split("\t")[2]
					Score_test = initialize_score(sentences_counter,total_sentences)
					str_test = test.split("\t")[3]
					for index,letter in enumerate(str_test):
						if letter.isalpha():
							if index<(len(str_test)-1) and (str_test[index+1].isalpha()):
								for key in Score_test:
									try:
									#print(key,df[letter][key],sum_row[letter])
										Score_test[key]+= math.log10(df[letter+str_test[index+1]][key]/sum_column[key])
									except:
										pass
										#print("exception",letter+str_test[index+1],key)
					test_result = max(Score_test.items(), key=operator.itemgetter(1))[0]
					#True positive
					if(test_result in actual):
						correct_test+=1				
					total_test+=1
			print("Test on training set. The acuracy is ",correct_test/total_test)



		################# Evaluation of model #################
		eval_model+="{:.4f}".format(correct_result/total)+"\n"
		for key in key_list:
			temp_result = 0
			if Precision[key][1] != 0:
				temp_result=Precision[key][0]/Precision[key][1]
			eval_model+="{:.4f}".format(temp_result)+"  "

		eval_model+="\n"
		for key in key_list:
			temp_result = 0
			if Recall[key][1]!=0:
				temp_result = Recall[key][0]/Recall[key][1]
			eval_model+="{:.4f}".format(temp_result)+"  "

		eval_model+="\n"
		
		for key in key_list:
			pre_temp=0
			re_temp=0
			if Precision[key][1] != 0:
				pre_temp = Precision[key][0]/Precision[key][1]
			if Recall[key][1]!=0:
				re_temp = Recall[key][0]/Recall[key][1]
			if pre_temp==0 and re_temp==0:
				eval_model+="{:.4f}".format(0)+"  "
			else:
				macro_F1+=2 * pre_temp*re_temp/(pre_temp+re_temp)
				average_F1+= Recall[key][1] * 2 * pre_temp*re_temp/(pre_temp+re_temp)
				eval_model+="{:.4f}".format(2 * pre_temp*re_temp/(pre_temp+re_temp))+"  "
		eval_model+="\n"+ "{:.4f}".format(macro_F1/len(key_list)) +"  " + "{:.4f}".format(average_F1/total)
		with open("eval_"+str(V)+"_2_"+str(smooth_value)+".txt","w") as file:
			file.write(eval_model)

def trigrams(V,smooth_value,debug,training,testing):
	Vocabulary_bank = {}
	Vocabulary_bank['eu'] = {}
	Vocabulary_bank['ca'] = {}
	Vocabulary_bank['gl'] = {}
	Vocabulary_bank['es'] = {}
	Vocabulary_bank['en'] = {}
	Vocabulary_bank['pt'] = {}
	if training==None and testing==None:
		result_list = parse_file(None,None)
	else:
		result_list = parse_file(training,testing)

	training_lists = result_list[0]
	test_lists = result_list[1]
	lowercase_set = set(string.ascii_lowercase)
	letter_set = set(string.ascii_letters)
	isalpha_set = set(string.ascii_letters)

	output=""
	Precision = {}
	Recall = {}
	F1 = {}
	correct_result = 0
	total = 0
	macro_F1=0.0
	average_F1=0.0
	output = ""
	eval_model = ""
	key_list = ["eu","ca","gl","es","en","pt"]
	sentences_counter={}
	for key in Vocabulary_bank:
		sentences_counter[key]=0
		Precision[key]=[0.0,0.0]
		Recall[key] = [0.0,0.0]
		F1[key] = 0

	if V==0:
		################# Building Vocabulary#################
		for letter in lowercase_set:
			for letter2 in lowercase_set:
				for letter3 in lowercase_set:
					for key in Vocabulary_bank:
						Vocabulary_bank[key][letter+letter2+letter3] = smooth_value

		################# Training for V = 0 #################		
		for tr_str in training_lists:
			tr_list = tr_str.split("\t")
			sentences_counter[tr_list[2]]+=1
			for index,letter in enumerate(tr_list[3]):
				if letter.lower() in lowercase_set:
					if (index<(len(tr_list[3])-2) and index<(len(tr_list[3])-1)) and (tr_list[3][index+1].lower() in lowercase_set and tr_list[3][index+2].lower() in lowercase_set):
						#print(letter.lower()+tr_list[3][index+1].lower()+tr_list[3][index+2].lower())
						Vocabulary_bank[tr_list[2]][letter.lower()+tr_list[3][index+1].lower()+tr_list[3][index+2].lower()]+=1

		################# Result #################
		df = pd.DataFrame.from_dict(Vocabulary_bank).T
		#print(df)
		sum_column = df.sum(axis=1)
		print(sum_column)
		sum_row = df.sum(axis=0)
		print(sum_row)
		#print(df)
		total_sentences = len(training_lists)
		print(total_sentences)
		correct_result = 0
		total = 0

		for test in test_lists:
			if len(test.split("\t"))==4:
				actual = test.split("\t")[2]
				Score = initialize_score(sentences_counter,total_sentences)
				#print(Score)
				output+= test.split("\t")[0]+ "  "
				str_test = test.split("\t")[3]

				for index,letter in enumerate(str_test):
					if letter.lower() in lowercase_set:
						if (index<(len(str_test)-2) and index<(len(str_test)-1)) and (str_test[index+1].lower() in lowercase_set and str_test[index+2].lower() in lowercase_set):
							ch = letter.lower()+str_test[index+1].lower()+str_test[index+2].lower()
							for key in Score:
								try:
								#print(key,df[letter][key],sum_row[letter])
									Score[key]+= math.log10(df[ch][key]/sum_column[key])
								except Exception as inst:
									if smooth_value!=0:
										Score[key]+= math.log10(smooth_value/sum_column[key])
									else:
										Score[key]+= math.log10(0.000000000000000000000000000000000000000000000000000000000001)
									#print("exception",ch,actual)
							#print(Score)
				################# Building evaluation #################
				test_result = max(Score.items(), key=operator.itemgetter(1))[0]
				output+= test_result + "  " + str(Score[test_result]) + "  " +actual
				Recall[actual][1]+=1
				Precision[test_result][1]+=1
				#True positive
				if(test_result in actual):
					#print(actual)
					correct_result+=1
					output+="  correct"
					Precision[test_result][0]+=1
					Recall[test_result][0]+=1
				else:
					
					output+="  wrong"
				total+=1
				output+="\n"
			#print("Test result is:",max(Score),"with a score of",Score[max(Score)],"\nActual result is:",actual,"\n")
		print("The accuracy is",correct_result/total)
		with open("trace_0_3_"+str(smooth_value)+".txt","w") as file:
			file.write(output)

		total_test=0
		correct_test=0
		counter=0
		if debug:
			################# Test on Training set #####################
			for test in training_lists:
				if len(test.split("\t"))==4:
					actual = test.split("\t")[2]
					Score_test = initialize_score(sentences_counter,total_sentences)
					str_test = test.split("\t")[3]

					for index,letter in enumerate(str_test):
						if letter.lower() in lowercase_set:
							if (index<(len(str_test)-2) and index<(len(str_test)-1)) and (str_test[index+1].lower() in lowercase_set and str_test[index+2].lower() in lowercase_set):
								ch = letter.lower()+str_test[index+1].lower()+str_test[index+2].lower()
								for key in Score_test:
									try:
										#print(key)
									#print(key,df[letter][key],sum_row[letter])
										Score_test[key]+= math.log10(df[ch][key]/sum_column[key])
										counter+=1
									except:
										pass
										#print("not here")
										#print("exception",ch,actual)
					test_result = max(Score_test.items(), key=operator.itemgetter(1))[0]
					#print(test_result)
					#True positive
					if(test_result in actual):
						correct_test+=1
					total_test+=1
			print("Test on training set. The acuracy is ",correct_test/total_test)

		################# Evaluation of model #################
		eval_model+="{:.4f}".format(correct_result/total)+"\n"
		for key in key_list:
			temp_result = 0
			if Precision[key][1] != 0:
				temp_result=Precision[key][0]/Precision[key][1]
			eval_model+="{:.4f}".format(temp_result)+"  "

		eval_model+="\n"
		for key in key_list:
			temp_result = 0
			if Recall[key][1]!=0:
				temp_result = Recall[key][0]/Recall[key][1]
			eval_model+="{:.4f}".format(temp_result)+"  "

		eval_model+="\n"
		
		for key in key_list:
			pre_temp=0
			re_temp=0
			if Precision[key][1] != 0:
				pre_temp = Precision[key][0]/Precision[key][1]
			if Recall[key][1]!=0:
				re_temp = Recall[key][0]/Recall[key][1]
			if pre_temp==0 and re_temp==0:
				eval_model+="{:.4f}".format(0)+"  "
			else:
				macro_F1+=2 * pre_temp*re_temp/(pre_temp+re_temp)
				average_F1+= Recall[key][1] * 2 * pre_temp*re_temp/(pre_temp+re_temp)
				eval_model+="{:.4f}".format(2 * pre_temp*re_temp/(pre_temp+re_temp))+"  "
		eval_model+="\n"+ "{:.4f}".format(macro_F1/len(key_list)) +"  " + "{:.4f}".format(average_F1/total)
		with open("eval_"+str(V)+"_3_"+str(smooth_value)+".txt","w") as file:
			file.write(eval_model)

	elif V==1:
		################# Building Vocabulary#################
		for letter in letter_set:
			for letter2 in letter_set:
				for letter3 in letter_set:
					for key in Vocabulary_bank:
						Vocabulary_bank[key][letter+letter2+letter3] = smooth_value

		################# Training for V = 0 #################		
		for tr_str in training_lists:
			tr_list = tr_str.split("\t")
			sentences_counter[tr_list[2]]+=1
			for index,letter in enumerate(tr_list[3]):
				if letter in letter_set:
					if (index<(len(tr_list[3])-2) and index<(len(tr_list[3])-1)) and (tr_list[3][index+1] in letter_set and tr_list[3][index+2] in letter_set):
						Vocabulary_bank[tr_list[2]][letter+tr_list[3][index+1]+tr_list[3][index+2]]+=1

		################# Result #############################
		df = pd.DataFrame.from_dict(Vocabulary_bank).T
		sum_column = df.sum(axis=1)
		sum_row = df.sum(axis=0)
		#print(df)
		total_sentences = len(training_lists)
		correct_result = 0
		total = 0
		for test in test_lists:
			if len(test.split("\t"))==4:
				actual = test.split("\t")[2]
				Score = initialize_score(sentences_counter,total_sentences)
				output+= test.split("\t")[0]+ "  "
				str_test = test.split("\t")[3]

				for index,letter in enumerate(str_test):
					if letter in letter_set:
						if (index<(len(str_test)-2) and index<(len(str_test)-1)) and (str_test[index+1] in letter_set and str_test[index+2] in letter_set):
							ch = letter+str_test[index+1]+str_test[index+2]
							for key in Score:
								try:
								#print(key,df[letter][key],sum_row[letter])
									Score[key]+= math.log10(df[ch][key]/sum_column[key])
								except:
									if smooth_value!=0:
										Score[key]+= math.log10(smooth_value/sum_column[key])
									else:
										Score[key]+= math.log10(0.000000000000000000000000000000000000000000000000000000000001)
									#print("exception",ch,actual)

				################# Building evaluation #################
				test_result = max(Score.items(), key=operator.itemgetter(1))[0]
				output+= test_result + "  " + str(Score[test_result]) + "  " +actual
				Recall[actual][1]+=1
				Precision[test_result][1]+=1
				#True positive
				if(test_result in actual):
					#print(actual)
					correct_result+=1
					output+="  correct"
					Precision[test_result][0]+=1
					
					Recall[test_result][0]+=1
				else:
					output+="  wrong"
				total+=1
				output+="\n"

			#print("Test result is:",max(Score),"with a score of",Score[max(Score)],"\nActual result is:",actual,"\n")
		print("The accuracy is",correct_result/total)
		with open("trace_1_3_"+str(smooth_value)+".txt","w") as file:
			file.write(output)

		if debug:
			################# Test on Training set #####################
			total_test=0
			correct_test=0
			for test in training_lists:
				if len(test.split("\t"))==4:
					actual = test.split("\t")[2]
					Score_test = initialize_score(sentences_counter,total_sentences)
					str_test = test.split("\t")[3]

					for index,letter in enumerate(str_test):
						if letter in letter_set:
							if (index<(len(str_test)-2) and index<(len(str_test)-1)) and (str_test[index+1] in letter_set and str_test[index+2] in letter_set):
								ch = letter+str_test[index+1]+str_test[index+2]
								for key in Score_test:
									try:
									#print(key,df[letter][key],sum_row[letter])
										Score_test[key]+= math.log10(df[ch][key]/sum_column[key])
									except:
										pass
										#print("exception",ch,actual)
					test_result = max(Score_test.items(), key=operator.itemgetter(1))[0]
					#True positive
					if(test_result in actual):
						correct_test+=1
					total_test+=1
			print("Test on training set. The acuracy is ",correct_test/total_test)



		################# Evaluation of model #################
		eval_model+="{:.4f}".format(correct_result/total)+"\n"
		for key in key_list:
			temp_result = 0
			if Precision[key][1] != 0:
				temp_result=Precision[key][0]/Precision[key][1]
			eval_model+="{:.4f}".format(temp_result)+"  "

		eval_model+="\n"
		for key in key_list:
			temp_result = 0
			if Recall[key][1]!=0:
				temp_result = Recall[key][0]/Recall[key][1]
			eval_model+="{:.4f}".format(temp_result)+"  "

		eval_model+="\n"
		
		for key in key_list:
			pre_temp=0
			re_temp=0
			if Precision[key][1] != 0:
				pre_temp = Precision[key][0]/Precision[key][1]
			if Recall[key][1]!=0:
				re_temp = Recall[key][0]/Recall[key][1]
			if pre_temp==0 and re_temp==0:
				eval_model+="{:.4f}".format(0)+"  "
			else:
				macro_F1+=2 * pre_temp*re_temp/(pre_temp+re_temp)
				average_F1+= Recall[key][1] * 2 * pre_temp*re_temp/(pre_temp+re_temp)
				eval_model+="{:.4f}".format(2 * pre_temp*re_temp/(pre_temp+re_temp))+"  "
		eval_model+="\n"+ "{:.4f}".format(macro_F1/len(key_list)) +"  " + "{:.4f}".format(average_F1/total)
		with open("eval_"+str(V)+"_3_"+str(smooth_value)+".txt","w") as file:
			file.write(eval_model)

	else:
		################# Building Vocabulary#################
		for letter in letter_set:
			for letter2 in letter_set:
				for letter3 in letter_set:
					for key in Vocabulary_bank:
						Vocabulary_bank[key][letter+letter2+letter3] = smooth_value

		################# Training for V = 2 #################	
		for tr_str in training_lists:
			tr_list = tr_str.split("\t")
			sentences_counter[tr_list[2]]+=1
			for index,letter in enumerate(tr_list[3]):
				if (index<(len(tr_list[3])-1) and index<(len(tr_list[3])-2)) and (letter.isalpha() and tr_list[3][index+1].isalpha() and tr_list[3][index+2].isalpha()):
					try:
						Vocabulary_bank[tr_list[2]][letter+tr_list[3][index+1]+tr_list[3][index+2]]+=1
					except:
						second_letter = tr_list[3][index+1]
						third_letter = tr_list[3][index+2]
						#################### RULE #1 if first letter in the bank #########################
						if letter in isalpha_set:
							#print("First letter in")
						#################### RULE #1.1 if second letter in the bank ######################
							if second_letter in isalpha_set:
								for key in Vocabulary_bank:
									for let in isalpha_set:
										for let2 in isalpha_set:
											Vocabulary_bank[key][let+let2+third_letter] = smooth_value
											Vocabulary_bank[key][third_letter+let+let2] = smooth_value
											Vocabulary_bank[key][let+third_letter+let2] = smooth_value
								isalpha_set.add(third_letter)

						#################### RULE #1.2 if third letter in the bank #######################
							elif third_letter in isalpha_set:
								for key in Vocabulary_bank:
									for let in isalpha_set:
										for let2 in isalpha_set:
											Vocabulary_bank[key][let+let2+second_letter] = smooth_value
											Vocabulary_bank[key][second_letter+let+let2] = smooth_value
											Vocabulary_bank[key][let+second_letter+let2] = smooth_value
								isalpha_set.add(second_letter)

							################ RULE #1.3 if none other letter in the bank ##################
							else:
								temp = [second_letter,third_letter]
								for key in Vocabulary_bank:
									for let in isalpha_set:
										for let2 in temp:
											for let3 in temp:
												Vocabulary_bank[key][let+let2+let3] = smooth_value
												Vocabulary_bank[key][let2+let+let3] = smooth_value
												Vocabulary_bank[key][let2+let3+let] = smooth_value

						#################### RULE #2 if second letter in the bank #########################							
						elif second_letter in isalpha_set:
							#print("Second letter in")
						#################### RULE #2.1 if third letter in the bank ######################
							if third_letter in isalpha_set:
								for key in Vocabulary_bank:
									for let in isalpha_set:
										for let2 in isalpha_set:
											Vocabulary_bank[key][let+let2+letter] = smooth_value
											Vocabulary_bank[key][letter+let+let2] = smooth_value
											Vocabulary_bank[key][let+letter+let2] = smooth_value
								isalpha_set.add(letter)
							else:
								temp = [letter,third_letter]
								for key in Vocabulary_bank:
									for let in isalpha_set:
										for let2 in temp:
											for let3 in temp:
												Vocabulary_bank[key][let+let2+let3] = smooth_value
												Vocabulary_bank[key][let2+let+let3] = smooth_value
												Vocabulary_bank[key][let2+let3+let] = smooth_value
						elif third_letter in isalpha_set:
							temp = [letter,second_letter]
							for key in Vocabulary_bank:
								for let in isalpha_set:
									for let2 in temp:
										for let3 in temp:
											Vocabulary_bank[key][let+let2+let3] = smooth_value
											Vocabulary_bank[key][let2+let+let3] = smooth_value
											Vocabulary_bank[key][let2+let3+let] = smooth_value
						else:
							#print("None letter in")
							temp = [letter,second_letter,third_letter]
							for key in Vocabulary_bank:
								for let in temp:
									for let2 in temp:
										for let3 in temp:
											Vocabulary_bank[key][let+let2+let3] = smooth_value
											Vocabulary_bank[key][let2+let+let3] = smooth_value
											Vocabulary_bank[key][let2+let3+let] = smooth_value
						Vocabulary_bank[tr_list[2]][letter+tr_list[3][index+1]+tr_list[3][index+2]]+=1

		################# Result #################
		df = pd.DataFrame.from_dict(Vocabulary_bank).T
		sum_column = df.sum(axis=1)
		sum_row = df.sum(axis=0)
		#print(df)
		total_sentences = len(training_lists)
		correct_result = 0
		total = 0
		for test in test_lists:
			if len(test.split("\t"))==4:
				actual = test.split("\t")[2]
				Score = initialize_score(sentences_counter,total_sentences)
				output+= test.split("\t")[0]+ "  "

				str_test = test.split("\t")[3]
				for index,letter in enumerate(str_test):
					if letter.isalpha():
						if (index<(len(str_test)-2) and index<(len(str_test)-1)) and (str_test[index+1].isalpha() and str_test[index+2].isalpha()):
							ch = letter+str_test[index+1]+str_test[index+2]
							for key in Score:
								try:
								#print(key,df[letter][key],sum_row[letter])
									Score[key]+= math.log10(df[ch][key]/sum_column[key])
								except:
									if smooth_value!=0:
										Score[key]+= math.log10(smooth_value/sum_column[key])
									else:
										Score[key]+= math.log10(0.000000000000000000000000000000000000000000000000000000000001)
									#print("exception",ch,actual)

				################# Building evaluation #################
				test_result = max(Score.items(), key=operator.itemgetter(1))[0]
				output+= test_result + "  " + str(Score[test_result]) + "  " +actual
				Recall[actual][1]+=1
				Precision[test_result][1]+=1
				#True positive
				if(test_result in actual):
					#print(actual)
					correct_result+=1
					output+="  correct"
					Precision[test_result][0]+=1
					Recall[test_result][0]+=1
				else:
					output+="  wrong"
				total+=1
				output+="\n"

			#print("Test result is:",max(Score),"with a score of",Score[max(Score)],"\nActual result is:",actual,"\n")
		print("The accuracy is",correct_result/total)
		with open("trace_2_3_"+str(smooth_value)+".txt","w") as file:
			file.write(output)

		if debug:
			################# Test on Training set #####################
			total_test=0
			correct_test=0
			for test in training_lists:
				if len(test.split("\t"))==4:
					actual = test.split("\t")[2]
					Score_test = initialize_score(sentences_counter,total_sentences)
					str_test = test.split("\t")[3]
					for index,letter in enumerate(str_test):
						if letter.isalpha():
							if (index<(len(str_test)-2) and index<(len(str_test)-1)) and (str_test[index+1].isalpha() and str_test[index+2].isalpha()):
								ch = letter+str_test[index+1]+str_test[index+2]
								for key in Score_test:
									try:
									#print(key,df[letter][key],sum_row[letter])
										Score_test[key]+= math.log10(df[ch][key]/sum_column[key])
									except:
										pass
										#print("exception",ch,actual)
					test_result = max(Score_test.items(), key=operator.itemgetter(1))[0]
					#True positive
					if(test_result in actual):
						correct_test+=1
					total_test+=1
			print("Test on training set. The acuracy is ",correct_test/total_test)

		################# Evaluation of model #################
		eval_model+="{:.4f}".format(correct_result/total)+"\n"
		for key in key_list:
			temp_result = 0
			if Precision[key][1] != 0:
				temp_result=Precision[key][0]/Precision[key][1]
			eval_model+="{:.4f}".format(temp_result)+"  "

		eval_model+="\n"
		for key in key_list:
			temp_result = 0
			if Recall[key][1]!=0:
				temp_result = Recall[key][0]/Recall[key][1]
			eval_model+="{:.4f}".format(temp_result)+"  "

		eval_model+="\n"
		
		for key in key_list:
			pre_temp=0
			re_temp=0
			if Precision[key][1] != 0:
				pre_temp = Precision[key][0]/Precision[key][1]
			if Recall[key][1]!=0:
				re_temp = Recall[key][0]/Recall[key][1]
			if pre_temp==0 and re_temp==0:
				eval_model+="{:.4f}".format(0)+"  "
			else:
				print(Recall[key][1])
				macro_F1+=2 * pre_temp*re_temp/(pre_temp+re_temp)
				average_F1+= Recall[key][1] * 2 * pre_temp*re_temp/(pre_temp+re_temp)
				eval_model+="{:.4f}".format(2 * pre_temp*re_temp/(pre_temp+re_temp))+"  "
		print(total)
		eval_model+="\n"+ "{:.4f}".format(macro_F1/len(key_list)) +"  " + "{:.4f}".format(average_F1/total)
		with open("eval_"+str(V)+"_3_"+str(smooth_value)+".txt","w") as file:
			file.write(eval_model)

def BYOM(smooth_value,training,testing,debug):
	Vocabulary_bank = {}
	Vocabulary_bank['eu'] = {}
	Vocabulary_bank['ca'] = {}
	Vocabulary_bank['gl'] = {}
	Vocabulary_bank['es'] = {}
	Vocabulary_bank['en'] = {}
	Vocabulary_bank['pt'] = {}
	if training==None and testing==None:
		result_list = parse_file(None,None)
	else:
		result_list = parse_file(training,testing)
	training_lists = result_list[0]
	test_lists = result_list[1]
	lowercase_set = set(string.ascii_lowercase)
	letter_set = set(string.ascii_letters)
	isalpha_set = set(string.ascii_letters)

	output=""
	Precision = {}
	Recall = {}
	F1 = {}
	correct_result = 0
	total = 0
	macro_F1=0.0
	average_F1=0.0
	output = ""
	eval_model = ""
	key_list = ["eu","ca","gl","es","en","pt"]
	case1,case2,case3=False,False,False
	sentences_counter={}
	for key in Vocabulary_bank:
		sentences_counter[key]=0
		Precision[key]=[0.0,0.0]
		Recall[key] = [0.0,0.0]
		F1[key] = 0

	################# Building Vocabulary#################
	for letter in letter_set:
		for letter2 in letter_set:
			for letter3 in letter_set:
				for key in Vocabulary_bank:
					Vocabulary_bank[key][letter+letter2+letter3] = smooth_value

	for tr_str in training_lists:
		tr_list = tr_str.split("\t")
		wordlist = tr_list[3].split(" ")
		sentences_counter[tr_list[2]]+=1
		for index,word in enumerate(wordlist):

			if(determine_word(word.translate(str.maketrans('', '', string.punctuation)))):
				word=word.translate(str.maketrans('', '', string.punctuation))
				case1=True
				try:
					Vocabulary_bank[tr_list[2]][word]+=1
				except:
					for key in Vocabulary_bank:
						Vocabulary_bank[key][word]=smooth_value
					Vocabulary_bank[tr_list[2]][word]+=1
			
			if(case1 and ((index+1<len(wordlist) and determine_word(wordlist[index+1].translate(str.maketrans('', '', string.punctuation)))))):
				word2=wordlist[index+1].translate(str.maketrans('', '', string.punctuation))
				case2=True
				try:
					Vocabulary_bank[tr_list[2]][word+"_"+word2]+=1
				except:
					for key in Vocabulary_bank:
						Vocabulary_bank[key][word+"_"+word2]=smooth_value
					Vocabulary_bank[tr_list[2]][word+"_"+word2]+=1

			if(case1 and case2 and ((index+2<len(wordlist) and determine_word(wordlist[index+2].translate(str.maketrans('', '', string.punctuation)))))):
				word3=wordlist[index+2].translate(str.maketrans('', '', string.punctuation))
				try:
					Vocabulary_bank[tr_list[2]][word+"_"+word2+"_"+word3]+=1
				except:
					for key in Vocabulary_bank:
						Vocabulary_bank[key][word+"_"+word2+"_"+word3]=smooth_value
					Vocabulary_bank[tr_list[2]][word+"_"+word2+"_"+word3]+=1
			
			case1,case2,case3=False,False,False

	
	################# Training for V = 2 #################	
	for tr_str in training_lists:
		tr_list = tr_str.split("\t")
		for index,letter in enumerate(tr_list[3]):
			if (index<(len(tr_list[3])-1) and index<(len(tr_list[3])-2)) and (letter.isalpha() and tr_list[3][index+1].isalpha() and tr_list[3][index+2].isalpha()):
				try:
					Vocabulary_bank[tr_list[2]][letter+tr_list[3][index+1]+tr_list[3][index+2]]+=1
				except:
					second_letter = tr_list[3][index+1]
					third_letter = tr_list[3][index+2]
					#################### RULE #1 if first letter in the bank #########################
					if letter in isalpha_set:
						#print("First letter in")
					#################### RULE #1.1 if second letter in the bank ######################
						if second_letter in isalpha_set:
							for key in Vocabulary_bank:
								for let in isalpha_set:
									for let2 in isalpha_set:
										Vocabulary_bank[key][let+let2+third_letter] = smooth_value
										Vocabulary_bank[key][third_letter+let+let2] = smooth_value
										Vocabulary_bank[key][let+third_letter+let2] = smooth_value
							isalpha_set.add(third_letter)

					#################### RULE #1.2 if third letter in the bank #######################
						elif third_letter in isalpha_set:
							for key in Vocabulary_bank:
								for let in isalpha_set:
									for let2 in isalpha_set:
										Vocabulary_bank[key][let+let2+second_letter] = smooth_value
										Vocabulary_bank[key][second_letter+let+let2] = smooth_value
										Vocabulary_bank[key][let+second_letter+let2] = smooth_value
							isalpha_set.add(second_letter)

						################ RULE #1.3 if none other letter in the bank ##################
						else:
							temp = [second_letter,third_letter]
							for key in Vocabulary_bank:
								for let in isalpha_set:
									for let2 in temp:
										for let3 in temp:
											Vocabulary_bank[key][let+let2+let3] = smooth_value
											Vocabulary_bank[key][let2+let+let3] = smooth_value
											Vocabulary_bank[key][let2+let3+let] = smooth_value

					#################### RULE #2 if second letter in the bank #########################							
					elif second_letter in isalpha_set:
						#print("Second letter in")
					#################### RULE #2.1 if third letter in the bank ######################
						if third_letter in isalpha_set:
							for key in Vocabulary_bank:
								for let in isalpha_set:
									for let2 in isalpha_set:
										Vocabulary_bank[key][let+let2+letter] = smooth_value
										Vocabulary_bank[key][letter+let+let2] = smooth_value
										Vocabulary_bank[key][let+letter+let2] = smooth_value
							isalpha_set.add(letter)
						else:
							temp = [letter,third_letter]
							for key in Vocabulary_bank:
								for let in isalpha_set:
									for let2 in temp:
										for let3 in temp:
											Vocabulary_bank[key][let+let2+let3] = smooth_value
											Vocabulary_bank[key][let2+let+let3] = smooth_value
											Vocabulary_bank[key][let2+let3+let] = smooth_value
					elif third_letter in isalpha_set:
						temp = [letter,second_letter]
						for key in Vocabulary_bank:
							for let in isalpha_set:
								for let2 in temp:
									for let3 in temp:
										Vocabulary_bank[key][let+let2+let3] = smooth_value
										Vocabulary_bank[key][let2+let+let3] = smooth_value
										Vocabulary_bank[key][let2+let3+let] = smooth_value
					else:
						#print("None letter in")
						temp = [letter,second_letter,third_letter]
						for key in Vocabulary_bank:
							for let in temp:
								for let2 in temp:
									for let3 in temp:
										Vocabulary_bank[key][let+let2+let3] = smooth_value
										Vocabulary_bank[key][let2+let+let3] = smooth_value
										Vocabulary_bank[key][let2+let3+let] = smooth_value
					Vocabulary_bank[tr_list[2]][letter+tr_list[3][index+1]+tr_list[3][index+2]]+=1

	################# Result #################

	df = pd.DataFrame.from_dict(Vocabulary_bank).T
	#df.T.to_csv("Values.txt", sep='\t')
	sum_column = df.sum(axis=1)
	sum_row = df.sum(axis=0)
	#print(df)
	total_sentences = len(training_lists)
	correct_result = 0
	total = 0
	for test in test_lists:
		if len(test.split("\t"))==4:
			actual = test.split("\t")[2]
			Score = initialize_score_byom(sentences_counter,total_sentences)
			output+= test.split("\t")[0]+ "  "

			str_test = test.split("\t")[3]
			for index,letter in enumerate(str_test):
				if letter.isalpha():
					if (index<(len(str_test)-2) and index<(len(str_test)-1)) and (str_test[index+1].isalpha() and str_test[index+2].isalpha()):
						ch = letter+str_test[index+1]+str_test[index+2]
						for key in Score:
							try:
							#print(key,df[letter][key],sum_row[letter])
								Score[key]+= math.log10(df[ch][key]/sum_column[key])
							except:
								if smooth_value!=0:
									Score[key]+= math.log10(smooth_value/sum_column[key])
								else:
									Score[key]+= math.log10(0.000000000000000000000000000000000000000000000000000000000001)
								#print("exception",ch,actual)

			
			wordlist=str_test.split(" ")
			for index,word in enumerate(wordlist):
				if(determine_word(word) and (index+1<len(wordlist) and determine_word(wordlist[index+1])) and (index+2<len(wordlist) and determine_word(wordlist[index+2]))):
					case1=True
					ch1 = word.translate(str.maketrans('', '', string.punctuation))+"_"+wordlist[index+1].translate(str.maketrans('', '', string.punctuation))+"_"+wordlist[index+2].translate(str.maketrans('', '', string.punctuation))
				
				if(determine_word(word) and (index+1<len(wordlist) and determine_word(wordlist[index+1]))):
					case2=True
					ch2 = word.translate(str.maketrans('', '', string.punctuation))+"_"+wordlist[index+1].translate(str.maketrans('', '', string.punctuation))
				if(determine_word(word)):
					case3=True
					ch3 = word.translate(str.maketrans('', '', string.punctuation))
				if case1:
					for key in Score:
						try:
							Score[key]+= math.log10(df[ch1][key]/sum_column[key])
						except:
							if smooth_value!=0:
								Score[key]+= math.log10(smooth_value/sum_column[key])
							else:
								Score[key]+= math.log10(0.000000000000000000000000000000000000000000000000000000000001)
				if case2:
					for key in Score:
						try:
							Score[key]+= math.log10(df[ch2][key]/sum_column[key])
						except:
							if smooth_value!=0:
								Score[key]+= math.log10(smooth_value/sum_column[key])
							else:
								Score[key]+= math.log10(0.000000000000000000000000000000000000000000000000000000000001)
				if case3:
					for key in Score:
						try:
							Score[key]+= math.log10(df[ch3][key]/sum_column[key])
						except:
							if smooth_value!=0:
								Score[key]+= math.log10(smooth_value/sum_column[key])
							else:
								Score[key]+= math.log10(0.000000000000000000000000000000000000000000000000000000000001)
					
			case1,case2,case3=False,False,False

			################# Building evaluation #################
			test_result = max(Score.items(), key=operator.itemgetter(1))[0]
			output+= test_result + "  " + str(Score[test_result]) + "  " +actual
			Recall[actual][1]+=1
			Precision[test_result][1]+=1
			#True positive
			if(test_result in actual):
				#print(actual)
				correct_result+=1
				output+="  correct"
				Precision[test_result][0]+=1
				Recall[test_result][0]+=1
			else:
				output+="  wrong"
			total+=1
			output+="\n"

		#print("Test result is:",max(Score),"with a score of",Score[max(Score)],"\nActual result is:",actual,"\n")
	print("The accuracy is",correct_result/total)
	with open("Demo_trace_BYOM_"+str(smooth_value)+".txt","w") as file:
		file.write(output)

	if debug:
		df.T.to_csv("Values.txt", sep='\t')
		################# Test on Training set #####################
		total_test=0
		correct_test=0
		for test in training_lists:
			if len(test.split("\t"))==4:
				actual = test.split("\t")[2]
				Score_test = initialize_score(sentences_counter,total_sentences)
				str_test = test.split("\t")[3]
				for index,letter in enumerate(str_test):
					if letter.isalpha():
						if (index<(len(str_test)-2) and index<(len(str_test)-1)) and (str_test[index+1].isalpha() and str_test[index+2].isalpha()):
							ch = letter+str_test[index+1]+str_test[index+2]
							for key in Score_test:
								try:
								#print(key,df[letter][key],sum_row[letter])
									Score_test[key]+= math.log10(df[ch][key]/sum_column[key])
								except:
									pass
									#print("exception",ch,actual)
				test_result = max(Score_test.items(), key=operator.itemgetter(1))[0]
				#True positive
				if(test_result in actual):
					correct_test+=1
				total_test+=1
		print("Test on training set. The acuracy is ",correct_test/total_test)

	################# Evaluation of model #################
	eval_model+="{:.4f}".format(correct_result/total)+"\n"
	for key in key_list:
		temp_result = 0
		if Precision[key][1] != 0:
			temp_result=Precision[key][0]/Precision[key][1]
		eval_model+="{:.4f}".format(temp_result)+"  "

	eval_model+="\n"
	for key in key_list:
		temp_result = 0
		if Recall[key][1]!=0:
			temp_result = Recall[key][0]/Recall[key][1]
		eval_model+="{:.4f}".format(temp_result)+"  "

	eval_model+="\n"
	
	for key in key_list:
		pre_temp=0
		re_temp=0
		if Precision[key][1] != 0:
			pre_temp = Precision[key][0]/Precision[key][1]
		if Recall[key][1]!=0:
			re_temp = Recall[key][0]/Recall[key][1]
		if pre_temp==0 and re_temp==0:
			eval_model+="{:.4f}".format(0)+"  "
		else:
			macro_F1+=2 * pre_temp*re_temp/(pre_temp+re_temp)
			average_F1+= Recall[key][1] * 2 * pre_temp*re_temp/(pre_temp+re_temp)
			eval_model+="{:.4f}".format(2 * pre_temp*re_temp/(pre_temp+re_temp))+"  "
	eval_model+="\n"+ "{:.4f}".format(macro_F1/len(key_list)) +"  " + "{:.4f}".format(average_F1/total)
	with open("Demo_eval_BYOM_.txt","w") as file:
		file.write(eval_model)

def BYOM_bigram(smooth_value,training,testing,debug):

	Vocabulary_bank = {}
	Vocabulary_bank['eu'] = {}
	Vocabulary_bank['ca'] = {}
	Vocabulary_bank['gl'] = {}
	Vocabulary_bank['es'] = {}
	Vocabulary_bank['en'] = {}
	Vocabulary_bank['pt'] = {}
	if training==None and testing==None:
		result_list = parse_file(None,None)
	else:
		result_list = parse_file(training,testing)
	training_lists = result_list[0]
	test_lists = result_list[1]
	lowercase_set = set(string.ascii_lowercase)
	letter_set = set(string.ascii_letters)
	isalpha_set = set(string.ascii_letters)

	output=""
	Precision = {}
	Recall = {}
	F1 = {}
	correct_result = 0
	total = 0
	macro_F1=0.0
	average_F1=0.0
	output = ""
	eval_model = ""
	key_list = ["eu","ca","gl","es","en","pt"]
	sentences_counter={}
	case1,case2,case3=False,False,False
	for key in Vocabulary_bank:
		sentences_counter[key]=0
		Precision[key]=[0.0,0.0]
		Recall[key] = [0.0,0.0]
		F1[key] = 0

	################# Building Vocabulary#################
	for letter in letter_set:
		for letter2 in letter_set:
			for key in Vocabulary_bank:
				#set(string.ascii_letters) = [a-zA-Z]
				Vocabulary_bank[key][letter+letter2] = smooth_value

	for tr_str in training_lists:
		tr_list = tr_str.split("\t")
		wordlist = tr_list[3].split(" ")
		sentences_counter[tr_list[2]]+=1
		for index,word in enumerate(wordlist):

			if(determine_word(word.translate(str.maketrans('', '', string.punctuation)))):
				word=word.translate(str.maketrans('', '', string.punctuation))
				case1=True
				try:
					Vocabulary_bank[tr_list[2]][word]+=1
				except:
					for key in Vocabulary_bank:
						Vocabulary_bank[key][word]=smooth_value
					Vocabulary_bank[tr_list[2]][word]+=1
			
			if(case1 and ((index+1<len(wordlist) and determine_word(wordlist[index+1].translate(str.maketrans('', '', string.punctuation)))))):
				word2=wordlist[index+1].translate(str.maketrans('', '', string.punctuation))
				case2=True
				try:
					Vocabulary_bank[tr_list[2]][word+"_"+word2]+=1
				except:
					for key in Vocabulary_bank:
						Vocabulary_bank[key][word+"_"+word2]=smooth_value
					Vocabulary_bank[tr_list[2]][word+"_"+word2]+=1

			if(case1 and case2 and ((index+2<len(wordlist) and determine_word(wordlist[index+2].translate(str.maketrans('', '', string.punctuation)))))):
				word3=wordlist[index+2].translate(str.maketrans('', '', string.punctuation))
				try:
					Vocabulary_bank[tr_list[2]][word+"_"+word2+"_"+word3]+=1
				except:
					for key in Vocabulary_bank:
						Vocabulary_bank[key][word+"_"+word2+"_"+word3]=smooth_value
					Vocabulary_bank[tr_list[2]][word+"_"+word2+"_"+word3]+=1
			
			case1,case2,case3=False,False,False

		################# Training for V = 2 #################
		for index,letter in enumerate(tr_list[3]):
			if index<(len(tr_list[3])-1) and (letter.isalpha() and tr_list[3][index+1].isalpha()):
				try:
					Vocabulary_bank[tr_list[2]][letter+tr_list[3][index+1]]+=1

				except:
					if letter in isalpha_set:
						for key in Vocabulary_bank:
							for let in isalpha_set:
								Vocabulary_bank[key][tr_list[3][index+1]+let] = smooth_value
								Vocabulary_bank[key][let+tr_list[3][index+1]] = smooth_value
						isalpha_set.add(tr_list[3][index+1])

					elif tr_list[3][index+1] in isalpha_set:
						for key in Vocabulary_bank:
							for let in isalpha_set:
								Vocabulary_bank[key][letter+let] = smooth_value
								Vocabulary_bank[key][let+letter] = smooth_value
						isalpha_set.add(letter)

					else:
						for key in Vocabulary_bank:
							for let in isalpha_set:
								Vocabulary_bank[key][letter+let] = smooth_value
								Vocabulary_bank[key][let+letter] = smooth_value
								Vocabulary_bank[key][tr_list[3][index+1]+let] = smooth_value
								Vocabulary_bank[key][let+tr_list[3][index+1]] = smooth_value

						Vocabulary_bank[tr_list[2]][letter+letter] = smooth_value
						Vocabulary_bank[tr_list[2]][tr_list[3][index+1]+letter] = smooth_value
						Vocabulary_bank[tr_list[2]][tr_list[3][index+1]+tr_list[3][index+1]] = smooth_value
						Vocabulary_bank[tr_list[2]][letter+tr_list[3][index+1]] = smooth_value

						isalpha_set.add(tr_list[3][index+1])
						isalpha_set.add(letter)

					Vocabulary_bank[tr_list[2]][letter+tr_list[3][index+1]]+=1

	################# Result #################

	df = pd.DataFrame.from_dict(Vocabulary_bank).T
	
	sum_column = df.sum(axis=1)
	sum_row = df.sum(axis=0)
	#print(df)
	total_sentences = len(training_lists)
	correct_result = 0
	total = 0
	for test in test_lists:
		if len(test.split("\t"))==4:
			actual = test.split("\t")[2]
			Score = initialize_score(sentences_counter,total_sentences)
			output+= test.split("\t")[0]+ "  "

			str_test = test.split("\t")[3]
			for index,letter in enumerate(str_test):
				if index<(len(tr_list[3])-1) and (letter.isalpha() and tr_list[3][index+1].isalpha()):
					ch = letter+str_test[index+1]
					for key in Score:
						try:
						#print(key,df[letter][key],sum_row[letter])
							Score[key]+= math.log10(df[ch][key]/sum_column[key])
						except:
							if smooth_value!=0:
								Score[key]+= math.log10(smooth_value/sum_column[key])
							else:
								Score[key]+= math.log10(0.000000000000000000000000000000000000000000000000000000000001)
							#print("exception",ch,actual)

			
			wordlist=str_test.split(" ")
			for index,word in enumerate(wordlist):
				if(determine_word(word) and (index+1<len(wordlist) and determine_word(wordlist[index+1])) and (index+2<len(wordlist) and determine_word(wordlist[index+2]))):
					case1=True
					ch1 = word.translate(str.maketrans('', '', string.punctuation))+"_"+wordlist[index+1].translate(str.maketrans('', '', string.punctuation))+"_"+wordlist[index+2].translate(str.maketrans('', '', string.punctuation))
				
				if(determine_word(word) and (index+1<len(wordlist) and determine_word(wordlist[index+1]))):
					case2=True
					ch2 = word.translate(str.maketrans('', '', string.punctuation))+"_"+wordlist[index+1].translate(str.maketrans('', '', string.punctuation))
				if(determine_word(word)):
					case3=True
					ch3 = word.translate(str.maketrans('', '', string.punctuation))
				if case1:
					for key in Score:
						try:
							Score[key]+= math.log10(df[ch1][key]/sum_column[key])
						except:
							if smooth_value!=0:
								Score[key]+= math.log10(smooth_value/sum_column[key])
							else:
								Score[key]+= math.log10(0.000000000000000000000000000000000000000000000000000000000001)
				if case2:
					for key in Score:
						try:
							Score[key]+= math.log10(df[ch2][key]/sum_column[key])
						except:
							if smooth_value!=0:
								Score[key]+= math.log10(smooth_value/sum_column[key])
							else:
								Score[key]+= math.log10(0.000000000000000000000000000000000000000000000000000000000001)
				if case3:
					for key in Score:
						try:
							Score[key]+= math.log10(df[ch3][key]/sum_column[key])
						except:
							if smooth_value!=0:
								Score[key]+= math.log10(smooth_value/sum_column[key])
							else:
								Score[key]+= math.log10(0.000000000000000000000000000000000000000000000000000000000001)
					
			case1,case2,case3=False,False,False

			################# Building evaluation #################
			test_result = max(Score.items(), key=operator.itemgetter(1))[0]
			output+= test_result + "  " + str(Score[test_result]) + "  " +actual
			Recall[actual][1]+=1
			Precision[test_result][1]+=1
			#True positive
			if(test_result in actual):
				#print(actual)
				correct_result+=1
				output+="  correct"
				Precision[test_result][0]+=1
				Recall[test_result][0]+=1
			else:
				output+="  wrong"
			total+=1
			output+="\n"

		#print("Test result is:",max(Score),"with a score of",Score[max(Score)],"\nActual result is:",actual,"\n")
	print("The accuracy is",correct_result/total)
	with open("trace_BYOM_"+str(smooth_value)+".txt","w") as file:
		file.write(output)

	if debug:
		df.T.to_csv("Values.txt", sep='\t')
		################# Test on Training set #####################
		total_test=0
		correct_test=0
		for test in training_lists:
			if len(test.split("\t"))==4:
				actual = test.split("\t")[2]
				Score_test = initialize_score(sentences_counter,total_sentences)
				str_test = test.split("\t")[3]
				for index,letter in enumerate(str_test):
					if letter.isalpha():
						if (index<(len(str_test)-2) and index<(len(str_test)-1)) and (str_test[index+1].isalpha() and str_test[index+2].isalpha()):
							ch = letter+str_test[index+1]+str_test[index+2]
							for key in Score_test:
								try:
								#print(key,df[letter][key],sum_row[letter])
									Score_test[key]+= math.log10(df[ch][key]/sum_column[key])
								except:
									pass
									#print("exception",ch,actual)
				test_result = max(Score_test.items(), key=operator.itemgetter(1))[0]
				#True positive
				if(test_result in actual):
					correct_test+=1
				total_test+=1
		print("Test on training set. The acuracy is ",correct_test/total_test)

	################# Evaluation of model #################
	eval_model+="{:.4f}".format(correct_result/total)+"\n"
	for key in key_list:
		temp_result = 0
		if Precision[key][1] != 0:
			temp_result=Precision[key][0]/Precision[key][1]
		eval_model+="{:.4f}".format(temp_result)+"  "

	eval_model+="\n"
	for key in key_list:
		temp_result = 0
		if Recall[key][1]!=0:
			temp_result = Recall[key][0]/Recall[key][1]
		eval_model+="{:.4f}".format(temp_result)+"  "

	eval_model+="\n"
	
	for key in key_list:
		pre_temp=0
		re_temp=0
		if Precision[key][1] != 0:
			pre_temp = Precision[key][0]/Precision[key][1]
		if Recall[key][1]!=0:
			re_temp = Recall[key][0]/Recall[key][1]
		if pre_temp==0 and re_temp==0:
			eval_model+="{:.4f}".format(0)+"  "
		else:
			macro_F1+=2 * pre_temp*re_temp/(pre_temp+re_temp)
			average_F1+= Recall[key][1] * 2 * pre_temp*re_temp/(pre_temp+re_temp)
			eval_model+="{:.4f}".format(2 * pre_temp*re_temp/(pre_temp+re_temp))+"  "
	eval_model+="\n"+ "{:.4f}".format(macro_F1/len(key_list)) +"  " + "{:.4f}".format(average_F1/total)
	with open("eval_BYOM_.txt","w") as file:
		file.write(eval_model)

def determine_word(word):
	if len(word)==0:
		return False
	
	bool_word = True
	for char in word:
		if not char.isalpha():
			bool_word=False
	return bool_word

if __name__ == '__main__':
	if len(sys.argv)<4:
		training=sys.argv[1]
		testing = sys.argv[2]
		BYOM(0.09,training,testing,False)
		#BYOM_bigram(0.09,training,testing,False)
	elif len(sys.argv)==4:
		smooth_value = float(sys.argv[3])
		size = int(sys.argv[2])
		V = int(sys.argv[1])
		if size == 1:
			unigrams(V,smooth_value,False,None,None)
		elif size == 2:
			bigrams(V,smooth_value,False,None,None)
		else:
			trigrams(V,smooth_value,False,None,None)

	elif len(sys.argv)==6:
		smooth_value = float(sys.argv[3])
		size = int(sys.argv[2])
		V = int(sys.argv[1])
		training_file = sys.argv[4]
		testing_file = sys.argv[5]
		if size == 1:
			unigrams(V,smooth_value,False,training_file,testing_file)
		elif size == 2:
			bigrams(V,smooth_value,False,training_file,testing_file)
		else:
			trigrams(V,smooth_value,False,training_file,testing_file)

