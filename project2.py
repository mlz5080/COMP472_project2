import sys,re,math
import pandas as pd
import string
import operator
import subprocess

def parse_file(test_file,training_file):
	test_file = "./OriginalDataSet/test-tweets-given.txt"
	training_file = "./OriginalDataSet/training-tweets.txt"

	test_string = ""
	test_list=[]
	training_string = ""
	training_list=[]
	with open(test_file, "r") as file:
		for line in file:
			test_list.append(line)

	with open(training_file, "r") as file:
		for line in file:
			training_list.append(line)

	return [training_list,test_list]

def initialize_score(df,total):
	Score={}
	for i in df.keys():
		Score[i]=abs(math.log(10,df[i]/total))
		#Score[i]=0
	#print(Score)
	return Score

def V_0():
	pass

def V_1():
	pass

def V_2():
	pass

def unigrams(V,smooth_value):
	Vocabulary_bank = {}
	Vocabulary_bank['eu'] = {}
	Vocabulary_bank['ca'] = {}
	Vocabulary_bank['gl'] = {}
	Vocabulary_bank['es'] = {}
	Vocabulary_bank['en'] = {}
	Vocabulary_bank['pt'] = {}
	result_list = parse_file("hello","world")
	training_lists = result_list[0]
	test_lists = result_list[1]
	lowercase_set = set(string.ascii_lowercase)
	letter_set = set(string.ascii_letters)
	output = ""
	if V==0:
		################# Building Vocabulary#################
		for letter in lowercase_set:
			for key in Vocabulary_bank:
				Vocabulary_bank[key][letter] = smooth_value

		################# Training for V = 0 #################		
		for tr_str in training_lists:
			tr_list = tr_str.split("\t")
			filtered_str = re.sub(r"[^A-Za-z]+", '', tr_list[3]).lower()
			#print(filtered_str)
			for letter in filtered_str:
				Vocabulary_bank[tr_list[2]][letter]+=1

		################# Result #################
		df = pd.DataFrame.from_dict(Vocabulary_bank).T
		sum_column = df.sum(axis=1)
		sum_row = df.sum(axis=0)
		#print(df)
		total_sentences = sum_column.sum()
		correct_result = 0
		total = 0
		for test in test_lists:
			if len(test.split("\t"))==4:
				output+= test.split("\t")[0]+ "  "+test.split("\t")[2]
				actual = test.split("\t")[2]
				Score = initialize_score(sum_column,total_sentences)

				str_test = test.split("\t")[3]
				filtered_str = re.sub(r"[^A-Za-z]+", '', str_test).lower()
				for letter in filtered_str:
					try:
						for key in Score:
							#print(key,df[letter][key],sum_row[letter])
							Score[key]+= abs(math.log(10,df[letter][key]/sum_row[letter]))
					except:
						pass
						#print("exception")
				output+= str(max(Score.items(), key=operator.itemgetter(1))[1]) + "  " +max(Score.items(), key=operator.itemgetter(1))[0]
				if(max(Score.items(), key=operator.itemgetter(1))[0] in actual):
					#print(actual)
					correct_result+=1
					output+="  correct"
				else:
					output+=" wrong"
				total+=1
				output+="\n"
			#print("Test result is:",max(Score),"with a score of",Score[max(Score)],"\nActual result is:",actual,"\n")
		with open("trace_0_1_"+str(smooth_value)+".txt","w") as file:
			file.write(output)
		print("The acturacy is",correct_result/total)
	elif V==1:
		################# Building Vocabulary#################
		for letter in letter_set:
			for key in Vocabulary_bank:
				Vocabulary_bank[key][letter] = smooth_value

		################# Training for V = 1 #################	
		for tr_str in training_lists:
			tr_list = tr_str.split("\t")
			filtered_str = re.sub(r"[^A-Za-z]+", '', tr_list[3])
			for letter in filtered_str:
				Vocabulary_bank[tr_list[2]][letter]+=1

		################# Result #################
		df = pd.DataFrame.from_dict(Vocabulary_bank).T
		sum_column = df.sum(axis=1)
		sum_row = df.sum(axis=0)
		#print(df)
		total_sentences = sum_column.sum()
		correct_result = 0
		total = 0
		for test in test_lists:
			if len(test.split("\t"))==4:
				output+= test.split("\t")[0]+ "  "+test.split("\t")[2]
				actual = test.split("\t")[2]
				Score = initialize_score(sum_column,total_sentences)

				str_test = test.split("\t")[3]
				filtered_str = re.sub(r"[^A-Za-z]+", '', str_test)
				for letter in filtered_str:
					try:
						for key in Score:
							#print(key,df[letter][key],sum_row[letter])
							Score[key]+= abs(math.log(10,df[letter][key]/sum_row[letter]))
					except:
						pass
						#print("exception")
				output+= str(max(Score.items(), key=operator.itemgetter(1))[1]) + "  " +max(Score.items(), key=operator.itemgetter(1))[0]
				if(max(Score.items(), key=operator.itemgetter(1))[0] in actual):
					#print(actual)
					correct_result+=1
					output+="  correct"
				else:
					output+=" wrong"
				total+=1
				output+="\n"
			#print("Test result is:",max(Score),"with a score of",Score[max(Score)],"\nActual result is:",actual,"\n")
		with open("trace_1_1_"+str(smooth_value)+".txt","w") as file:
			file.write(output)

		print("The acturacy is",correct_result/total)

	else:
		################# Building Vocabulary#################
		for letter in letter_set:
			for key in Vocabulary_bank:
				Vocabulary_bank[key][letter] = smooth_value

		################# Training for V = 2 #################	
		for tr_str in training_lists:
			tr_list = tr_str.split("\t")
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
		total_sentences = sum_column.sum()
		correct_result = 0
		total = 0
		for test in test_lists:
			if len(test.split("\t"))==4:
				output+= test.split("\t")[0]+ "  "+test.split("\t")[2]
				actual = test.split("\t")[2]
				Score = initialize_score(sum_column,total_sentences)

				str_test = test.split("\t")[3]
				for letter in str_test:
					if letter.isalpha():
						try:
							for key in Score:
								#print(key,df[letter][key],sum_row[letter])
								Score[key]+= abs(math.log(10,df[letter][key]/sum_row[letter]))
						except:
							pass
						#print("exception")
				output+= str(max(Score.items(), key=operator.itemgetter(1))[1]) + "  " +max(Score.items(), key=operator.itemgetter(1))[0]
				if(max(Score.items(), key=operator.itemgetter(1))[0] in actual):
					#print(actual)
					correct_result+=1
					output+="  correct"
				else:
					output+=" wrong"
				total+=1
				output+="\n"
			#print("Test result is:",max(Score),"with a score of",Score[max(Score)],"\nActual result is:",actual,"\n")
		with open("trace_2_1_"+str(smooth_value)+".txt","w") as file:
			file.write(output)

		print("The acturacy is",correct_result/total)

def bigrams(V,smooth_value):
	Vocabulary_bank={}
	Vocabulary_bank['eu'] = {}
	Vocabulary_bank['ca'] = {}
	Vocabulary_bank['gl'] = {}
	Vocabulary_bank['es'] = {}
	Vocabulary_bank['en'] = {}
	Vocabulary_bank['pt'] = {}
	result_list = parse_file("hello","world")
	training_lists = result_list[0]
	test_lists = result_list[1]
	lowercase_set = set(string.ascii_lowercase)
	letter_set = set(string.ascii_letters)
	isalpha_set = set(string.ascii_letters)
	output=""
	if V==0:
		################# Building Vocabulary#################
		for letter in lowercase_set:
			for letter2 in lowercase_set:
				for key in Vocabulary_bank:
					Vocabulary_bank[key][letter+letter2] = smooth_value

		################# Training for V = 0 #################		
		for tr_str in training_lists:
			tr_list = tr_str.split("\t")
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
		total_sentences = sum_column.sum()
		correct_result = 0
		total = 0
		for test in test_lists:
			if len(test.split("\t"))==4:
				output+= test.split("\t")[0]+ "  "+test.split("\t")[2]
				actual = test.split("\t")[2]
				Score = initialize_score(sum_column,total_sentences)

				str_test = test.split("\t")[3]
				for index,letter in enumerate(str_test):
					if letter.lower() in lowercase_set:
						if index<(len(str_test)-1) and str_test[index+1].lower() in lowercase_set:
							for key in Score:
								try:
								#print(key,df[letter][key],sum_row[letter])
									Score[key]+= abs(math.log(10,df[letter.lower()+str_test[index+1].lower()][key]/sum_row[letter.lower()+str_test[index+1].lower()]))
								except:
									pass
									#print("exception",letter.lower()+str_test[index+1].lower(),key)
				output+= str(max(Score.items(), key=operator.itemgetter(1))[1]) + "  " +max(Score.items(), key=operator.itemgetter(1))[0]
				if(max(Score.items(), key=operator.itemgetter(1))[0] in actual):
					#print(actual)
					correct_result+=1
					output+="  correct"
				else:
					output+=" wrong"
				total+=1
				output+="\n"
			#print("Test result is:",max(Score),"with a score of",Score[max(Score)],"\nActual result is:",actual,"\n")
		with open("trace_0_2_"+str(smooth_value)+".txt","w") as file:
			file.write(output)

		print("The acturacy is",correct_result/total)

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
			#print(filtered_str)
			for index,letter in enumerate(tr_list[3]):
				if letter.lower() in set(string.ascii_letters):
					if index<(len(tr_list[3])-1) and tr_list[3][index+1] in letter_set:
						Vocabulary_bank[tr_list[2]][letter+tr_list[3][index+1]]+=1

		################# Result #################
		df = pd.DataFrame.from_dict(Vocabulary_bank).T
		sum_column = df.sum(axis=1)
		sum_row = df.sum(axis=0)
		#print(df)

		total_sentences = sum_column.sum()
		correct_result = 0
		total = 0
		for test in test_lists:
			if len(test.split("\t"))==4:
				output+= test.split("\t")[0]+ "  "+test.split("\t")[2]
				actual = test.split("\t")[2]
				Score = initialize_score(sum_column,total_sentences)

				str_test = test.split("\t")[3]
				for index,letter in enumerate(str_test):
					if letter in letter_set:
						if index<(len(str_test)-1) and str_test[index+1] in letter_set:
							for key in Score:
								try:
								#print(key,df[letter][key],sum_row[letter])
									Score[key]+= abs(math.log(10,df[letter+str_test[index+1]][key]/sum_row[letter+str_test[index+1]]))
								except:
									pass
									#print("exception",letter+str_test[index+1],key)
				output+= str(max(Score.items(), key=operator.itemgetter(1))[1]) + "  " +max(Score.items(), key=operator.itemgetter(1))[0]
				if(max(Score.items(), key=operator.itemgetter(1))[0] in actual):
					#print(actual)
					correct_result+=1
					output+="  correct"
				else:
					output+=" wrong"
				total+=1
				output+="\n"
			#print("Test result is:",max(Score.items(), key=operator.itemgetter(1))[0],"with a score of",Score[max(Score.items(), key=operator.itemgetter(1))[0]],"\nActual result is:",actual,"\n")
		print("The acturacy is",correct_result/total)
		with open("trace_1_2_"+str(smooth_value)+".txt","w") as file:
			file.write(output)

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

		total_sentences = sum_column.sum()
		correct_result = 0
		total = 0
		for test in test_lists:
			if len(test.split("\t"))==4:
				output+= test.split("\t")[0]+ "  "+test.split("\t")[2]
				actual = test.split("\t")[2]
				Score = initialize_score(sum_column,total_sentences)
				str_test = test.split("\t")[3]
				for index,letter in enumerate(str_test):
					if letter.isalpha():
						if index<(len(str_test)-1) and (str_test[index+1].isalpha()):
							for key in Score:
								try:
								#print(key,df[letter][key],sum_row[letter])
									Score[key]+= abs(math.log(10,df[letter+str_test[index+1]][key]/sum_row[letter+str_test[index+1]]))
								except:
									pass
									#print("exception",letter+str_test[index+1],key)
				output+= str(max(Score.items(), key=operator.itemgetter(1))[1]) + "  " +max(Score.items(), key=operator.itemgetter(1))[0]
				if(max(Score.items(), key=operator.itemgetter(1))[0] in actual):
					#print(actual)
					correct_result+=1
					output+="  correct"
				else:
					output+=" wrong"
				total+=1
				output+="\n"
			#print("Test result is:",max(Score.items(), key=operator.itemgetter(1))[0],"with a score of",Score[max(Score.items(), key=operator.itemgetter(1))[0]],"\nActual result is:",actual,"\n")
		print("The acturacy is",correct_result/total)
		with open("trace_2_2_"+str(smooth_value)+".txt","w") as file:
			file.write(output)


def trigrams(V,smooth_value):
	Vocabulary_bank = {}
	Vocabulary_bank['eu'] = {}
	Vocabulary_bank['ca'] = {}
	Vocabulary_bank['gl'] = {}
	Vocabulary_bank['es'] = {}
	Vocabulary_bank['en'] = {}
	Vocabulary_bank['pt'] = {}
	result_list = parse_file("hello","world")
	training_lists = result_list[0]
	test_lists = result_list[1]
	lowercase_set = set(string.ascii_lowercase)
	letter_set = set(string.ascii_letters)
	isalpha_set = set(string.ascii_letters)
	output = ""
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
			for index,letter in enumerate(tr_list[3]):
				if letter.lower() in lowercase_set:
					if (index<(len(tr_list[3])-2) and index<(len(tr_list[3])-1)) and (tr_list[3][index+1].lower() in lowercase_set and tr_list[3][index+2].lower() in lowercase_set):
						Vocabulary_bank[tr_list[2]][letter.lower()+tr_list[3][index+1].lower()+tr_list[3][index+2].lower()]+=1

		################# Result #################
		df = pd.DataFrame.from_dict(Vocabulary_bank).T
		sum_column = df.sum(axis=1)
		sum_row = df.sum(axis=0)
		#print(df)
		total_sentences = sum_column.sum()
		correct_result = 0
		total = 0

		for test in test_lists:
			if len(test.split("\t"))==4:
				actual = test.split("\t")[2]
				Score = initialize_score(sum_column,total_sentences)
				output+= test.split("\t")[0]+ "  "+test.split("\t")[2]
				str_test = test.split("\t")[3]

				for index,letter in enumerate(str_test):
					if letter.lower() in lowercase_set:
						if (index<(len(str_test)-2) and index<(len(str_test)-1)) and (str_test[index+1].lower() in lowercase_set and str_test[index+2].lower() in lowercase_set):
							ch = letter.lower()+str_test[index+1].lower()+str_test[index+2].lower()
							for key in Score:
								try:
								#print(key,df[letter][key],sum_row[letter])
									Score[key]+= abs(math.log(10,df[ch][key]/sum_row[ch]))
								except:
									pass
									#print("exception",ch,actual)
				output+= str(max(Score.items(), key=operator.itemgetter(1))[1]) + "  " +max(Score.items(), key=operator.itemgetter(1))[0]
				if(max(Score.items(), key=operator.itemgetter(1))[0] in actual):
					#print(actual)
					correct_result+=1
					output+="  correct"
				else:
					output+=" wrong"
				total+=1
				output+="\n"
			#print("Test result is:",max(Score),"with a score of",Score[max(Score)],"\nActual result is:",actual,"\n")
		print("The acturacy is",correct_result/total)
		with open("trace_0_3_"+str(smooth_value)+".txt","w") as file:
			file.write(output)

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
			#print(filtered_str)
			################# Training for V = 0 #################		
		for tr_str in training_lists:
			tr_list = tr_str.split("\t")
			for index,letter in enumerate(tr_list[3]):
				if letter in letter_set:
					if (index<(len(tr_list[3])-2) and index<(len(tr_list[3])-1)) and (tr_list[3][index+1] in letter_set and tr_list[3][index+2] in letter_set):
						Vocabulary_bank[tr_list[2]][letter+tr_list[3][index+1]+tr_list[3][index+2]]+=1

		################# Result #################
		df = pd.DataFrame.from_dict(Vocabulary_bank).T
		sum_column = df.sum(axis=1)
		sum_row = df.sum(axis=0)
		#print(df)
		total_sentences = sum_column.sum()
		correct_result = 0
		total = 0
		for test in test_lists:
			if len(test.split("\t"))==4:
				actual = test.split("\t")[2]
				Score = initialize_score(sum_column,total_sentences)
				output+= test.split("\t")[0]+ "  "+test.split("\t")[2]
				str_test = test.split("\t")[3]

				for index,letter in enumerate(str_test):
					if letter in letter_set:
						if (index<(len(str_test)-2) and index<(len(str_test)-1)) and (str_test[index+1] in letter_set and str_test[index+2] in letter_set):
							ch = letter+str_test[index+1]+str_test[index+2]
							for key in Score:
								try:
								#print(key,df[letter][key],sum_row[letter])
									Score[key]+= abs(math.log(10,df[ch][key]/sum_row[ch]))
								except:
									pass
									#print("exception",ch,actual)
				output+= str(max(Score.items(), key=operator.itemgetter(1))[1]) + "  " +max(Score.items(), key=operator.itemgetter(1))[0]
				if(max(Score.items(), key=operator.itemgetter(1))[0] in actual):
					#print(actual)
					correct_result+=1
					output+="  correct"
				else:
					output+=" wrong"
				total+=1
				output+="\n"
			#print("Test result is:",max(Score),"with a score of",Score[max(Score)],"\nActual result is:",actual,"\n")
		print("The acturacy is",correct_result/total)
		with open("trace_1_3_"+str(smooth_value)+".txt","w") as file:
			file.write(output)
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
		total_sentences = sum_column.sum()
		correct_result = 0
		total = 0
		for test in test_lists:
			if len(test.split("\t"))==4:
				actual = test.split("\t")[2]
				Score = initialize_score(sum_column,total_sentences)
				output+= test.split("\t")[0]+ "  "+test.split("\t")[2]

				str_test = test.split("\t")[3]
				for index,letter in enumerate(str_test):
					if letter.isalpha():
						if (index<(len(str_test)-2) and index<(len(str_test)-1)) and (str_test[index+1].isalpha() and str_test[index+2].isalpha()):
							ch = letter+str_test[index+1]+str_test[index+2]
							for key in Score:
								try:
								#print(key,df[letter][key],sum_row[letter])
									Score[key]+= abs(math.log(10,df[ch][key]/sum_row[ch]))
								except:
									pass
									#print("exception",ch,actual)
				output+= str(max(Score.items(), key=operator.itemgetter(1))[1]) + "  " +max(Score.items(), key=operator.itemgetter(1))[0]
				if(max(Score.items(), key=operator.itemgetter(1))[0] in actual):
					#print(actual)
					correct_result+=1
					output+="  correct"
				else:
					output+=" wrong"
				total+=1
				output+="\n"
			#print("Test result is:",max(Score),"with a score of",Score[max(Score)],"\nActual result is:",actual,"\n")
		print("The acturacy is",correct_result/total)
		with open("trace_2_3_"+str(smooth_value)+".txt","w") as file:
			file.write(output)

if __name__ == '__main__':
	# result_list = parse_file("hello","world")
	# training_list = result_list[0]
	# test_list = result_list[1]
	if len(sys.argv)<2:
		print("Error!")
	else:
		
		smooth_value = float(sys.argv[3])
		size = int(sys.argv[2])
		V = int(sys.argv[1])
			#re.sub(r"[^a-z]+", '', mystr)
		if size == 1:
			unigrams(V,smooth_value)
		elif size == 2:
			bigrams(V,smooth_value)
			#re.sub(r"[^A-Za-z]+", '', mystr)
		else:
			trigrams(V,smooth_value)
			#mystr.isalpha()
			#break

		# for i in training_list:
		# 	print(i.split("\t"))
		# 	data_list = i.split("\t")


