import sys,re
import pandas as pd
import string

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
		print(pd.DataFrame.from_dict(Vocabulary_bank).T)

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
		print(pd.DataFrame.from_dict(Vocabulary_bank).T)

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
		print(pd.DataFrame.from_dict(Vocabulary_bank).T)

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
		print(pd.DataFrame.from_dict(Vocabulary_bank).T)

	elif V==1:
		################# Building Vocabulary#################
		for letter in set(string.ascii_letters):
			for letter2 in set(string.ascii_letters):
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
		print(pd.DataFrame.from_dict(Vocabulary_bank).T)

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

							Vocabulary_bank[tr_list[2]][letter+tr_list[3][index+1]] = smooth_value
							isalpha_set.add(tr_list[3][index+1])
							isalpha_set.add(letter)

						Vocabulary_bank[tr_list[2]][letter+tr_list[3][index+1]]+=1

		################# Result #################
		print(pd.DataFrame.from_dict(Vocabulary_bank).T)

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
		print(pd.DataFrame.from_dict(Vocabulary_bank).T)

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
		print(pd.DataFrame.from_dict(Vocabulary_bank))
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
						if letter in isalpha_set:
							if second_letter in isalpha_set
								for key in Vocabulary_bank:
									for let in isalpha_set:
										for let2 in isalpha_set
											Vocabulary_bank[key][let+let2+third_letter] = smooth_value
											Vocabulary_bank[key][third_letter+let+let2] = smooth_value
											Vocabulary_bank[key][let+third_letter+let2] = smooth_value
								isalpha_set.add(third_letter)
							elif third_letter in isalpha_set:
								for key in Vocabulary_bank:
									for let in isalpha_set:
										for let2 in isalpha_set
											Vocabulary_bank[key][let+let2+second_letter] = smooth_value
											Vocabulary_bank[key][second_letter+let+let2] = smooth_value
											Vocabulary_bank[key][let+second_letter+let2] = smooth_value
								isalpha_set.add(second_letter)
							else:
								temp = [second_letter,third_letter]
								for key in Vocabulary_bank:
									for let in isalpha_set:
										for let2 in temp:
											for let3 in temp:
												Vocabulary_bank[key][let+let2+let3] = smooth_value
												Vocabulary_bank[key][let2+let+let3] = smooth_value
												Vocabulary_bank[key][let2+let3+let] = smooth_value

						elif tr_list[3][index+1] in mystr or tr_list[3][index+1].lower() in mystr:
							for key in Vocabulary_bank:
								for let in mystr:
									Vocabulary_bank[key][letter+let] = smooth_value
									Vocabulary_bank[key][let+letter] = smooth_value
									Vocabulary_bank[key][letter+let.upper()] = smooth_value
									Vocabulary_bank[key][let.upper()+letter] = smooth_value
						else:
							for key in Vocabulary_bank:
								Vocabulary_bank[key][letter+tr_list[3][index+1]] = smooth_value
							if letter+"a" not in Vocabulary_bank[tr_list[2]]:
								for key in Vocabulary_bank:
									for let in mystr:
										Vocabulary_bank[key][letter+let] = smooth_value
										Vocabulary_bank[key][let+letter] = smooth_value
										Vocabulary_bank[key][letter+let.upper()] = smooth_value
										Vocabulary_bank[key][let.upper()+letter] = smooth_value
							elif tr_list[3][index+1]+"a" not in Vocabulary_bank[tr_list[2]]:
								for key in Vocabulary_bank:
									for let in mystr:
										#print(let)
										Vocabulary_bank[key][tr_list[3][index+1]+let] = smooth_value
										Vocabulary_bank[key][let+tr_list[3][index+1]] = smooth_value
										Vocabulary_bank[key][tr_list[3][index+1]+let.upper()] = smooth_value
										Vocabulary_bank[key][let.upper()+tr_list[3][index+1]] = smooth_value
						Vocabulary_bank[tr_list[2]][letter+tr_list[3][index+1]]+=1

		################# Result #################
		print(pd.DataFrame.from_dict(Vocabulary_bank).T)

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


