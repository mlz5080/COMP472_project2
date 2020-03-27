import sys,re
import pandas as pd

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
	mystr = "abcdefghijklmnopqrstuvwxyz"
	if V==0:
		################# Building Vocabulary#################
		for letter in mystr:
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
		print(pd.DataFrame.from_dict(Vocabulary_bank))

	elif V==1:
		################# Building Vocabulary#################
		for letter in mystr:
			for key in Vocabulary_bank:
				Vocabulary_bank[key][letter] = smooth_value
		for letter in mystr.upper():
			for key in Vocabulary_bank:
				Vocabulary_bank[key][letter] = smooth_value

		################# Training for V = 1 #################	
		for tr_str in training_lists:
			tr_list = tr_str.split("\t")
			filtered_str = re.sub(r"[^A-Za-z]+", '', tr_list[3])
			for letter in filtered_str:
				Vocabulary_bank[tr_list[2]][letter]+=1

		################# Result #################
		print(pd.DataFrame.from_dict(Vocabulary_bank))

	else:
		################# Building Vocabulary#################
		for letter in mystr:
			for key in Vocabulary_bank:
				Vocabulary_bank[key][letter] = smooth_value
		for letter in mystr.upper():
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
		print(pd.DataFrame.from_dict(Vocabulary_bank))

def bigrams(V,smooth_value):
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
	mystr = "abcdefghijklmnopqrstuvwxyz"
	if V==0:
		################# Building Vocabulary#################
		for letter in mystr:
			for letter2 in mystr:
				for key in Vocabulary_bank:
					Vocabulary_bank[key][letter+letter2] = smooth_value

		################# Training for V = 0 #################		
		for tr_str in training_lists:
			tr_list = tr_str.split("\t")
			#print(filtered_str)
			for index,letter in enumerate(tr_list[3]):
				if letter.lower() in mystr:
					if index<(len(tr_list[3])-1) and tr_list[3][index+1].lower() in mystr:
						print(letter+tr_list[3][index+1].lower())
						Vocabulary_bank[tr_list[2]][letter.lower()+tr_list[3][index+1].lower()]+=1

		################# Result #################
		print(pd.DataFrame.from_dict(Vocabulary_bank))

	elif V==1:
		################# Building Vocabulary#################
		for letter in mystr:
			for letter2 in mystr:
				for key in Vocabulary_bank:
					#lower+lower
					Vocabulary_bank[key][letter+letter2] = smooth_value
					#lower+upper
					Vocabulary_bank[key][letter+letter2.upper()] = smooth_value
					#upper+lower
					Vocabulary_bank[key][letter.upper()+letter2] = smooth_value
					#upper+upper
					Vocabulary_bank[key][letter.upper()+letter2.upper()] = smooth_value


		################# Training for V = 0 #################		
		for tr_str in training_lists:
			tr_list = tr_str.split("\t")
			#print(filtered_str)
			for index,letter in enumerate(tr_list[3]):
				if letter.lower() in mystr:
					if index<(len(tr_list[3])-1) and tr_list[3][index+1].lower() in mystr:
						Vocabulary_bank[tr_list[2]][letter.lower()+tr_list[3][index+1]]+=1

		################# Result #################
		print(pd.DataFrame.from_dict(Vocabulary_bank).T)

	else:
		################# Building Vocabulary#################
		for letter in mystr:
			for letter2 in mystr:
				for key in Vocabulary_bank:
					#lower+lower
					Vocabulary_bank[key][letter+letter2] = smooth_value
					#lower+upper
					Vocabulary_bank[key][letter+letter2.upper()] = smooth_value
					#upper+lower
					Vocabulary_bank[key][letter.upper()+letter2] = smooth_value
					#upper+upper
					Vocabulary_bank[key][letter.upper()+letter2.upper()] = smooth_value

		################# Training for V = 2 #################	
		for tr_str in training_lists:
			tr_list = tr_str.split("\t")
			for index,letter in enumerate(tr_list[3]):
				if index<(len(tr_list[3])-1) and letter.isalpha() and tr_list[3][index+1].isalpha():
					try:
						Vocabulary_bank[tr_list[2]][letter+tr_list[3][index+1]]+=1
					except:
						if letter in mystr:
							for key in Vocabulary_bank:
								for let in mystr:
									print(let)
									Vocabulary_bank[key][tr_list[3][index+1]+let] = smooth_value
									Vocabulary_bank[key][let+tr_list[3][index+1]] = smooth_value
									Vocabulary_bank[key][tr_list[3][index+1]+let.upper()] = smooth_value
									Vocabulary_bank[key][let.upper()+tr_list[3][index+1]] = smooth_value
						else:
							for key in Vocabulary_bank:
								for let in mystr:
									print(let)
									Vocabulary_bank[key][letter+let] = smooth_value
									Vocabulary_bank[key][let+letter] = smooth_value
									Vocabulary_bank[key][letter+let.upper()] = smooth_value
									Vocabulary_bank[key][let.upper()+letter] = smooth_value
						Vocabulary_bank[tr_list[2]][letter+tr_list[3][index+1]]+=1

		################# Result #################
		print(pd.DataFrame.from_dict(Vocabulary_bank).T)

def trigrams(V,smooth_value):
	Basque = {}
	Catalan = {}
	Galician = {}
	Spanish = {}
	English = {}
	Portuguese = {}
	pass

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
			print("here")
			unigrams(V,smooth_value)
		elif size == 2:
			print("here")
			bigrams(V,smooth_value)
			#re.sub(r"[^A-Za-z]+", '', mystr)
		else:
			pass
			#mystr.isalpha()
			#break

		# for i in training_list:
		# 	print(i.split("\t"))
		# 	data_list = i.split("\t")


