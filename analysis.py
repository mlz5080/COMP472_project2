import pandas as pd
import re
import numpy as np

def parse_file(test_file,training_file):
	if test_file==None:
		test_file = "./OriginalDataSet/test-tweets-given.txt"
	if training_file==None:
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

def get_sentence_number(dataset,file):
	key_list = ["eu","ca","gl","es","en","pt"]
	mydi={}
	for key in key_list:
		mydi[key]=0

	for i in dataset:
		data = i.split("\t")
		mydi[data[2]]+=1
	print(file, " has the following sentence frequencies",mydi)
	print("The total number of sentence is ",sum(mydi.values()))
	return mydi



def get_character_frequency(dataset,file):
	key_list = ["eu","ca","gl","es","en","pt"]
	mydi_uni={}
	mydi_big={}
	mydi_tri={}
	mydi_v1={}
	mydi_v2={}
	for key in key_list:
		mydi_v1[key]=0
		mydi_v2[key]=0

	for i in dataset:
		data = i.split("\t")
		mydi_v1[data[2]]+= len(re.sub(r"[^A-Za-z]+", '', data[3]))
		for char in data[3]:
			if char.isalpha():
				mydi_v2[data[2]]+=1
	print(file, " has the following character frequencies with V=1",mydi_v1)
	print("The total number of character with V=1 is ",sum(mydi_v1.values()))
	print(file, " has the following character frequencies with V=2",mydi_v2)
	print("The total number of character with V=2 is ",sum(mydi_v2.values()))

	return [mydi_v1,mydi_v2]
	
def get_angle(vector_1,vector_2):
	unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
	unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
	dot_product = np.dot(unit_vector_1, unit_vector_2)
	angle = np.arccos(dot_product)
	return round(angle, 2)
	
if __name__ == '__main__':
	dataset = parse_file(None,None)
	original_train_sentence = get_sentence_number(dataset[0],"Original training set")
	original_test_sentence = get_sentence_number(dataset[1],"Original testing set")
	print()
	original_train_char = get_character_frequency(dataset[0],"Original training set")
	original_test_char = get_character_frequency(dataset[1],"Original testing set")
	print()
	dataset = parse_file("./demo/test12.txt",None)
	demo_test_sentence = get_sentence_number(dataset[1],"Demo testing set")
	print()
	demo_test_char = get_character_frequency(dataset[1],"Demo testing set")

	print()
	print("The angle between original training and testing, frequencies of sentence is",get_angle(list(original_train_sentence.values()),list(original_test_sentence.values())))
	print("The angle between original training and demo testing, frequencies of sentence is",get_angle(list(original_train_sentence.values()),list(demo_test_sentence.values())))

	print("The angle between original training and testing, frequencies of character with V=1 is",get_angle(list(original_train_char[0].values()),list(original_test_char[0].values())))
	print("The angle between original training and testing, frequencies of character with V=2 is",get_angle(list(original_train_char[1].values()),list(original_test_char[1].values())))
	print("The angle between original training and demo testing, frequencies of character with V=1 is",get_angle(list(original_train_char[0].values()),list(demo_test_char[0].values())))
	print("The angle between original training and demo testing, frequencies of character with V=2 is",get_angle(list(original_train_char[1].values()),list(demo_test_char[1].values())))


