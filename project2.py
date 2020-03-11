

def parse_file():
	test_file = "./OriginalDataSet/test-tweets-given.txt"
	training_file = "./OriginalDataSet/training-tweets.txt"

	test_string = ""
	training_string = ""

	with open(test_file, "r") as file:
		for line in file:
			test_string+=line

	with open(training_file, "r") as file:
		for line in file:
			training_string+=line

	return [training_string,test_string]


if __name__ == '__main__':


	print(parse_file()[0])