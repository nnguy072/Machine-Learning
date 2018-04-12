from pprint import pprint   #printing neatly for debugging

# ************************************************************************
# Question 0
# read in data from file and stored into 2d array
def populateDataSet(fileName):
  file = open(fileName, "r")
  fileBuffer = file.readlines()
  tempList = []
  for line in fileBuffer:
    noNewLine = line.strip('\n')
    if(noNewLine != ""):
      tempList.append(noNewLine.split(','))
  return tempList
# ************************************************************************

wineData = "wine.data.txt"
irisData = "iris.data.txt"

whichData = raw_input("Which data set?\n1. Wine\n2. iris\n")


if(whichData == '1'):
  dataSet = populateDataSet('wine.data.txt')
  pprint(dataSet)
elif(whichData == '2'):
  dataSet = populateDataSet('iris.data.txt')
  pprint(dataSet)