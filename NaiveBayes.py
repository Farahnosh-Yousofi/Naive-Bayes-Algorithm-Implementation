



#Seperate by class: for calculating the P(y) we have to separate data by their classes
def sperate_by_class(dataset):
    seperated = dict()
    for i in range(len(dataset)):
        datapoint = dataset[i]
        class_value = datapoint[-1]
        if class_value not in seperated:
            seperated[class_value] = []
        seperated[class_value].append(datapoint)
    return seperated

##Summarize the data set
## Finding the mean and standard deviation

#Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers) / float(len(numbers))


#Calculate the standard deviation of a list of numbers
def stdev(numbers):
    average = mean(numbers)
    variance  = sum((x-average)**2 for x in numbers)/float(len(numbers)-1)
    return variance**0.5


# Summarize a dataset
def summarize_dataset(dataset):
    summaries = [(mean(feature), stdev(feature), len(feature)) for feature in zip(*dataset)]
    del summaries[-1]  # remove the target column
    return summaries

dataset = [[3.393533211,2.331273381,0], [3.110073483,1.781539638,0], [1.343808831,3.368360954,0], [3.582294042,4.67917911,0], [2.280362439,2.866990263,0], [7.423436942,4.696522875,1], [5.745051997,3.533989803,1], [9.172168622,2.511101045,1], [7.792783481,3.424088941,1], [7.939820817,0.791637231,1]]

print(summarize_dataset(dataset))