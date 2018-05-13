To run the experiments follow below steps:

1.  Install Python 2.7 64 bit

2.  INSTALL BELOW PYTHON LIBRARIES:
	pip install scipy
	pip install numpy
	pip install matplotlib
	pip install pandas
	pip install sklearn

3. Put GP.py, main.py, probe_test.csv, test_limits.csv in one folder .

   If you want , you can configure the variables explained below as per your requirements.

   (otherwise the algorithm will run with deafult parameters)


4. run the main.py 


RESULTS FORMAT:

1. result folder will be created in the same directory where other python file resides

2. Inside result folder , folders will be created for each test

3. Output file format in each folder :

	By default all results with spatial temporal with progressive sampling and 

	"consolidated_test_results_proposed_mehtod.csv"    Intra wafer Kmean_Intra_wafer_clustering clustering == False and Inter wafer Kmean_Intra_wafer_clustering wafer clustering == True
	"consolidated_test_results_previous_mehtod.csv"   Intra wafer Kmean_Intra_wafer_clustering clustering == False and Inter wafer Kmean_Intra_wafer_clustering wafer clustering == False
	"consolidated_test_results_proposed_mehtod_cluster.csv"  Intra wafer Kmean_Intra_wafer_clustering clustering == True and Inter wafer Kmean_Intra_wafer_clustering wafer clustering == True
	"consolidated_test_results_previous_mehtod_cluster.csv"  Intra wafer Kmean_Intra_wafer_clustering clustering == True and Inter wafer Kmean_Intra_wafer_clustering wafer clustering == False

VARIABLES DESCRIPTIONS :


There are 2 files which has to be put in one common folder.
1. GP.py ----> GaussianProcess Class and some functions for 5 folds cross validation to find optimal hyper parameters
2. main.py ----> file to run the algorithm

In main.py :

There are 2 variable, which can be configured to run different combination of runs:

a) Kmean_Intra_wafer_clustering (Intra wafer clustering) b) Kmean_Inter_wafer_clustering (Inter wafer clustering)

if Kmean_Intra_wafer_clustering == True and Kmean_Inter_wafer_clustering == True :
	- Spatial Temporal Wafer-level Correlation Modeling with Progressive sampling and 
	- with inter wafer clustering and intra wafer clustering 

if Kmean_Intra_wafer_clustering == False and Kmean_Inter_wafer_clustering == False :
	- Spatial Temporal Wafer-level Correlation Modeling with Progressive sampling 
	- with no clustering

if Kmean_Intra_wafer_clustering == False and Kmean_Inter_wafer_clustering == True :
	- Spatial Temporal Wafer-level Correlation Modeling with Progressive sampling and 
	- with only inter wafer clustering 


The variables which can be configured for progressive sampling

sampling_steps = 4   #Progressive sampling steps

sample_perc = 0.025  #2.5 %    # % of training data in each sample step
							   # total training data budget 10%



Other Variables : 

i) total_devices :

Dictionary containing Index of the last die in each wafer (used to select data from different wafers from the probe_test.csv file)

if you are using different file other than probe_test.csv , change the key and value accordingly 

key of the total_device dictionary variable : Wafer No.
value of the total_device dictionary variable key : index of the last die in each wafer

total_devices : {0:2395,1:4784,2:7177,3:9567,4:11963,5:14357,6:16739,7:19134,8:21532,9:23925,10:26320,11:28717,12:31100,13:33493,14:35884,15:38264,
               16:40662,17:43056,18:45455,19:47847,20:50153,21:52551,22:54936}


For example If you want to use only 2 wafers , then assign the variable as :

total_devices : {0:2395,1:4784}


2) min_i :

Starting Index of First wafer (by default : 1)

If your starting wafer is other than first wafer, change it to accordingly


3) probe_tests : To read Data set file ( Put the files in the same folder where GP.py and main.py are there or give the actual path instead of "probe_tests.csv" ) 

 probe_tests = pd.read_csv("probe_tests.csv", engine ="python",header=None)

4) test_limits : file having specification limits for all the probe test :

test_limits = pd.read_csv("test_limits.csv", engine ="python",header=None)


5) Inter_wafer_clusters :
 
 Number of Inter wafer clusters 

 By deafult = 4 (for 24 wafers)

(We fixed the maximum inter wafer cluster size since we were limited by computation power (Didn’t have laptop with 16 GB RAM) so we tried to limit maximum wafer in each inter wafers cluster to 6 )


6) pb_idx range in the main.py file for loop :

  Modify according to your requirements.By defaults, for all the probe test , models will be created

 default : for pb_idx in range(3,105):      range according to probe_test.csv file …total 105 columns…form 3rd column to 104 columns , probe tests values are there 


By deault values for each of the above variables :

#Read the dataset
probe_tests = pd.read_csv("probe_tests.csv", engine ="python",header=None)
test_limits = pd.read_csv("test_limits.csv", engine ="python",header=None)

#total_test = 102


# Each wafer's ending index
total_devices = {0:2395,1:4784,2:7177,3:9567,4:11963,5:14357,6:16739,7:19134,8:21532,9:23925,10:26320,11:28717,12:31100,13:33493,14:35884,15:38264,
               16:40662,17:43056,18:45455,19:47847,20:50153,21:52551,22:54936}

#total_devices = {0:2395,1:4784,2:7177,3:9567,4:11963,5:14357,6:16739}
Inter_wafer_clusters = 4   #Inter wafer clusters 
min_i = 1
radial = False
Kmean_Intra_wafer_clustering = True   #Intra wafer clustering
Kmean_Inter_wafer_clustering = True   #Inter wafer clustering

#Progressive sampling steps
sampling_steps = 4
# % of training data in each sample step
#total training data budget 10%
sample_perc = 0.025  #2.5 % 



line nos in main.py for all variables :

300 to 316




