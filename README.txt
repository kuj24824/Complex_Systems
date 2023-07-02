Complexity analysis of a City planning 2D Cellular Automata


-----------------------------------------
-------------DESCRIPTION-----------------
-----------------------------------------

The code we produced contains two models. One is supposed to represent 
the model used in the paper "Cellular automata and fractal urban form: 
a cellular modelling approach to the evolution of urban land-use
patterns" which tries to show how cities develope over time given
for possible states per location on a 2D grid. These 4 states are V
for Vacant, R for Residential, C for Commercial and I for Industrial.
This model recreates the plots seen in the paper, based on a fixed 
amount of cell-transitions per iteration. For complexity analysis on 
the otherwe chose to use a more probabilistic approach as comparison.
This entails creating cell transitions based on a probability that is
scaled by a downscaled version of the numbers given in the paper. 


-----------------------------------------
-------------GETTING STARTED-------------
-----------------------------------------

To run the code a jupyter notebook installation is required along with
a recent python release (eg. 3.9 as the time of writing). The python
packages are detailed in requirements.txt.

--------------Update the necessary python files here-------------



-----------------------------------------
-------------HOW TO USE------------------
-----------------------------------------
In order to run the main model, please open the "CA_city_revised_with _parameter_adjusting.ipynb"
file in CSS final code of main model folder to run the model. The final section of the file "Start Storing data", it stores the value
of some analysis of the model, such as fractal dimension, cluster information, conditional 
entropy, etc, with varying values of the growth rate of commercial and alpha (stochastic 
perturbation term). We also tried some other variations of changing parameters. For instance,
varying alpha and grid sizes. One can run "toring_data_Alpha_and_G_rate.py" and
"storing_data_Alpha_and_size.py" to store these additional data frames we used. However, 
in this case, we stored the final grids in npz file, and we used "CA_convert_npz_to_csv.ipynb" 
to convert these npz files into CSV data frame files. For convenience, one can directly use 
"cluster_info_size_and_Alpha.csv" and "cluster_info_growth_rate_and_Alpha.csv" to run the 
analysis of this second version of data frames in "Additional_data_analysis.ipynb" file. 
For the first version of the data frame, the data is stored as "CA_simulation_data_v1.csv",
and one can run analysis of the data by "Data_analysis_of_simulation_result.ipynb" . 

For our own model, please run each cell in 'BaseCityplanner.ipynb' in the 'Challenger Model'-folder 
for comparative probabilistic model solutions.
