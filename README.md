# solarOutputPrediction

The following code was built for a solar energy company who wanted to build out site specific models based on data they had, hoping to generate more accurate predictions than their existing models. The following code implements and tunes random forest regression models. The data itself is abstracted out of the actual model code and left as a parameter, so that the user of the code can simply add the requisite dependant and independent data, decide on a train, validate, and test split, as well as a number of other parameters, and the model will automatically generate and tune the optimal model (scikitlearn random forest regression model used for the model, and hyperopt is used for hyperparameter tuning).

The following code can be used to implement in just a few lines, optimized, robust random forest models based on whatever data you would like.

Within the context of figuring out snowshedding, more preliminary work was done in a few of the other files in order to calculate snowshedding and the speed at which snow fell off of the solar panels. 
