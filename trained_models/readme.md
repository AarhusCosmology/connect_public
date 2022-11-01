```connect```	will store all trained models here. Each model is a directory and can be loaded	with the command
```
tf.keras.models.load_model(/path/to/model, compile=False)
```

All information	regarding the output of	the model is stored in ```/path/to/model/output_info.pkl```, which
is a binary pickle file containing a dictionary. The following entries are included

|                       Description                       |                              Dictionary entry                              |
|---------------------------------------------------------|----------------------------------------------------------------------------|
| The names of the input parameters:                      |  ```output_info['input_names']```                                          |
| Emulated output types, ```Cl```, ```derived```, etc.:   |  ```output_info['output_Cl']```, ```output_info['output_derived']```, etc. |
| The $\ell$ values emulated:                             |  ```output_info['ell']```                                                  |
| A dictionary contining indices of output types          |  ```output_info['interval']```                                             |
| A dictionary containing information of normalisation    |  ```output_info['normalize']```                                            |

```output_info['normalize']``` has an entry called ```'method'``` where the method for normalisation is stored. Depending on the method,
other entries are used to revert the normalised output back to actual data, e.g. for ```output_info['normalize']['method'] = 'standardization'```
(default) the entries ```'mean'``` and ```'variance'``` will reverse the normalisation as 
```
model           = tf.keras.models.load_model(/path/to/model, compile=False)
data_normalized = model.predict(input_params)
variance        = output_info['normalize']['variance']
mean            = output_info['normalize']['mean']
data            = data_normalized*np.sqrt(variance) + mean
```

The model directory also contains all the test data excluded from the training data set as ```/path/to/model/test_data.pkl```. 
This can be used to statistically verify the performance of a model. For this purpose, validation accuracy and loss is also included
in seperate pickle files.