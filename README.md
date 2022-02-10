# PRAT_Rain_Nowcasting
Repository for the PRAT Project about Rain Nowcasting

The code of the project is in the `src` folder. 


## 1. To train a model

To train a model you can use the file `main.py`. The command `python3 main.py --help` helps you look at the different parameters to specify. 

Example : 

```
python3 main.py --epochs 50 --batch_size 4 --input_length 12 --output_length 12 --network TrajGRU
```

Be careful, you need to change the folder path of the data in `main.py`. We also deleted from the dataset all PPMatrix folders since we don't use them to build the dataset. You may get an error if the dataset still contains those folder.  

The mean and variance parameters for the wind that appeared in `dataset.py` have been computed from the file `compute_wind_statistics.py`.

## 2. To evaluate a model. 

To evaluate the model you will use the file `eval_model.py`. You can use the command `python3 main.py --help` to look at the different parameters to specify. 

You need to provide the location to the `.pth` file that contains the network. 
You may also need to change the folder path of the data. 
