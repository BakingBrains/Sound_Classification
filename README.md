# Sound_Classification
Sound classification on Urban Sound Dataset</b>

Download the urban sound dataset [here](https://urbansounddataset.weebly.com/urbansound8k.html) </br>
- Extract the dataset on to a folder </br>
- Install pytorch (gpu verison if you have gpu) </br>
- Install the requirements</br>
```
pip install -r requirements.txt
```

- Run the dataset.py file (change the path to the dataset)</br>
```
python dataset.py
```
This will give the number of items in the dataset</br>

- Then run modelcnn.py to know about the structure of the CNN model.</br>
- Then start the training. (change the path to the dataset)</br>
```
python train.py
```

- After the training, the model will be saved in a specified folder. Use the saved model for inferencing. </br>
Run </br>
```
python inference.py
```

Reference: https://github.com/musikalkemist/pytorchforaudio
