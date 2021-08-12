# Gray2Color
Implementation of [_Let there be Color!_](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/data/colorization_sig2016.pdf)
by Satoshi Iizuka, Edgar Simo-Serra and Hiroshi Ishikawa.  
  
  
<!-- ![Original Grayscale Image](https://github.com/bhattsahil1/Let-there-be-Color/blob/main/Documents/house.jpg) -->

Original Grayscale Image (any resolution)             |  Colorized Image
:-------------------------:|:-------------------------:
![Grayscale Image](https://github.com/bhattsahil1/Let-there-be-Color/blob/main/Documents/house.jpg)  |  ![Colorized Image](https://github.com/bhattsahil1/Let-there-be-Color/blob/main/Documents/result-house.jpg)

### Dataset download
[Places365-Standard](http://places2.csail.mit.edu/download.html) 
dataset will be downloaded and split into _train/dev/test_ subsets.
Out of the entire dataset, 10 categories will be considered.

```bash
>> sbatch dataset.sh (or simply run it as ./dataset.sh)
```

### Training phase
Run:
```bash
>> sbatch run.sh
```
Checkpoints of models are saved on every epoch.
Training can be interrupted and resumed anytime.
Resume by executing:
```bash
>> python3 loader.py places10.yaml --model [prev_model_path]
```
where [prev_model_path] is a previously saved model checkpoint.

### Colorize image
Select the best model and run:
```bash
>> python3 colorize.py [imgname].jpg [best_model_path]
```  
Colorized image will available with the name result-[imgname].jpg (Even .png files would work)  
