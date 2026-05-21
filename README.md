# FINAL PROJECT

## Usage Instructions

It's highly recommended to use a device with an NVIDIA GPU with drivers and CUDA pre-installed, as well as a Python installation. Then, run:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install matplotlib
```

That's all. If you want to retrain a model, empty the checkpoints directory (but don't delete the folder), and run train.py. Feel free to modify the parameters to your liking. If you want to benchmark the existing best checkpoint, use test.py and modify the parameters, and if you want visualizations, you can refer to main.py. 

To run train.py, simply modify the hyperparameters (or run as-is) and allow it to run until completion, either via the Run option in an IDE or python train.py while in the directory. You'll find checkpoints created in the checkpoint folder, but we provide pre-existing checkpoints from runs we've done if you want to skip this part. Running the test.py suite is identical- vary the parameters, run directly, no CLI interfacing required. To get visualizations of the model in action, navigate to train.py, and, for example if you want to generate 5 images of routing done to n = 20 points, you just add:

```python
if __name__ == '__main__':
        visualize(n=20, count=5)
```

To the bottom of train.py. You can use this to generate as many visualizations as you like, and stack them one after the other like so:

```python
if __name__ == '__main__':
        visualize(n=20, count=5)
        visualize(n=17, count=1)
        visualize(n=3, count=2)
        visualize(n=11, count=6)
```

Finally, if you want to benchmark the model on your own, hidden dataset, here are a few things to note. Make sure you create a .txt file and enter it inside the benchmarks folder. The file must be of this format:

```
0.0391, 0.2889
0.132, 0.9923
0.2421, 0.7734
```

Each row is the x, y coordinates of a point in a [0,1]² grid, and so they can't exceed 1 or be below 0. You can have a minimum of 3 rows and a maximum of 25, since our model is trained from n = 3 to n = 25. This is the format our model takes. Put as many valid .txt files with this format in benchmarks/ and run main.py with, at the bottom:

```python
if __name__ == '__main__':
        benchmarks()
```

And you'll find the Steiner-routed trees in the images/ folder.

To find the relevant WL and diagrams using Borah's Algorithm, go to the comparisons folder and run borah.py. To run FLUTE instead, run flute.py. Do note, we provide an interface for using these algorithms, but are not the original authors of the FLUTE and Borah's implementations. 

Credit for Borah's Algorithm: [Professor Sungkyu Lim](https://sites.usc.edu/ee680-lim/codes/)

Credit for FLUTE: [Chris Chu](https://github.com/The-OpenROAD-Project-Attic/flute/tree/master)

## Training and Testing

We trained the model, as described in the paper, with only two changes. First, we made the reduction in batch size from lower to higher n smoother, to better accomodate VRAM of lower sizes. Second, instead of training each degree for 40k steps, we did it for 10k steps. The device used was an NVIDIA 4070 GPU, and training concluded in ~6 hours.

We benchmarked REST, Borah, and FLUTE at (A = 3) and (A = 18) from n = 3 to n = 25, to identify where the model stands. We find REST beats Borah 17/23 times, beats FLUTE (A = 3) 17/23 times, but beats FLUTE (A = 18) only 2/23 times. On average, Borah yields a result 0.74% longer than REST, FLUTE (A = 3) 0.42% longer, and FLUTE (A = 18) 0.61% longer. In terms of speed, Borah's is thousands of times slower than the other three, with FLUTE (A = 18) and REST having comparable speeds, and FLUTE (A = 3) being a few times faster than these two. 

From this, we can conclude REST is comparable or superior to conventional algorithms, even when trained using limited resources. The full details are below. A more detailed ablation analysis of REST itself is in ablation.md.

For wirelength:

| Degree | Borah | FLUTE (A=3) | FLUTE (A=18) | REST |
|--------|-------|-------------|--------------|------|
| 3 | 1.0010 | 1.0016 | 0.9807 | 0.9956 |
| 4 | 1.2629 | 1.2800 | 1.2548 | 1.2762 |
| 5 | 1.5301 | 1.5257 | 1.4911 | 1.5027 |
| 6 | 1.6702 | 1.7026 | 1.7034 | 1.6978 |
| 7 | 1.8718 | 1.8683 | 1.8707 | 1.8710 |
| 8 | 2.0145 | 2.0333 | 2.0308 | 2.0370 |
| 9 | 2.2289 | 2.1796 | 2.1848 | 2.1808 |
| 10 | 2.3123 | 2.3344 | 2.3107 | 2.3186 |
| 11 | 2.4760 | 2.4492 | 2.4330 | 2.4511 |
| 12 | 2.6663 | 2.5731 | 2.5527 | 2.5683 |
| 13 | 2.6985 | 2.6827 | 2.6655 | 2.6867 |
| 14 | 2.8291 | 2.7996 | 2.7929 | 2.7997 |
| 15 | 2.9196 | 2.9256 | 2.8897 | 2.9083 |
| 16 | 3.0279 | 3.0149 | 2.9803 | 3.0048 |
| 17 | 3.1015 | 3.1126 | 3.0975 | 3.1035 |
| 18 | 3.2519 | 3.2188 | 3.1700 | 3.1956 |
| 19 | 3.3034 | 3.3011 | 3.2733 | 3.2890 |
| 20 | 3.3696 | 3.3885 | 3.3514 | 3.3828 |
| 21 | 3.4963 | 3.4989 | 3.4395 | 3.4659 |
| 22 | 3.5634 | 3.5837 | 3.5275 | 3.5503 |
| 23 | 3.6969 | 3.6503 | 3.6108 | 3.6311 |
| 24 | 3.7450 | 3.7393 | 3.6835 | 3.7109 |
| 25 | 3.8416 | 3.8148 | 3.7521 | 3.7920 |

For time (in ms):

| Degree | Borah | FLUTE (A=3) | FLUTE (A=18) | REST |
|--------|-------|-------------|--------------|------|
| 3 | 0.518 | 0.002 | 0.002 | 0.040 |
| 4 | 2.395 | 0.002 | 0.002 | 0.031 |
| 5 | 5.370 | 0.003 | 0.002 | 0.044 |
| 6 | 10.315 | 0.003 | 0.002 | 0.063 |
| 7 | 18.690 | 0.003 | 0.003 | 0.079 |
| 8 | 30.806 | 0.003 | 0.003 | 0.092 |
| 9 | 46.560 | 0.003 | 0.003 | 0.113 |
| 10 | 68.279 | 0.007 | 0.010 | 0.136 |
| 11 | 97.506 | 0.007 | 0.014 | 0.158 |
| 12 | 135.618 | 0.009 | 0.027 | 0.185 |
| 13 | 180.442 | 0.009 | 0.040 | 0.216 |
| 14 | 240.781 | 0.010 | 0.062 | 0.248 |
| 15 | 316.156 | 0.011 | 0.083 | 0.283 |
| 16 | 412.727 | 0.011 | 0.110 | 0.318 |
| 17 | 523.103 | 0.012 | 0.138 | 0.417 |
| 18 | 656.538 | 0.011 | 0.173 | 0.469 |
| 19 | 795.151 | 0.013 | 0.216 | 0.516 |
| 20 | 958.326 | 0.014 | 0.256 | 0.572 |
| 21 | 1179.169 | 0.014 | 0.299 | 0.625 |
| 22 | 1395.242 | 0.016 | 0.351 | 0.676 |
| 23 | 1665.355 | 0.016 | 0.406 | 0.723 |
| 24 | 1937.177 | 0.018 | 0.460 | 0.798 |
| 25 | 2401.332 | 0.018 | 0.513 | 0.850 |
