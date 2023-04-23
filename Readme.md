# SIGIR 2023

>Candidate–aware Graph Contrastive Learning for Recommendation
## environment
<p>
This article is based on RecBole's implementation of CGCL on **RTX 1080 12G CUDA 10.2** devices using Pytorch, thanks to RecBole for its contribution.
The code is implemented using Python 3.7.7, and some of the key packages are as follows:
</p>

## package

>recbole == 1.0.0<br>
torch == 1.13.0<br>
torchvision == 0.4.0<br>
numpy == 1.21.6<br>
pandas == 1.3.5<br>
scipy == 1.6.0<br>


## Dataset statistics

>| DataSets | Users | Items | Interactions | Density | <br>
| Yelp | 45,478 | 30,709 | 1,777,765 | 0.00127 | <br>
| Gowalla | 29,859 | 40,989 | 1,027,464 | 0.00084 | <br>
| Books | 58,145 | 58,052 | 2,517,437 | 0.00075 | <br>

<p>
Notes:<br>
If you use Recbole's automatic data set download function, due to network problems, the downloaded dataset will be incomplete, and for your convenience, we also provide the original data set we downloaded.
</p>

## Important parameter settings:
### yelp2018 dataset

>embedding_size: 64<br>
n_layers: 3<br>
learner: adam<br>
learning_rate: 1e-3<br>
reg_weight: 1e-4<br>
ssl_temp: 0.1<br>
hyper_layers: 1<br>
alpha: 0.5<br>
beta: 0.5<br>
gamma: 0.5<br>
ssl_reg_alpha: 1e-5<br>
ssl_reg_beta: 1e-5<br>
ssl_reg_gamma: 1e-6<br>


### gowalla-merged dataset

>embedding_size: 64<br>
n_layers: 3<br>
learner: adam<br>
learning_rate: 1e-3<br>
reg_weight: 1e-4<br>
ssl_temp: 0.1<br>
hyper_layers: 1<br>
alpha: 0.4<br>
beta: 0.4<br>
gamma: 0.4<br>
ssl_reg_alpha: 1e-5<br>
ssl_reg_beta: 1e-5<br>
ssl_reg_gamma: 1e-6<br>

### amazon-books dataset

>embedding_size: 64<br>
n_layers: 3<br>
learner: adam<br>
learning_rate: 1e-3<br>
reg_weight: 1e-4<br>
ssl_temp: 0.1<br>
hyper_layers: 1<br>
alpha: 0.6<br>
beta: 0.6<br>
gamma: 0.6<br>
ssl_reg_alpha: 1e-5<br>
ssl_reg_beta: 1e-5<br>
ssl_reg_gamma: 1e-6<br>


<p>
Note<br>
For fair comparison, the results reported by GCL-based methods (SGL, NCL, CGCL) all use a 3-layer LightGCNGNN skeleton, and we conduct a grid search based on the results in the original paper to find the optimal parameters.
</p>

## command
<p>
You can use the following command to start three datasets separately, and the results will be saved:
</p>

>nohup python main.py --dataset yelp  --config_files=test.yaml --gpu_id=0 >CGCL-yelp.txt 2>&1 &<br>
>nohup python main.py --dataset gowalla-merged  --config_files=test.yaml --gpu_id=0 >CGCL-gowalla-merged.txt 2>&1 &<br>
>nohup python main.py --dataset amazon-books  --config_files=test.yaml --gpu_id=0 >CGCL-amazon-books.txt 2>&1 &<br>

<p>
We also provide Table 2, the optimal values reported by our proposed method, and the results on the three datasets are shown in CGCL-yelp.txt, CGCL-gowalla-merged.txt, CGCL-amazon-books.txt three files.
In addition, the training time of CGCL per round in the provided logs may vary greatly on different machines, as the device needs to be shared with other users.
</p>

## Contacte
>If you have any questions, please let us know.
