# colab-tensorflow-tpu-example
colab-tensorflow-tpu-example

Tensorflow v2와 Bert 모델을 이용한 colab GPU와 TPU로 학습하는 예제입니다.
두개의 ipynb는 동일한 코드이고, TPU와 GPU 사용 부분만 변경되어져 있습니다. 

### 1. example, method to change GPU mode and TPU mode in one code :

##### GPU
```python
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"],cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()) #GPU모드 
print('Number of devices: {}'.format(strategy.num_replicas_in_sync)) # GPU
with strategy.scope():
    sentiment_model = create_sentiment_bert() 
```

##### TPU
```python
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver) # TPU모드 
with strategy.scope():
    sentiment_model = create_sentiment_bert() 
```


### 2. model training speed

##### GPU
GPU was randomly selected for P100, k80, and T4. The picture below is T4
physical_device_desc: "device: 0, name: Tesla T4, pci bus id: 0000:00:04.0, compute capability: 7.5"
![screenshot1](https://github.com/hyeonsangjeon/colab-tensorflow-tpu-example/blob/main/pic/T4_GPU_time_per_epoch.png?raw=true)


##### TPU
![screenshot1](https://github.com/hyeonsangjeon/colab-tensorflow-tpu-example/blob/main/pic/TPUs_time_per_epoch.png?raw=true)
