# SageMaker-ML-Projects
This is my journey on SageMaker!

   **ML-SageMaker 1.0**

 → Fully managed Machine Learning service that developers and DS can use to build train and deploy machine learning models.

**Notebook Instances**:


1. EC2 instance - elastic compute cloud and spin up a managed instance.
2. Not going to have SSH access - we want to pick the right family and there are families {t, m, c, p} → Pick the right version, ml.t3.medium.
3. Pick the right version too - every version is the latest version provided by the instance.
4.  Latest version of the instance: always more cost optimal.


**Add an EBS volume**


1. SageMaker put these together - stores data for us and by default, we get 5 GB and if a ML model has more than this memory, it will pick the right memory EBS volume for you.
2. Everything on the EBS volume persists.
3. Add or create a git repository - this repository will give access to code and within SM, we can add repositories.


**Security Settings**


1. Encryption
2. Root Volume Access
3. Internet access
4. VPC connection


**Use a Lifecycle Config**


1. **Create, start a notebook** - it is a bash script and we can copy data from S3 with a timeout for 15 minutes
2. Run in the background with “&”


If you want to run an ML model on that notebook instance:


**1. Attach a portion of the GPU to the EC2 instance (Elastic Inference)
2. Select on the basis of size, version and bandwidth.
3. Terminal is extremely important since it gives information about
    1. Instance operating at on a line by line basis
    2. EBS volume starts at the pagemaker work
    3. cd into SageMaker
4. There are 200 example learning models - go over them and think through the data distribution - use and it will copy it in your own notebook**


**Pointers**:


1. Keep points and cost low by depending on LAMBDA - Turn notebook instances off when need be
2. Resize on the fly → We can increase the notebook instance, stop it and edit the EBS or EC2 instance
3. Multi threading
4. Execution role - keep in mind the policies in mind


**Amazon SM Built-In Algorithms**


1. Need to focus on a machine learning model and data to balance.
2. Overfitting: If the model is too big for the data that we are using, it is going to be too specific and not be able to generalize in a better way.
3. Undercutting: If the dataset is too large and model is not large enough, we will underfoot.
4. We need to know that we can collaborate with domain experts to build ML models → Take that information and add as columns and features in the model and use that to collaborate with domain experts


**Breaking Down the Problem**


1. Use case: 
    1. What data do I have on this data?
    2. What model should I use (out of the whole toolkit)?
    3. Based on the model, how do I frame the problem to use models I know? (We will map them to the ML based solution)


**Built-In Algorithms**


1.** Classification**: Compromise 80% of the real world scenarios
    1. Linear Learner
    2. XGboost
    3. KNN
    4. Factorization Machines
        (Think about a real world problem, and map it to one of the algorithms)
2. **Computer Vision**:
    1. Image classification - What class does the image fall into
    2. Object detection - based on the training data, we want to identify the object
    3. Semantic segmentation - When we need to draw pixel maps, to know the different between a cat, sidewalk and so on.
3. **Topic Modeling**: Data set full of tweets, and if we want to look for topics, we can build that with topics
    1. LDA
    2. NTM
4. **Working with Text**:
    1. Blazing text:
        1. Supervised (supports classification)
        2. Unsupervised (gets embeddings)
5. **Recommendation**: Data can be used to be managed.
    1. Factorization Machines (KNN)
6. **Forecasting**: Datasets related to each other (Stock markets, etc) 
    1. DeepAR
    2. Capture the inter relationship between the relationships
7. **Anomaly Detection**: Unsupervised algorithms
    1. Random Cut Forests *
    2. IP Insights *
8. **Clustering**:
    1. Kmeans
    2. KNN
9. **Sequence Translations**: Translate various sequences (building a language translator)
    1. Seq2Seq
10. **Regression**: Algorithm to generate a continuous data
    1. Linear regression 
    2. XG Boost
    3. KNN
11. **Feature Reduction**: When we need to break down a large dataset, we can plug into the PCA, and break it into sub components.
    1. PCA
    2. Object2Vec

**Built in algorithms: Are registered in the ECR (Elastic Container Registry)**

**Bring Your Own Model (BYOM)**


1. There are many ways to train models on SM, the first one is to use Built In algorithms
    
2. **Script mode**: Can bring the own docker container, leverage the solution on AWS market place, or train it on the notebook instance.
    1. AWS SM manages a Docker Container in ECR
    2. How do I write my own model?
    3. Pick one of the open source containers that we are actively managing.
    4. We will write the code for the model and run it on those containers
    5. Point to the AWS container
    6. Write the code
    7. Specify the entry point in the SM estimator
    8. Include any extra libraries
    9. Use our web-server for inference
3. Write your own model however you please
    1. Point to your model within the docker file
    2. Register your container on ECR
    3. Point to your container’s address in ECR
    4. Don’t forget to implement a serve() function.


**ML Marketplace**:


1. **Algorithms**: You can train your data on a set of code
    
2. **Models**: Pre trained model artifact
    1. Both of them are on the subscription model but there is a free tier.


**Training and Tuning**:

Every model run on SageMaker training job has its own ephemeral cluster. We will have a dedicated EC2 instances alive for the number of seconds that the model is training on.


1. If there are multiple ML models, there can be dedicated instances for each of these.
2. Cluster comes down immediately as the model stops training:
    1. No More Virtual Environments
    2. No more conflicting packages or dependencies
3. **Estimator**:
    1. Execution role, 
    2. Algorithm, 
    3. SageMaker Estimator (these components are constant)
4. **Splitting Data for ML**:
    1. Take a dataset to make decisions
    2. All of the columns are there and split the one column
    3. After Feature Engineering: Split into train, validation, test sets
        1. Training: 60-85% of the data
        2. Validation & Test: rest
    4. Perform a technique on validation data and check how model did on the test data.
    5. Send these to channels in SM

First, fit the model into the training data, look at the performance of the model on validation (perform hyper parameter tuning on validation data), and then how well the model did on the test data.


1. Send the data splits into S3 channels
2. Call Model.fit - a new cluster comes online, and send logs out to cloud watch
3. Monitor the performance.


Now to evaluate the model, we do the following:


1. Confusion Matrix
2. Recall: How many positive cases are we catching? (Number of true positives)
3. Precision: Starts with true positives but it is the true positive rate (Great precision if model precision positive once someone tested for diabetes)

**Use both of the following for hyper parameter tuning:**


1. **Parameter**: Features/Data supplied to the model
2. **Hyper parameter**: Extra math (Extra terms) → real numbers of the data
    1. In SageMaker:
        1. Pick the **HyperParameters** and the **ranges**
        2. On **XGBoost**:
            1. eta, min_child_weight, alpha, max_depth
            2. Objective metric: term used to check how well the model performs
            3. Job Parameters: Tell SM the total number of jobs
                1. Number of total jobs
                2. Number of jobs in parallel


→ 3 models are pulled and trained against the objective metric with a baysian optimizer.

1. **Tuning**: Trying as much as you can (running large number of jobs)

**Deploy your ML models to Production At Scale**


1. **Endpoint**: These can be multiple EC2 instances in multiple AZs, to promote high availability.
2. When you specify model.deploy, and this spins up a managed endpoint, and if it is more than one EC2 instance, it will serve prediction responses.
3. Load Balancer → Health checks.
4. Endpoint → RESTFul API
    1. Customers use LAMBDA to connect to two points


→ SM Endpoint - best for online inferences - data from internet and serve responses with low latency

**Tuning ML Models to High Accuracies: SM Automatic Model Tuning**


1. **Training jobs**:
    1.  Notebook instance with EC2 and EBS volume
    2. Call Model.fit, shoot data to an S3 bucket
    3. SM spins additional EC2 instances (Ephemeral cluster - dedicated to the specific model)
    4. Image → lives in ECR (Elastic Container Registry)
    5. SM copies image onto that ephemeral cluster → training process
    6. SM writes the trained model back to s3
    7. Scalable, cost efficient.
    8. What are tuning jobs? 
        
2. **Hyperparameter Tuning Jobs**:
    1. We most commonly start the notebook, create a tuner, then tuner.fit
    2. SM first copies data out to S3, and after this, Model training happens in multiple rounds:
        1. SM spins 3 different versions of the models (Docker containers) → 3 training jobs run against the data
        2. Results are pulled into the Bayesian Optimizer (object performance of the model is pulled in)
        3. BO looks at the AUC, and re optimizers 3 more jobs with different hyperparameters (6 jobs in total now)
        4. At the end, we get a graph:
            
    3. 
3. How to set up a hyperparameter tuning job:
    1. Objective Metric (AUC) → Recall, precision, etc.
    2. Hyperparameters & Ranges → In the case of XGBoost - maximum depth of tree
    3. Job Specs → think about the total number of jobs to run in parallel in periods of time


**How to use Hyper parameter tuning with my own model?**


1. Use hyper parameter tuning with the built in algorithms, but to bring model in docker and script mode, use the customized manner.
2. Set up a single single dictionary for this.
3. If you like the tuner, and do not want to wait for 7 rounds, use the random search.


1. 
2. Maximize efficiency across tuning jobs, inherit the parent tuning job, using the warm start - need the identical data and algorithm.
3. You can set up Transfer Learning if you want to use more data, and pick up where a previous parent job left off and apply that to a new child tuning job.


How to compare results across tuning jobs?


1. **Use SM search** → 


**Machine Learning Process**


1. **Business Problem **→ Identify the customer need, talk to the customers, and analyze some of the pain points.
2. **ML Problem Framing** → What is the framework you want to apply to solve the need/problem?
3. **Data Collection** → Need to have previous data to validate the problem and solve that. For example, to solve skin cancer, you need to have images of that data.
4. **Data Integration** → Identify the external factors
5. **Data Preparation Cleaning** → This is time consuming, and we want to make sure the data is noiseless and completely clean.
6. **Data Visualization** → Visualize the data and how it looks like
7. **Feature Engineering** → Let’s day there are multiple columns in the database and there is no single meaning of a single row or column, but getting all of them together makes sense. Select the algorithm for the similar problem:
8. **Model creation** → Evaluate the model based on the algorithm in different ways 
9. **Revisit the business problem**/add more data/more feature engineering to validate the model outputs
10. Once business goals are met, **deploy the model for more consumption**, and monitor or debug the model and repeat.


1. **Data breakdown**:
    1. SageMaker does not help here, we focus on 
        1.** S3
        2. Glue
        3. Athena
        4. EMR
        5. Redshift & Spectcrum**
    2. These services help collect and clean data
        
2. **Visualization**:
    1. Select the ML Algorithm
    2. Select the visualization features
    3. DL Algorithm Training
    4. Evaluation Metric
    5. SageMaker provides a configured environment to make this step possible → We can use this to manage and set up the training cluster.
    6. We can distribute and manage these training clusters → Once we identify the algorithm, we scale, distribute and train the model to make it time efficient.


**4 Main SageMaker Features**


1. **SM notebooks service**:
    1. Agile, Reliable
    2. Built in notebook and readily available to use right away
    3. Create in a small or GPU powered machine
2. **SM Algorithms**:
    1. Built in and created grounds up to make sure these algorithms can be distributed and run in parallel
    2. 14 different algorithms
    3. You can bring in your own algorithm
3. **SM Training Service**:
    1. One click training
    2. Managed Distributed model training service
4. **SM Hosting Service**:
    1. Support the deployment process
    2. Instances, Autoscaling, APIs, Load balancing

**SM is modular - we do not have to do all of these in parallel! We can only use one of these if we need to.**


  **Distributed ML training with PyTorch and Amazon SageMaker**



**1. What is distributed training? When to use it?
2. How to apply PyTorch to distributed training?
3. SageMaker/Distributed Training/Tools Needed?**


**Goal**: Reducing time-to-train of your PyTorch models is crucial to improving your productivity and reducing your time-to-solution. Goal is to learn how to efficiently scale the training workloads to multiple instances, with Amazon SageMaker to do the heavy lifting. We will learn how to bring in the PyTorch code to distribute training across a large number of CPUs and GPUs.

**Steps are followed in SageMaker and provided in the GitHub repo.**

The following steps are given:


**1. Preparing a dataset for SageMaker for uploading them to S3
2. Writing the PyTorch training script for the purpose of distributed training
3. Writing the SageMaker SDK functions to run distributed training
4. Running distributed training jobs on the specified number of CPU instances
5. Deploying the trained models to endpoints using SageMaker and evaluating them
6. Running high-performance and large scale training on the GPUs**


**What is the primary objective for training?**

→ How much time it takes to train the model to reach the desired accuracy. How can we speed up this training process and provide more compute - there are two different paradigms:


1. **Scaling up**: Replacing with a more powerful CPU/GPU (this is preferred before going to distributed training, once we hit a wall with this use:)
2. Scaling out: Taking the dataset and using multiple machines to train faster.

** **Always scale up before scaling out**.


**How distributed training works?**


1. In a typical training process, we split up the dataset into batches (deep learning) and these go through a training process (forward/backward pass of neural networks)
    
2. **Bottleneck**: these batches go through sequentially
    
3. When we have more CPUs/GPUs: In this way we can spread the batched to different instances and have copies on different instances, and be able to effectively train on a larger batch size.
    
4. Have to make changes to the code often to manage these instances (CPU/GPUs)


**Approaches to Distributed Training**



1. Parameter server approach: All worked could either be CPUs or GPUs and there was a central node (that was a bottleneck) and responsible for taking care of the workers and so on.
    
2. Ring All Reduce: Removing the central authorities, and the instances can communicate and perform, and all reduce operation is a way to average gradients across all different nodes.
    
    PyTorch has several ways to achieve this in a faster way. We do not have to wait for the entire gradient computation but we can do it in parallel.


**How PyTorch and SageMaker work together?**


1. PyTorch: We write code on this for deep learning frameworks and train models.
2. SageMaker: Managing the infrastructure, training, building and deploying models.


We bring in the PyTorch training script and then we bring in the SageMaker SDK API (estimator or fit) and then SageMaker trains this on 8 GPUs or CPUs (on the estimator function). It takes the PyTorch training script, go through the training on the specified number of instances that were specified. It also manages the dataset movement from s3 to instances and vice versa. 

Think of it as a shared responsibility model: We write the script, the training model and writing the sagemaker SDK estimator and fit. SageMaker is responsible for the infrastructure, data movement between s3 and backing up the models and the checkpoints to s3. It takes care of the infrastructure in a managed way. Let’s dive into the code.


