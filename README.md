# Enhancing-Neural-Network-Interpretability
The dependencies required for the code are listed in the requirements.txt. Please ensure that the necessary dependencies are installed before running the code.
1.pip install requirments.txt  
2.train inception_v1  
  run inception_v1_train_val.py in inception_v1 folder  (This method requires training a network model first. Please train the network based on the number of recognition categories you need to explain. Here, we provide the training method for Inception-V1.)
3. export inception_v1.pb and place it in ACE folder  (The trained network parameters need to be imported into our method, which will then be used to explain the model you have trained.)
4.The script susu_v1_ace_run.py contains the main steps of our method. This step allows the extraction of feature maps learned by any convolutional layer of the network and converts these feature maps into Concept Activation Vectors (CAVs). The so-called Concept Activation Vectors represent the concepts referred to in this method. So please run susu_v1_ace_run.py in ACE folder.
5. 

Instructions on the dataset：Since the dataset involves personal privacy, we only disclose a small part of it.
