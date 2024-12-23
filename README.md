# Enhancing-Neural-Network-Interpretability
1.The dependencies required for the code are listed in the requirements.txt. Please ensure that the necessary dependencies are installed before running the code.

2.This method requires training a network model first. Please train the network based on the number of recognition categories you need to explain. Here, we provide the training method for Inception-V1.If needed, please run inception_v1_train_val.py in inception_v1 folder.

3. The trained network parameters need to be imported into our method, which will then be used to explain the model you have trained. Please export inception_v1.pb and place it in ACE folder.

4.The script susu_v1_ace_run.py contains the main steps of our method. This step allows the extraction of feature maps learned by any convolutional layer of the network and converts these feature maps into Concept Activation Vectors (CAVs). The so-called Concept Activation Vectors represent the concepts referred to in this method. So please run susu_v1_ace_run.py in ACE folder.

5. After obtaining the concepts, it is necessary to calculate the similarity between the concepts. Please run the computing_similarity.py file.
  
6. Select the highly similar concepts and place them in the Deep_Q_Network folder. Then, run run_this.py. This step reduces the multiple concepts that could cause incorrect decisions by the neural network to the minimum number required. In other words, the smallest set of concepts is identified to correct the neural network predictions.To specifically select similar concepts, please refer to the related paper for detailed guidance.
7. Instructions on the dataset：Since the dataset involves personal privacy, we only disclose a small part of it.

