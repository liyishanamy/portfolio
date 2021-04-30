## About Me
<p>My name is Yishan Li. My friends always call me Amy. I am from Chengdu, Sichuan, China. I did my undergrad in Computer Science at Queen's University and I am currently completing my master degree(data science and artificial intelligence) at University of Waterloo.</p>
---


### Software Development

**Health Track[NoSQL database project]**
<p>This project is called the health track. This platform is designed to help doctors to keep track of the patients who are tested positive of COVID19.  Patients can use this platform to report their daily health status (including today's symptoms, today's temperature, whether they have gone somewhere. if they have, whether they wear a facial mask). The doctor can be able to manage all his patients' health status data and overall statistic of the data(data, bar char, line graph etc) so that they can visulize the situation more clearly and they can also care for their patients more efficiently. Only after 14 days without fever and no symptoms can they make an appointment with doctor. Doctor can also manage all the appointments made by patients, they can also update the test result via this platform.  </p>

---
- [Front End](https://github.com/liyishanamy/healthtrackingplatform_frontend)
- [Back End](https://github.com/liyishanamy/healthInformation_backend)
- [API](https://docs.google.com/document/d/15rplKs8JGBj297dSzQFhMPIbVfc3abBbPUhmvtrWwfU/edit)

***Highlights of this project***
<ol>
  <li>Use <strong>Mongodb</strong> to store all the data</li>
  <li>Use <strong>Express</strong> to make API service</li> 
         <ul>
            <li>Applying both <strong>access token and basic authentication</strong> method to request access to the API</li>
            <li>Implementing <strong>Authentication+Access Token+ Refresh Token</strong> using <strong>JWT</strong></li>
            <li>Applying <strong>image hash</strong> mechanism to save the users' profile pictures to stregthen the security</li>
            <li>Using <strong>hash/salt</strong> to store users password. </li>
            <li>Implementing <strong>rate limit</strong> to restrict the use of some API(ex, Users can only change password once a day)</li>
            <li>Applying <strong>Pagination</strong> to the some of the API(which return large amount of data)</li>
            <li>Implementing Chatting System using <strong>socket io</strong></li>
         </ul>
        
  <li>Use <strong>Reactjs</strong> to make frontend</li>
        <ul>
          <li>Applying invitation code mechanism to cluster patients</li>
          <li>Using <strong>regex</strong> to check the whether the password is strong enough</li>
          <li>Displaying Data visualization(Visualization of the patients' data)+ Export the Data into CSV</li>
          <li>Using <strong>Redux</strong> to manage the software global state</li>
          <li>Using existing Library--<strong> Mapbox API</strong> to record and intuitively display where the patients live</li>
          <li>Chat system supports both group chating and private chatting</li>
          <li>Performing quality assurance after implementation and taking notes of the API documentation</li>
        </ul>
   </ol>
---

**Movie online platform[SQL database project]**
<p>Database Design Project (SQL) developed a management system for cinema to store and manipulate movies and search movies or tickets by customers.  Customers can sign up and then sign into the account to view movies time slots, buy movies tickets, manage the movies tickets, etc, whereas admin can manage all the customers' status, movies status, and check some related statistics about the recent movies, for example, which movie is the most popular one recently, etc. </p>

- [Front End](https://github.com/liyishanamy/movie)
- [Back End](https://github.com/liyishanamy/movie)

***Highlights of this project***
<ol>
  <li>Use <strong>XAMPP</strong> was installed to grant the local server setup</li>
  <li>Designing <strong>ER-model</strong> and translating the model to relational schema</li>
  <li>Database structure can be clearly established in <strong>phpMyAdmin</strong></li>
  <li>Developing the movie theatre website using <strong>PHP</strong></li>
  <li>Customizing the user interface.</li>
  <li>Performing quality assurance after implementation</li>
</ol>

***Eat What***
<p>This is a personal project a mobile app using React native, which randomly gives a recommendation of what to eat nearby based on the user’s preferred transportation. Users can be able to vieww the reviews and rating associated with the recommendation restaurants. </p>

***Highlights of this project***
<ol>
      <li>The inspiration of doing project is originated from the real-life pain point. During the internship, my coworkers and I had a hard time deciding what to eat. So by designing this app, we could be able to get recommendation of what to eat. </li>
      <li>The functionality of the mobile app was integrated by handling <strong>yelp API and apple maps API</strong>.</li>
      <li>Implementing the application using <strong>react native</strong></li>
      <li>The users will select the transportation. Based on the transportation, our application can provide the recommendation restaurants to the users. Users can view the reviews and ratings of such restaurant. If user does not like the recommendation, our system will give another one, so on so forth until users satisfies.  Once satisfied, users can click on "Let's go" button, the system will redirect you to the apple map with the routes based on the transportation method you chose</li>
</ol>


***Aranyaka--Mini wechat program***
<p>Worked with student-run software-development club to build up an event-organizing tool that intends to help students organize an event using mini program in WeChat.</p>

***Highlights of this project***
<ol>
  <li>The mini program was developed using <strong>JavaScript</strong> on cloud based Tencent platform.</li>
  <li>The user data was managed by the Tecent platform cloud service</li>
  <li>This project involves the process of doing research, defining business requirement, surveying the user groups, designing and optimizing user interface before starting developing the application</li>
  <li>Ensure that the Wechat user should grant the permission before they get authorized to the service.</li>
  <li> Focus on developing user profile page and the main page </li>
  <li> Following the API document to ensure that the our data can be properly stored/updated/removed/showed in the database and can be successfully manipulated and interacted with the front-end program.</li>
   <li>Performing quality assurance after implementation</li>
</ol>


### Data Science

---
***Movie Recommendation system***
- [Movie-Recommendation system](https://github.com/liyishanamy/MovieRecommendationSystem/blob/main/movieRecommendationWithFeatureEngineering.ipynb) 
- [Report](https://github.com/liyishanamy/MovieRecommendationSystem/blob/main/d4.pdf)

Nowadays, the explosive growth of digital information creates a great challenge for users, hindering timely access to items of interest available online. Recommendation systems help drive user engagement on the online platforms by generating personalized recommendations based on a user’s past behaviour. In recent years, the recommendation system has become an effective approach to avoid information overload for users. How to effectively recommend a personalized
movie to the target user becomes a significant research problem. Recent research has focused on improving the recommendation system’s performance by extending the contentbased approach, collaborative filtering approach and hybrid approach. However, in this paper, I focus on developing a data-driven hybrid solution to deliver recommendation tasks. The recommendation system trains three regression models, namely KNN, ridge regression model, and random forest models, on a new dataset produced by leveraging rating averages, content-based approach and collaborative filtering approach. Then, I empirically compared three hybrid regression
models’ performance with the baseline model, matrix factorization algorithm. The test result shows the remarkable effectiveness of my proposed hybrid solution over the baseline model. The best-performed model, the random forest algorithm, improves 10.8% compared to the baseline model. This
study explores that enhancing the models’ performance is not restricted to extending the models. However, the feature engineering procedure can also extensively boost the recommendation system accuracy and enable the regression models to deliver better recommendation tasks than a robust algorithm, matrix factorization without using feature engineering.

***Highlights of this project***
<ol>
  <li>The dataset I used to train and evaluate the model is MovieLens(ml-latest-small), a subset of the movie dataset(ml-largest-large). The main dataset I used is rating, linking, moviemetadata</li>
  <li>A new dataset with new features is constructed by leveraging averages, content-based approach, collaborative filtering approach.</li>
  <ul>
    <li>Averages features: The average ratings of all movies given by a specific user and the average ratings of a specific movie provided by all users. These two averages are added to the rating table for all users and movies in each row</li>
    <li>Content-based features: I started by combining all the metadata features I am interested in, which are keyword, director, genre, and main character. Then I applied TFIDF to capture how important a particular term is to a movie document. The sequence of TFIDF scores has now become the new feature for each movie. Similarly, I applied cosine similarity between the TFIDF scores for each movie and obtained a movie similarity matrix. For every movie in the movie similarity matrix, I recorded the top three most similar movies rated by the target user as the new features. If insufficient features are obtained, I will replace the feature with the corresponding movie average rating.</li>
    <li>Collaborative filtering features, I started by constructing the user movie interaction matrix from the user rating dataset and applied the cosine similarity on each user. For every user in the user similarity matrix, I extracted the top three most similar users who also rated the particular movie and recorded their ratings as new features. If insufficient features are obtained, I will replace the features with the corresponding user average ratings. </li>
   </ul>
<li>Experiment Design:Matrix factorization is a robust algorithm that works by decomposing the user-movie interaction matrix into a product of user matrix and movie matrix using singular value decomposition. In this study, the matrix factorization algorithm is treated as a baseline model. The proposed models are KNN, ridge regression and random forest algorithms. The new dataset is randomly split into 90% training data and 10%testing data. The baseline model and proposed models are trained on the same 90% of training data and evaluated on same 10% testing data but with different features.I use RMSE and MAE as the evaluation metrics to evaluate the performance of the models.5-fold cross-validation are applied on the proposed models to find the best set of hyperparameters that fit the models well.
</li>
<li>Model:</li>
 <ul>
   <li>KNN is one of the traditional models in the recommendation system. It works by grouping the target user with the k nearest neighbour to compute an average of the neighbours’ ratings.The fivefold cross-validation result suggests me to choose neighbour size to be 8. </li>
   <li>Linear regression establishes a  relationship between the dependent variable and one or more independent variables using a  best fit straight line. Ridge regression works by introducing a penalty term in the objective function to prevent the model from being overfitting.Since the error change is not sensitive to the magnitude of lambda. Five fold cross-validation suggests me to choose lambda - 1.5</li>
   <li>The random forest algorithm is an ensemble learning model that prevents the model from being overfitted by constructing multiple decision trees and allows each decision tree to train on samples of the training data.  This ensures that the regression model does not overly rely on any individual features.Five fold cross-validation suggests me to choose max_depth is 10. </li>
   <li>Result: Baseline model performs well during training time but perform very badly on the testing data, which indicates overfitting occurs. Three proposed models performed significantly better than the baseline model. In particular, The random Forest model achieved the best performance among all regression models, making a 10.8% improvement compared to the baseline model. Those models are also tested using the five times two cross-validation paired t-test to compare if the performance of the two models is significantly different. The test result shows that the improvement made by proposed regression models is not due to statistical chance. 
   </li>
   </ul>
   <li>Another experiment is carried over to determines how different features impacts the model performance. I divide the eight new features into three new subset,containing averages features, content based features and collaborative filtering features.Then I trained the 3 proposed models on these sets respectively. It is surprising to find that the movie and user rating averages are the most important features.Also, combining all eight features allow the models to learn the extra correlation between features and thus extensively enhance the model performances.  In this project, the dataset I used is small and some users only rated a small portion of movies. This leads to the data sparsity problem that the system has difficulty finding sufficient similar users and therefore largely limits the usefulness of the extracted features.This might be why content based  features and collaborative filtering features are not contributing that much. 
   </li>
   
</ol>
  
 
***Data-intensive Distributed Computing***
<ol>
  <li>Counting the word in a long article and calculating the PMI between words using <strong>Mapreduce -Java</strong> with algorithm <strong>Pairs and Stripes</strong></li>
  <li>Counting the word in a long article and calculating the PMI/Bigrams between words using <strong>Spark -Scala</strong> with algorithm <strong>Pairs and Stripes</strong></li>
  <li>Optimizing the inverted Index project(search) by implementing index compression,buffering posting, term partitioning using <strong>Mapreduce-java</strong>.  </li>
</ol>


***Music Genre classification***
- [Music Genre classification](https://github.com/astralcai/music-genre-classification) 
- [Report](https://github.com/liyishanamy/music-genre-classification_report)

<p>Worked with my team to deploy different deep learning models to classify the different music genres using TensorFlow.</p>

***Highlights of this project***
<ol>
  <li>Selecting <strong>GTZAN and benchmark</strong> pre-existing dataset to train the model</li>
  <li>Apply <strong>short-time Fourier transform</strong> to conduct data preprocessing. Specifically speaking, to tranform the invisible music data to visible spectrogram</li>
  <li>Deploying <strong>Convolutional Neural Network</strong> and train the model to learn different music genre spectrogram.(Predicted the testing set correctly with 82% accuracy after training and optimizing our models)</li>
  <li>We chose pre-trained network(VGG-16) as our second model.With the replacement of the original pre-trained fully connected layer and softmax classifier on top of the convolutional layers with our own fully connected hidden layer, and a softmax output layer of 5 nodes(5 possible music genre). We found out the VGG 16 model does not work well upon the spectrogram </li>
  <li>Deploying Recurrent Neural Network and train the model to learn different music genre spectrogram.(Predicted the testing set correctly with 84% accuracy after training. To make an optimization of the RNN model, we explored another approach--RNN+attention which further enhanced the prediction result(Accuracy reaches 88% upon the unseen testing set) </li>
  <li>We also applied several approaches to prevent overfitting</li>
  <ul>
    <li>To reduce the number of parameters if possible. To decrease the complexity of the model, we chose to only use the output of the final BGRU layer, which improved the classification accuracy by around 2%.</li>
    <li>Another approach is to use validation data during training. The model was set to stop training and revert to the weights which produced the best results if the validation accuracy does not improve for 3 consecutive epochs</li>
    <li>We also added dropouts to the classifier, where some neurons are ”shut off” for every iteration. This prevents the network from depending too much on the outputs of a small subset of neurons.</li>
  </ul>
  <li>Writing the final report(paper) to summarize the project</li>
</ol>

---
***E-commerce product classification***
- [E-commerce product classification](https://github.com/liyishanamy/kaggleCompetition) 
<p>Applying ensemble learning to classify 27 E-commerce products based on the categorical data(gender/baseColor/season/usage/noisyTextDescription/noisy Image)</p>
***Highlights of this project***
<ol>
  <li>Clean and preprocess the dataset</li>
  <ul>
    <li>Turn the categorical data to one hot vector</li>
    <li>Clean up the noisy text description</li>
    <li>Merge the categorical data with the </li>
    <li>Match up the noisy image with the corresponding training label.</li>
  </ul>
  <li>Algorithm used</li>
  <ul>
    <li>Apply the random forest with adaboost on categorical data</li>
    <li>Apply the multi layer neural network on categorical data</li>
    <li>Apply tfidf on merged text description and SVM as the model </li>
    <li>Use the pre-trained word embedding to transform the text description data and train the bidirectional LSTM on the text description data </li>
    <li>Train CNN model on the noisy images with augmentation</li>
    <li>Train CNN model on the noisy images without augmentation</li>
    <li>Concatenate two CNN models(one is to train on text description/one is to train on noisy images) on the dataset and merge the feature maps generated from two CNN and train the feature maps using multi layer neural network</li>
    <li>Apply the weighted majority voting based on the predictions made by 7 models</li>
  </ul>
  <li>Get more than 96% accuracy on the testing data</li>
</ol>


***Toxic-comment detection project***
- [Toxic-comment detection project](https://github.com/liyishanamy/detect_toxic_comment) 
- [Report](https://github.com/liyishanamy/detect_toxic_comment/blob/master/cs651_finalproject.pdf)
<p>Applying ensemble learning to build a binary classifier(clean/toxic comment)and a multilabel classifier(clean/toxic/severe toxic/obscene/insult/threat/identity hate) using <strong>PySpark</strong></p>
***Highlights of this project***
<ol>
  <li>Clean and preprocess the dataset</li>
  <ul>
    <li>Convert all the texts to lower cases,removed special characters and digits from each comments</li>
    <li>Adjust the space accordingly and ensure only one white space is found between words.</li>
    <li>Employ the snowball Stemmer algorithm to reduce the inflectional forms of words</li>
    <li>Remove the stop words from the comments</li>
    <li>Given the fact that the dataset is highly imbalanced, we applied the techniques - oversampling and undersampling to obtain a balanced dataset </li>
  </ul>
  <li>Feature Extraction</li>
  <ul>
    <li>TFIDF(A statistical measure that is used to evaluate how important of the features)</li>
    <li>N-gram(N=1,2,3) + TFIDF</li>
  </ul>
  <li>Model Used</li>
  <ul>
    <li>Train the binary classifier and multi-label classifier(one vs Rest) respectively - Logistic Regression, SVM, Naive Bayse,Decision Trees,Gradient boosted tree</li>
    <li>Apply the majority voting based on the predictions made by the 5 models</li>
    <li>Do the above experiment on the ensemble model with different feature technique/with or without undersampling and oversampling</li>
  </ul>
  <li>Conclusion:Our model is able to detect unclean comments with 93% accuracy and is able to categorize unclean comments into the categories hate_speech, toxic, insult and obscene with 80% accuracy.By leveraging Apache Spark, our training framework is designed to scale well with the amount of data provided. Optimal performance for toxic detection occurred when we used unigrams and bigrams instead of higher-ordered n- grams for feature extraction, suggesting that toxic detection is a syntactic problem. To improve the recall of the multilabel classifier, we have tried techniques like undersampling and oversampling. The results show that the two techniques improve the recall significantly but degrades the overall test accuracy. To solve the class imbalance problem without compromising the overall performance of our model, we need to increase the minority class with good samples. One way we could accomplish this is by injecting unclean comments from other datasets into our dataset. Additionally, while the difference between clean and unclean data may be largely syntactic, the difference between a threat and an insult could be semantic in nature. This motivates us to explore other pre-trained word embeddings capable of capturing semantic information (i.e., glove, word2vec, BERT). Most of the research we have surveyed shows that using these pre-trained word embeddings boosts the model performance in text analysis tasks</li>

</ol>
***Assignment***

***Backpropagation***
---
- [Hand-writting Digit](https://github.com/liyishanamy/neuralNetwork-handWrittenDigit)
<p>This is a personal project that deploys backpropagation algorithm to train the model to classify the handwritten digit.After learning the training set.</p>


***Iris Dataset***
- [Iris Dataset-supervised learning](https://github.com/liyishanamy/iris-dataset)
<p>Implementing a perceptron(supervised learning) to classify different types of iris based on the certain properties of the iris using both <strong>Python and tensorflow</strong>. Getting more than 97% accuracy.</p>
---

- [Iris Dataset-unsupervised learning](https://github.com/liyishanamy/iris_LVQ)
<p>Implementing a unsupervised learning algorithm to classify different types of iris based on the certain properties of the iris using <strong>Python</strong>. Getting more than 95% accuracy.</p>

---

- [Iris Dataset-PCA](https://github.com/liyishanamy/iris_LVQ)
<p>Applying <strong>principal component analysis(PCA)</strong> to reduce the dimensionality of the iris dataset to see if a better accuracy could be obtained </p>

---
***KNN***
---
- [Hand-writting Digit](https://github.com/liyishanamy/machineLearningAssignment/tree/main/680%20assignment)
<p>This is a personal project that deploys <strong>KNN algorithm with cross validation method</strong> to classify the handwritten digit.</p>

***Logistic Regression***
---
- [Hand-writting Digit](https://github.com/liyishanamy/machineLearningAssignment/tree/main/680%20assignment)
<p>This is a personal project that deploys <strong>logistic regression algorithm with cross validation method</strong> to classify the handwritten digit.</p>

***Linear regression project***
---
- [Linear Regression](https://github.com/liyishanamy/machineLearningAssignment/tree/main/680%20assignment)
<p>This is a personal project that deploys <strong>linear regression with cross validation method</strong> to classify the data.</p>

***Linear regression project***
---
- [Linear Regression](https://github.com/liyishanamy/machineLearningAssignment/tree/main/680%20assignment)
<p>This is a personal project that deploys <strong>linear regression with cross validation method</strong> to classify the data.</p>

***Gaussian process project***
---
- [Gaussian Process](https://github.com/liyishanamy/machineLearningAssignment/tree/main/680%20assignment)
<p>This is a personal project that deploys <strong>gaussian process with cross validation method</strong> to classify the data.</p>


***Generalized linear regression project***
---
- [Generalized linear regression](https://github.com/liyishanamy/machineLearningAssignment/tree/main/680%20assignment)
<p>This is a personal project that deploys <strong>generalized linear regression with cross validation method</strong> to classify the data.</p>

***CNN***
---
- [CNN](https://github.com/liyishanamy/machineLearningAssignment/tree/main/680%20assignment)
<p>This is a personal project that deploys <strong>deep neural network CNN </strong> to classify the cifar10 image dataset and compare the performance of CNN with different CNN layers,different activation function .</p>

