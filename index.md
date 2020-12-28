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

***Data-intensive Distributed Computing(the code cannot be posted until the end of term)***
<ol>
  <li>Counting the word in a long article and calculating the PMI between words using <strong>Mapreduce -Java</strong> with algorithm <strong>Pairs and Stripes</strong></li>
  <li>Counting the word in a long article and calculating the PMI/Bigrams between words using <strong>Spark -Scala</strong> with algorithm <strong>Pairs and Stripes</strong></li>
  <li>Optimizing the inverted Index project(search) by implementing index compression,buffering posting, term partitioning using <strong>Mapreduce-java</strong>.  </li>
</ol>

---

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




---

***Hand-writting Digit***

- [Hand-writting Digit](https://github.com/liyishanamy/neuralNetwork-handWrittenDigit)
<p>This is a personal project that deploys backpropagation algorithm to train the model to classify the handwritten digit.</p>

***Highlights of this project***
<ol>
  <li>Using <strong>MNIST database</strong> of handwriten digit as the data,which contains 60000 hand-writen digits</li>
  <li>Implementing the <strong>backpropagation algorithm</strong> using <strong>Python</strong> to let the model learn from the training set, and update the weight accordingly based on the error</li>
  <li>After learning the training set, the model can correctly classify the testing set(handwriten digit) with mor than 90% accuracy </li>
</ol>

- [Hand-writting Digit]
<p>This is a personal project that deploys <strong>KNN algorithm with cross validation method</strong> to classify the handwritten digit.</p>

---

- [Hand-writting Digit]
<p>This is a personal project that deploys <strong>logistic regression algorithm with cross validation method</strong> to classify the handwritten digit.</p>

---

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

***Linear regression project***
[data set]
<p>This is a personal project that deploys <strong>linear regression with cross validation method</strong> to classify the data.</p>

