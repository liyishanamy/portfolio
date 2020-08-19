## About Me
<p>My name is Yishan Li. My friends always call me Amy. I am from Chengdu, Sichuan, China. I did my undergrad in Queen's University and I am doing my master degree(data science and artificial intelligence) in University of Waterloo.</p>
---
## Portfolio

---

### Software Development

**Health Track[NoSQL database project]**
<p>This project is called the health track. This platform is designed to help doctors to keep track of the patients who are tested positive of COVID19.  Patients can use this platform to report their daily health status (including today's symptoms, today's temperature, whether they have gone somewhere. if they have, whether they wear a facial mask). The doctor can be able to manage all his patients' health status(data, bar char, line graph etc) so that they can view the visual data more clearly and they can also care for the patients more efficiently. Only after 14 days without fever and no symptoms can they make an appointment with doctor. Doctor can also manage all the appointments made by patients, they can also update the test result via this platform.  </p>

---
- [Front End](https://github.com/liyishanamy/healthtrackingplatform_frontend)
- [Back End](https://github.com/liyishanamy/healthInformation_backend)
- [API](https://docs.google.com/document/d/15rplKs8JGBj297dSzQFhMPIbVfc3abBbPUhmvtrWwfU/edit)

***Highlights of this project***
<ol>
  <li>Use <strong>Mongodb</strong> to store all the data</li>
  <li>Use <strong>Express</strong> to make API service</li>
  <ol>
    <li>Applying both <strong>access token and basic authentication</strong> method to request access to the API</li>
    <li>Implementing <strong>Authentication+Access Token+ Refresh Token</strong> using <strong>JWT</strong></li>
    <li>Applying <strong>image hash</strong> mechanism to save the users' profile pictures to stregthen the security </li>
    <li>Using <strong>hash/salt</strong> to store users password.</li>
    <li>Implementing <strong>rate limit</strong> to restrict the use of some API(ex, Users can only change password once a day)
    <li>Applying <strong>Pagination</strong> to the some of the API(which return large amount of data)</li>
    <li>Implementing Chatting System using <strong>socket io</strong></li>
     
  </ol>
  <li>Use <strong>Reactjs</strong> to make frontend</li>
  <ol>
    <li>Applying invitation code mechanism to cluster patients</li>
    <li>Using <strong>regex</strong> to check the whether the password is strong enough</li>
    <li>Displaying Data visualization(Visualization of the patients' data)+ Export the Data into CSV</li>
    <li>Using <strong>Redux</strong> to manage the software global state</li>
    <li>Chat system support both groub</li>
    <li>Performing quality assurance after implementation and taking notes of the API documentation</li>
    
  </ol>
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
</ol>

---
[Eat What](http://example.com/)
<img src="images/dummy_thumbnail.jpg?raw=true"/>

---

### Data Science

- [Music Genre classification](https://github.com/astralcai/music-genre-classification) - [Report] (https://github.com/liyishanamy/music-genre-classification_report)
<p>Worked with my team to deploy different deep learning models to classify the different music genres using TensorFlow.</p>

***Highlights of this project***
<ol>
  <li>Selecting <strong>GTZAN and benchmark</strong> pre-existing dataset to train the model</li>
  <li>Apply <strong>short-time Fourier transform</strong> to conduct data preprocessing. Specifically speaking, to tranform the invisible music data to visible spectrogram</li>
  <li>Deploying <strong>Convolutional Neural Network</strong> and train the model to learn different music genre spectrogram.(Predicted the testing set correctly with 82% accuracy after training and optimizing our models)</li>
  <li>We chose pre-trained network(VGG-16) as our second model.With the replacement of the original pre-trained fully connected layer and softmax classifier on top of the convolutional layers with our own fully connected hidden layer, and a softmax output layer of 5 nodes(5 possible music genre). We found out the VGG 16 model does not work well upon the spectrogram </li>
  <li>Deploying Recurrent Neural Network and train the model to learn different music genre spectrogram.(Predicted the testing set correctly with 84% accuracy after training. To make an optimization of the RNN model, we explored another approach--RNN+attention which further enhanced the prediction result(Accuracy reaches 88% upon the unseen testing set) </li>
  <li>We also applied several approaches to prevent overfitting</li>
  <ol>
    <li>To reduce the number of parameters if possible. To decrease the complexity of the model, we chose to only use the output of the final BGRU layer, which improved the classification accuracy by around 2%.</li>
    <li>Another approach is to use validation data during training. The model was set to stop training and revert to the weights which produced the best results if the validation accuracy does not improve for 3 consecutive epochs</li>
    <li>We also added dropouts to the classifier, where some neurons are ”shut off” for every iteration. This prevents the network from depending too much on the outputs of a small subset of neurons.</li>
  </ol>
  <li>Writing the final report(paper) to summarize the project(https://github.com/liyishanamy/music-genre-classification_report)</li>
</ol>

(VGG16, convolutional neural network, recurrent neural network) 
Predicted the music genre correctly with 82% accuracy after training and optimizing our models using GTZAN and benchmark dataset.</p>
- [Iris Dataset-supervised learning](https://github.com/liyishanamy/iris-dataset)
- [Hand-writting Digit](https://github.com/liyishanamy/neuralNetwork-handWrittenDigit)
- [Iris Dataset-unsupervised learning](https://github.com/liyishanamy/iris_LVQ)
- [Iris Dataset-PCA](https://github.com/liyishanamy/iris_LVQ)

---




---
<p style="font-size:11px">Page template forked from <a href="https://github.com/evanca/quick-portfolio">evanca</a></p>
<!-- Remove above link if you don't want to attibute -->
