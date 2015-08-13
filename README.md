#!/bin/bash

# Building nd4j
cd  ../nd4j
git pull origin master
mvn clean install -DskipTests

# Building dl4j
cd  ../deeplearning4j
git pull origin master
mvn clean install -DskipTests

cd ../scene-classification-spark
git pull origin master
mvn clean install -DskipTests

sh /root/spark/bin/spark-submit \
--driver-memory 25G \
--class org.deeplearning4j.App.java \
--executor-memory 25G \
--master spark://ec2-54-148-164-15.us-west-2.compute.amazonaws.com:7077 \
target/topicmodeling-1.0-SNAPSHOT.jar
