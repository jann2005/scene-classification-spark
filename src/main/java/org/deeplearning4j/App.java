package org.deeplearning4j;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.feature.StandardScaler;
import org.apache.spark.mllib.feature.StandardScalerModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.util.MLLibUtil;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import scala.Tuple2;

import java.util.Arrays;
import java.util.List;

/**
 * Hello world!
 *
 */
public class App {
    public static void main( String[] args ) {
        JavaSparkContext sc = new JavaSparkContext(new SparkConf().setMaster("local[*]").setAppName("scenes"));
        //load the images from the bucket setting the size to 28 x 28
        String s3Bucket = "s3n://scenesdata/data/*";
        List<String> labels = Arrays.asList("beach", "desert", "forest", "mountain", "rain", "snow");
        int nIn = 784;
        //normalize the data to zero mean and unit variance
        JavaRDD<LabeledPoint> data = MLLibUtil.fromBinary(sc.binaryFiles(s3Bucket), new ImageRecordReader(28,28,labels));
        StandardScaler scaler = new StandardScaler(true,true);

        final StandardScalerModel scalarModel = scaler.fit(data.map(new Function<LabeledPoint, Vector>() {
            @Override
            public Vector call(LabeledPoint v1) throws Exception {
                return v1.features();
            }
        }).rdd());

        //get the trained data for the train/test split
        JavaRDD<LabeledPoint> normalizedData = data.map(new Function<LabeledPoint, LabeledPoint>() {
            @Override
            public LabeledPoint call(LabeledPoint v1) throws Exception {
                Vector features = v1.features();
                Vector normalized = scalarModel.transform(features);
                return new LabeledPoint(v1.label(), normalized);
            }
        }).cache();

        //train test split 60/40
        JavaRDD<LabeledPoint>[] trainTestSplit = normalizedData.randomSplit(new double[]{60, 40});



        //setup the network
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list(4).backprop(true)
                .layer(0,new DenseLayer.Builder().nIn(nIn).nOut(600).activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(600).nOut(500).activation("relu")
                        .weightInit(WeightInit.XAVIER).dropOut(0.6)
                        .build())
                .layer(2, new DenseLayer.Builder().nIn(500).nOut(250).activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(3,new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation("softmax")
                        .nIn(250).nOut(labels.size()).weightInit(WeightInit.XAVIER)
                        .build()).build();
       //train the network
        SparkDl4jMultiLayer trainLayer = new SparkDl4jMultiLayer(sc.sc(),conf);
        //fit on the training set
        MultiLayerNetwork trainedNetwork = trainLayer.fit(sc, trainTestSplit[0]);
        final SparkDl4jMultiLayer trainedNetworkWrapper = new SparkDl4jMultiLayer(sc.sc(),trainedNetwork);

        // Compute raw scores on the test set.
        JavaRDD<Tuple2<Object, Object>> predictionAndLabels = trainTestSplit[1].map(
                new Function<LabeledPoint, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(LabeledPoint p) {
                        Vector prediction = trainedNetworkWrapper.predict(p.features());
                        return new Tuple2<Object, Object>(prediction, p.label());
                    }
                }
        );




        // Get evaluation metrics.
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
        double precision = metrics.fMeasure();
        System.out.println("F1 = " + precision);


    }
}
