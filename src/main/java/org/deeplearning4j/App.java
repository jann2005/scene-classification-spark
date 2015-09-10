package org.deeplearning4j;

import org.apache.commons.io.FileUtils;
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
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.util.MLLibUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import scala.Tuple2;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.util.Arrays;
import java.util.List;

/**
 * Hello world!
 *
 */
public class App {
    public static void main( String[] args) throws Exception {
        List<String> labels = Arrays.asList("beach", "desert", "forest", "mountain", "rain", "snow");

        final int numRows = 28;
        final int numColumns = 28;
        int nChannels = 3;
        int outputNum = labels.size();
        int batchSize = 500;
        int iterations = 10;
        int seed = 123;

        JavaSparkContext sc = new JavaSparkContext(new SparkConf().setMaster("local[*]").setAppName("scenes"));
        //load the images from the bucket setting the size to 28 x 28
        String s3Bucket = "s3n://scenesdata/data/*";
        //normalize the data to zero mean and unit variance
        JavaRDD<LabeledPoint> data = MLLibUtil.fromBinary(sc.binaryFiles(s3Bucket)
                , new ImageRecordReader(28,28,labels));
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

        //setup the network
        //setup the network
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .batchSize(batchSize)
                .iterations(iterations).regularization(true)
                .l2(2e-4).l1(0.1)
                .constrainGradientToUnitNorm(true)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list(3)
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(nChannels)
                        .nOut(6).dropOut(0.5)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(1, new SubsamplingLayer
                        .Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(216)
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())
                .inputPreProcessor(0, new FeedForwardToCnnPreProcessor(numRows, numColumns, nChannels))
                .inputPreProcessor(2, new CnnToFeedForwardPreProcessor())
                .backprop(true).pretrain(false)
                .build();
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
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(""));
        Nd4j.write(bos,trainedNetwork.params());
        FileUtils.write(new File("conf.yaml"),trainedNetwork.conf().toYaml());
        System.out.println("F1 = " + precision);


    }
}
