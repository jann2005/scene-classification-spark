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
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.util.MLLibUtil;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import scala.Tuple2;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Hello world!
 *
 */
public class AppLocal {
    public static void main( String[] args ) throws Exception {
        // Path to the labeled images
        String labeledPath = System.getProperty("user.home")+"/data";
        List<String> labels = new ArrayList<>(Arrays.asList("beach", "desert", "forest", "mountain", "rain", "snow"));

        // Instantiating a RecordReader pointing to the data path with the specified
        // height and width for each image.
        int height = 75;
        int width = 75;
        int nIn = height * width;
        RecordReader recordReader = new ImageRecordReader(height, width, true,labels);
        recordReader.initialize(new FileSplit(new File(labeledPath)));

        // Canova to Dl4j
        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader,60000, nIn,labels.size());



        //setup the network
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().l2(2e-4).learningRate(1e-3)
                .l1(1e-1).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .regularization(true).miniBatch(true).constrainGradientToUnitNorm(true)
                .list(4).backprop(true).pretrain(false)
                .layer(0,new DenseLayer.Builder().nIn(nIn).nOut(5000).activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(5000).nOut(4000).activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(2, new DenseLayer.Builder().nIn(4000).nOut(3000).activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(3,new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation("softmax")
                        .nIn(3000).nOut(labels.size()).weightInit(WeightInit.XAVIER)
                        .build()).build();

        MultiLayerNetwork trainedNetwork = new MultiLayerNetwork(conf);
        trainedNetwork.setListeners(new ScoreIterationListener(1));
        DataSet next = iter.next();
        next.scale();
        next.shuffle();
        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(0.6);
        DataSetIterator trainIterator = new ListDataSetIterator(testAndTrain.getTrain().asList(),100);
        while(trainIterator.hasNext())
            trainedNetwork.fit(trainIterator.next());
        Evaluation evaluation = new Evaluation(labels.size());
        evaluation.eval(testAndTrain.getTest().getLabels(),trainedNetwork.output(testAndTrain.getTest().getFeatureMatrix(), true));
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream("newimagemodel.bin"));
        Nd4j.write(bos,trainedNetwork.params());
        FileUtils.write(new File("conf.yaml"),trainedNetwork.conf().toYaml());
        System.out.println(evaluation.stats());


    }
}
