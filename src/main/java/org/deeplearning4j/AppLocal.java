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
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.util.MLLibUtil;
import org.deeplearning4j.util.SerializationUtils;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultipleEpochsIterator;
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
        //List<String> labels = new ArrayList<>(Arrays.asList("mountain", "rain"));
        final int numRows = 75;
        final int numColumns = 75;
        int nChannels = 3;
        int outputNum = labels.size();
        int batchSize = 1000;
        int iterations = 10;
        int seed = 123;

        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        // Instantiating a RecordReader pointing to the data path with the specified
        // height and width for each image.
        int nIn = numRows * numColumns * nChannels;

        //setup the network
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .batchSize(batchSize)
                .iterations(iterations)
                .constrainGradientToUnitNorm(true)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list(4)
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nOut(2).dropOut(0.5)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(1, new ConvolutionLayer.Builder(3, 3)
                        .nOut(2).dropOut(0.5)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(2, new SubsamplingLayer
                        .Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);

        new ConvolutionLayerSetup(builder,numRows,numColumns,nChannels);
        MultiLayerConfiguration conf = builder.build();


        MultiLayerNetwork trainedNetwork = new MultiLayerNetwork(conf);
        trainedNetwork.init();
        trainedNetwork.setListeners(new ScoreIterationListener(1));


        RecordReader recordReader = new ImageRecordReader(numRows, numColumns,nChannels, true,labels);
        recordReader.initialize(new FileSplit(new File(labeledPath)));
        labels.remove("data");
        // Canova to Dl4j
        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader,60000, nIn,labels.size());
        DataSet next2 = iter.next();
        next2.normalizeZeroMeanZeroUnitVariance();
        System.out.println("Training on data set of " + next2.numExamples() + " with num features " + next2.getFeatureMatrix().size(1) + " and labels " + next2.getLabels().size(1));
        SplitTestAndTrain testAndTrain = next2.splitTestAndTrain(0.8);


        DataSetIterator trainIterator = new ListDataSetIterator(testAndTrain.getTrain().asList(),10);
        while(trainIterator.hasNext()) {
            DataSet next = trainIterator.next();
            System.out.println("Training on one batch");
            trainedNetwork.fit(next);
            System.out.println("One batch done with score " + trainedNetwork.score());
        }


        Evaluation evaluation = new Evaluation(labels.size());
        evaluation.eval(testAndTrain.getTest().getLabels(),trainedNetwork.output(testAndTrain.getTest().getFeatureMatrix(), true));
        System.out.println(evaluation.stats());
     /* //  SplitTestAndTrain testAndTrain = next.splitTestAndTrain(0.6);
        DataSetIterator trainIterator = new MultipleEpochsIterator(5,new ListDataSetIterator(testAndTrain.getTrain().asList(),100));
        while(trainIterator.hasNext())
            trainedNetwork.fit(trainIterator.next());
        Evaluation evaluation = new Evaluation(labels.size());
        evaluation.eval(testAndTrain.getTest().getLabels(),trainedNetwork.output(testAndTrain.getTest().getFeatureMatrix(), true));
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream("newimagemodel.bin"));
        Nd4j.write(bos,trainedNetwork.params());
        FileUtils.write(new File("conf.yaml"),trainedNetwork.conf().toYaml());
        System.out.println(evaluation.stats());*/


    }
}
