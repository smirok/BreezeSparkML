package ru.hse.estimator

import math.sqrt
import breeze.linalg.{DenseMatrix, DenseVector, squaredDistance, sum}
import org.apache.spark.ml.linalg.{Vector => SparkVector}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable}
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.{Dataset, Encoder}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.StructType
import ru.hse.model.LinRegressionModel
import ru.hse.params.LinRegressionParams
import scala.util.control.Breaks.{break, breakable}


class LinRegressionEstimator(override val uid: String) extends Estimator[LinRegressionModel]
  with DefaultParamsWritable
  with LinRegressionParams {

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  def setLabelCol(value: String): this.type = set(labelCol, value)

  def setTol(value: Double): this.type = set(tol, value)

  def setLearningRate(value: Double): this.type = set(learningRate, value)

  def setBatchSize(value: Int): this.type = set(batchSize, value)

  override def copy(extra: ParamMap): Estimator[LinRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = schema

  override def fit(dataset: Dataset[_]): LinRegressionModel = {
    implicit val encoder: Encoder[SparkVector] = ExpressionEncoder()

    setInputCols(dataset.columns.filter(s => s != getLabelCol))
    setWeights(DenseVector.zeros(getInputCols.length))
    setBias(0)

    val assembler = new VectorAssembler().setInputCols(getInputCols ++ Array(getLabelCol)).setOutputCol("data")
    val dataVectors: Dataset[SparkVector] = assembler.transform(dataset).select("data").as[SparkVector]

    breakable {
      for (_ <- 0 to getMaxIter) {
        val preIterationWeights = getWeights

        val summary = dataVectors.rdd.mapPartitions((data: Iterator[SparkVector]) => {
          val weightSummarizer = new MultivariateOnlineSummarizer()
          val biasSummarizer = new MultivariateOnlineSummarizer()

          data.grouped(getBatchSize).foreach(block => {
            val xTrain = DenseMatrix(block.toArray.map(_.toArray.dropRight(1)): _*)
            val yTrain = DenseVector(block.toArray.map(_.toArray.last))

            val delta = xTrain * getWeights + getBias - yTrain
            val weightsDelta = (delta.t * xTrain).t / block.size.asInstanceOf[Double]
            val biasDelta = sum(delta) / block.size.asInstanceOf[Double]

            weightSummarizer.add(Vectors.dense(weightsDelta.toArray))
            biasSummarizer.add(Vectors.dense(Array(biasDelta)))
          })

          Iterator((weightSummarizer, biasSummarizer))
        }).reduce((x, y) => (x._1.merge(x._2), y._1.merge(y._2)))

        val weightsGrad = summary._1
        val biasGrad = summary._2

        setWeights(getWeights - getLearningRate * DenseVector(weightsGrad.mean.toArray))
        setBias(getBias - getLearningRate * biasGrad.mean.toArray(0))

        if (sqrt(squaredDistance(getWeights, preIterationWeights)) < getTol)
          break
      }
    }

    copyValues(new LinRegressionModel(uid + "_model").setParent(this))
  }
}

object LinRegressionEstimator extends DefaultParamsReadable[LinRegressionEstimator]
