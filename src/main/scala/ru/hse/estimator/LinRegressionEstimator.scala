package ru.hse.estimator

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.StructType
import ru.hse.model.LinRegressionModel
import ru.hse.params.LinRegressionParams
import ru.hse.Main.spark
import spark.implicits._
import org.apache.spark.sql.functions.col


class LinRegressionEstimator(override val uid: String) extends Estimator[LinRegressionModel]
  with DefaultParamsWritable
  with LinRegressionParams {

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  def setLabelCol(value: String): this.type = set(labelCol, value)

  override def copy(extra: ParamMap): Estimator[LinRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = schema

  override def fit(dataset: Dataset[_]): LinRegressionModel = {
    setInputCols(dataset.columns.filter(s => s != getLabelCol))
    setWeights(DenseVector.zeros(getInputCols.length))
    setBias(0)

    val y = DenseVector(dataset.select(dataset(getLabelCol)).map(_.getDouble(0)).collect())
    val data = dataset.drop(getLabelCol)
      .select(getInputCols.map(m => col(m)): _*)
      .map(row => row.toSeq.asInstanceOf[Seq[Double]].toArray).collect()
    val X = DenseMatrix(data: _*)

    for (_ <- 0 to getMaxIter) {
      val delta = X * getWeights + getBias - y
      setWeights(getWeights - (getLambda / X.rows) * (delta.t * X).t)
      setBias(getBias - getLambda / X.rows * sum(delta))
    }

    copyValues(new LinRegressionModel(uid + "_model").setParent(this))
  }
}

object LinRegressionEstimator extends DefaultParamsReadable[LinRegressionEstimator]
