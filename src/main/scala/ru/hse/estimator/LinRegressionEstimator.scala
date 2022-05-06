package ru.hse.estimator

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.StructType
import ru.hse.model.LinRegressionModel
import ru.hse.params.LinRegressionParams


class LinRegressionEstimator(override val uid: String) extends Estimator[LinRegressionModel]
  with DefaultParamsWritable
  with LinRegressionParams {

  override def copy(extra: ParamMap): Estimator[LinRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = schema

  override def fit(dataset: Dataset[_]): LinRegressionModel = {
    setInputCols(Array("features"))
    setWeights(DenseVector.zeros(3))
    setBias(0)

    import ru.hse.Main.spark.implicits._

    val data = dataset.select(dataset("x1"), dataset("x2"), dataset("x3")).map(row => Array(row.getDouble(0), row.getDouble(1), row.getDouble(2))).collect()
    val X = DenseMatrix(data:_*)

    val y = DenseVector(dataset.select(dataset("y")).map(_.getDouble(0)).collect())

    val lambda = 1e-2

    for (_ <- 0 to 10000) {
      val delta = X * getWeights + DenseVector.fill(X.rows) { getBias } - y
      setWeights(getWeights - (lambda / X.rows) * (delta.t * X).t)
      setBias(getBias - lambda / X.rows * sum(delta))
    }

    copyValues(new LinRegressionModel(uid + "_model").setParent(this))
  }
}

object LinRegressionEstimator extends DefaultParamsReadable[LinRegressionEstimator]
