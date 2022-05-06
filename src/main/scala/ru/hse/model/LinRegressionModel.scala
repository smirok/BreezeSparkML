package ru.hse.model

import breeze.linalg.{*, DenseMatrix, DenseVector}
import org.apache.spark.ml.Model
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}
import ru.hse.params.LinRegressionParams


class LinRegressionModel(override val uid: String) extends Model[LinRegressionModel]
  with DefaultParamsWritable
  with LinRegressionParams {

  override def copy(extra: ParamMap): LinRegressionModel = defaultCopy(extra)
  override def transformSchema(schema: StructType): StructType = schema
  override def transform(dataset: Dataset[_]): DataFrame = {

    import ru.hse.Main.spark.implicits._

    val data = dataset.select(dataset("x1"), dataset("x2"), dataset("x3")).map(row => Array(row.getDouble(0), row.getDouble(1), row.getDouble(2))).collect()
    val X = DenseMatrix(data:_*)

    val predictions = X * getWeights + DenseVector.fill(X.rows) { getBias }

    predictions.toArray.toSeq.toDF("y")
  }
}

object LinRegressionModel extends DefaultParamsReadable[LinRegressionModel]