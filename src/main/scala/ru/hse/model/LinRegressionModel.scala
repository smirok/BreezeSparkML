package ru.hse.model

import breeze.linalg.DenseMatrix
import org.apache.spark.ml.Model
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}
import ru.hse.params.LinRegressionParams
import ru.hse.Main.spark
import spark.implicits._

class LinRegressionModel(override val uid: String)
    extends Model[LinRegressionModel]
    with DefaultParamsWritable
    with LinRegressionParams {

  override def copy(extra: ParamMap): LinRegressionModel = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = schema

  override def transform(dataset: Dataset[_]): DataFrame = {

    val data = dataset
      .select(getInputCols.map(m => col(m)): _*)
      .map(row => row.toSeq.asInstanceOf[Seq[Double]].toArray)
      .collect()
    val X = DenseMatrix(data: _*)

    val predictions = X * getWeights + getBias

    predictions.toArray.toSeq.toDF(getLabelCol)
  }
}

object LinRegressionModel extends DefaultParamsReadable[LinRegressionModel]
