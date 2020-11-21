package org.apache.spark.ml.made

import java.text.AttributedCharacterIterator.Attribute

import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsReader, DefaultParamsWritable, DefaultParamsWriter, Identifiable, MLReadable, MLReader, MLWritable, MLWriter, SchemaUtils}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row}
import org.apache.spark.sql.types.StructType

trait LinearRegressionParams extends HasInputCol with HasOutputCol {
  def setInputCol(value: String) : this.type = set(inputCol, value)
  def setOutputCol(value: String) : this.type = set(outputCol, value)

//  protected def validateAndTransformSchema(schema: StructType): StructType = {
//    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())
//
//    if (schema.fieldNames.contains($(outputCol))) {
//      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
//      schema
//    } else {
//      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
//    }
//  }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams
with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("LinearRegression"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = ??? // {
//    // used to convert untyped dataframes to datasets with vectors
//    implicit val encoder: Encoder[Vector] = ExpressionEncoder()
//
//    val vectors: Dataset[Vector] = dataset.select(dataset($(inputCol)).as[Vector])
//
////    val dim: Int = AttributeGroup.fromStructField((dataset.schema($(inputCol)))).numAttributes.getOrElse(
////      vectors.first().size
////    )
//
//    val summary = vectors.rdd.mapPartitions((data: Iterator[Vector]) => {
//      val summarizer = new MultivariateOnlineSummarizer()
//      data.foreach(v => summarizer.add(mllib.linalg.Vectors.fromBreeze(v.asBreeze)))
//      Iterator(summarizer)
//    }).reduce(_ merge _)
//
//    copyValues(new LinearRegressionModel(
//      summary.mean.asML.toDense,
//      Vectors.fromBreeze(breeze.numerics.sqrt(summary.variance.asBreeze)).toDense
//    )).setParent(this)
//
//
////    val Row(row: Row) = dataset
////      .select(Summarizer.metrics("mean", "std").summary(dataset($(inputCol))))
////      .first()
////
////    copyValues(new LinearRegressionModel(row.getAs[Vector](0).toDense, row.getAs[Vector](1).toDense)).setParent(this)
//  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = ??? // defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = ??? // validateAndTransformSchema(schema)

}

//object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](
                           override val uid: String,
                           val means: DenseVector,
                           val stds: DenseVector) extends Model[LinearRegressionModel] with LinearRegressionParams { // with MLWritable {


//  private[made] def this(uid: String) =
//    this(uid, Vectors.zeros(0).toDense, Vectors.zeros(0).toDense)

//  private[made] def this(means: DenseVector, stds: DenseVector) =
//    this(Identifiable.randomUID("LinearRegressionModel"), means, stds)

  override def copy(extra: ParamMap): LinearRegressionModel = ??? // copyValues(new LinearRegressionModel(means, stds))

  override def transform(dataset: Dataset[_]): DataFrame = ??? //{
//    val transformUdf = dataset.sqlContext.udf.register(uid + "_transform",
//      (x: Vector) => {
//        Vectors.fromBreeze((x.asBreeze - means.asBreeze) /:/ stds.asBreeze)
//      })
//    dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))))
//  }

  override def transformSchema(schema: StructType): StructType = ??? // validateAndTransformSchema(schema)

//  override def write: MLWriter = new DefaultParamsWriter(this) {
//    override protected def saveImpl(path: String): Unit = {
//      super.saveImpl(path)
//
//      sqlContext.createDataFrame(Seq(means.toDense -> stds.toDense)).write.json(path + "/vectors")
//      }
//  }
}

//object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
//  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
//    override def load(path: String): LinearRegressionModel = {
////      val instance: LinearRegressionModel = DefaultParamsReader.loadParamsInstance[LinearRegressionModel](path, sc)
//
//      val vectors = sqlContext.read.json(path + "/vectors")
//
//      // used to convert untyped dataframes to datasets with vectors
//      implicit val encoder: Encoder[DenseVector] = ExpressionEncoder()
//
//      val (means, stds) = vectors.select(vectors("_1").as[DenseVector], vectors("_2").as[DenseVector]).first()
//
////      instance.copyValues(new LinearRegressionModel(means.toDense, stds.toDense))
//      val model = new LinearRegressionModel(means.toDense, stds.toDense)
//      model
//    }
//  }
//}