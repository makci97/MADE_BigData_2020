package org.apache.spark.ml.made

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasLabelCol, HasMaxIter, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsReader, DefaultParamsWritable, DefaultParamsWriter, Identifiable, MLReadable, MLReader, MLWritable, MLWriter, SchemaUtils}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}
import org.apache.spark.sql.types.{DoubleType, StructType}

import scala.util.control.Breaks.{break, breakable}

trait LinearRegressionParams extends HasInputCol with HasLabelCol with HasOutputCol with HasMaxIter {
  def setInputCol(value: String) : this.type = set(inputCol, value)
  def setLabelCol(value: String) : this.type = set(labelCol, value)
  def setOutputCol(value: String) : this.type = set(outputCol, value)
  def setMaxIter(value: Int) : this.type = set(maxIter, value)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())
    SchemaUtils.checkColumnType(schema, getLabelCol, DoubleType)

    if (schema.fieldNames.contains($(outputCol))) {
      if ($(outputCol) != $(inputCol)) {
        SchemaUtils.checkColumnType(schema, getOutputCol, DoubleType)
      }
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getLabelCol).copy(name = getOutputCol))
    }
  }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams
with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("LinearRegression"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    // used to convert untyped dataframes to datasets with vectors
    implicit val encoder: Encoder[Vector] = ExpressionEncoder()
    implicit val dEncoder: Encoder[Double] = ExpressionEncoder()

    val vectors: Dataset[(Vector, Double)] = dataset.select(dataset($(inputCol)).as[Vector], dataset($(labelCol)).as[Double])
    val dim: Int = AttributeGroup.fromStructField((dataset.schema($(inputCol)))).numAttributes.getOrElse(
      vectors.first()._1.size
    )
    val size = dataset.count()

    // init coefficients
    implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(0)))
    val Gausian = breeze.stats.distributions.Gaussian(0.0, 0.1)
    var coefficients = breeze.linalg.DenseVector.rand(dim, Gausian)
    var intercept: Double = 0.0

    breakable {
      for (i <- 0 to $(maxIter)) {
        val sum_b_w_error = vectors.rdd.map((features_y) => {
          val features = features_y._1.asBreeze
          val y = features_y._2
          val error = sum(features * coefficients) + intercept - y
          (error, error * features)
        }).reduce((a, b) => {
          (a._1 + b._1, a._2 + b._2)
        })

        val sum_intercept_error = sum_b_w_error._1
        val sum_coefficients_error = sum_b_w_error._2

        val grad_norm = sum(abs(sum_coefficients_error)) + abs(sum_intercept_error)
        if (grad_norm / size < 0.00001){
          break
        }

        intercept -= 1.0 / size * sum_intercept_error
        coefficients -= 1.0 / size * sum_coefficients_error
      }
    }

    copyValues(new LinearRegressionModel(Vectors.fromBreeze(coefficients).toDense, intercept)).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](
                           override val uid: String,
                           val coefficients: DenseVector,
                           val intercept: Double) extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {


//  private[made] def this(uid: String) =
//    this(uid, Vectors.zeros(0).toDense, Vectors.zeros(0).toDense)

  private[made] def this(coefficients: Vector, intercept: Double) =
    this(Identifiable.randomUID("LinearRegressionModel"), coefficients.toDense, intercept)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(new LinearRegressionModel(coefficients, intercept))

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = dataset.sqlContext.udf.register(uid + "_transform",
      (x: Vector) => {
        sum(x.asBreeze * coefficients.asBreeze) + intercept
      })
    dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)


      val data = coefficients.asInstanceOf[Vector] -> intercept.asInstanceOf[Double]

      sqlContext.createDataFrame(Seq(data)).write.parquet(path + "/data")
      }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/data")

      // Used to convert untyped dataframes to datasets with vectors
      implicit val encoder : Encoder[Vector] = ExpressionEncoder()
      implicit val dEncoder : Encoder[Double] = ExpressionEncoder()

      val (coefficients, intercept) = vectors.select(vectors("_1").as[Vector], vectors("_2").as[Double]).first()

      val model = new LinearRegressionModel(coefficients, intercept)
      metadata.getAndSetParams(model)
      model
    }
  }
}