package org.apache.spark.ml.made

import breeze.linalg.randomDouble._zero
import breeze.linalg.{*, DenseMatrix, DenseVector}
import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import com.google.common.io.Files
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.ml
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.made.{LinearRegressionModel, WithSpark}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.scalatest._
import org.scalatest.flatspec._
import org.scalatest.matchers._


class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 0.0001
  val maxIter = 1000
  lazy val coefficients: Vector = LinearRegressionTest._coefficients
  lazy val intercept: Double = LinearRegressionTest._intercept
  lazy val y: DenseVector[Double] = LinearRegressionTest._y
  lazy val df: DataFrame = LinearRegressionTest._df


  "Model" should "transform input data to prediction" in {
    val model = new LinearRegressionModel(
      coefficients = coefficients.toDense,
      intercept = intercept
    ).setInputCol("features").setOutputCol("features").setMaxIter(maxIter)
    val vectors: Array[Vector] = model.transform(df).collect().map(_.getAs[Vector](0))

    vectors.length should be(1000)
    for (i <- 0 to vectors.length) {
      vectors(i)(0) should be(y.valueAt(i) +- delta)
    }
  }
}

object LinearRegressionTest extends WithSpark{
  import spark.implicits._

  implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(0)))
  private val Gausian = breeze.stats.distributions.Gaussian(0.0, 1.0)
  val X: DenseMatrix[Double] = DenseMatrix.rand(1000, 3, Gausian)
  val _coefficients: Vector = Vectors.dense(0.5, -0.1, 0.2).toDense //DenseVector(0.5, -0.1, 0.2)
  val _intercept: Double = 1.2
  val _y: DenseVector[Double] = X * _coefficients.asBreeze + _intercept
  val data = DenseMatrix.horzcat(X, _y.asDenseMatrix.t)
  val _df = data(*, ::).iterator
    .map(x => (x(0), x(1), x(2), x(3)))
    .toSeq.toDF("x1", "x2", "x3", "y")
}

