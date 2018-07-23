############### PerCoordinateUpdater for ftrl ###############


package org.apache.spark.ml.optim

import breeze.linalg.{DenseVector, sum}
import breeze.numerics._
import org.apache.spark.mllib.linalg.{Vector, Vectors}

/**
  * :: DeveloperApi ::
  * Class used to perform steps (weight update) with per-coordinate learning rate.
  *
  */
abstract class PerCoordinateUpdater extends Serializable {
  def compute(
      features:Vector,
      label:Double,
      weightsOld: Vector,
      gradient: Vector,
      alpha: Double,
      beta: Double,
      l1: Double,
      l2: Double,
      n: Vector,
      z: Vector): (Vector, Double, Vector, Vector) = {

    val dimension  = weightsOld.size
//    val weightsNew = Vectors.zeros(dimension)
    val weightsNew = weightsOld
    for (i <- 0 until dimension){
      var wti:Double = 0
      if (abs(z.apply(i)> l1)){
        wti = (signum(z.apply(i))* l1 -z.apply(i)) / (l2 + (beta + sqrt(n.apply(i))) / alpha)
      }
      else {
        weightsNew(i) = wti
        }
    }

    val pt:Double = lr(weightsNew.asBreeze.toDenseVector, features.asBreeze.toDenseVector)
    val g:DenseVector[Double] = grad(label, pt, features.asBreeze.toDenseVector)
    val sigma:DenseVector[Double] = (sqrt(n.asBreeze.toDenseVector + g * g) - sqrt(n)) / alpha


    val zc:DenseVector[Double] = z.asBreeze.toDenseVector
    val zd = zc + g - sigma * weightsNew
    val g_square = g * g
    val nn = n.asBreeze.toDenseVector + g_square

    val loss= crossLoss(label, pt)

    (weightsNew, loss, Vectors.fromBreeze(nn),Vectors.fromBreeze(zd))
  }

  /**
    * 逻辑回归，计算得到样本的结果
    * @param w
    * @param x
    * @return
    */
  def lr(w: DenseVector[Double], x: DenseVector[Double]): Double ={
    val sigmoid = 1.0 / (1.0 + exp(-w.t * x))
    return sigmoid
  }

  /**
    * 交叉熵损失函数
    */
  def crossLoss(y: Double, y_hat: Double): Double ={

    return sum(DenseVector[Double](-y * log(y_hat) - (1 - y) * log(1 - y_hat)))
  }

  /**
    * 交叉熵损失函数对权重w的一阶导数
    * @param yt 原始标签
    * @param pt 预测结果
    * @param xt 输入样本
    * @return  梯度
    */
  def grad(yt: Double, pt: Double, xt: DenseVector[Double]): DenseVector[Double] ={
    return (pt - yt) * xt
  }

}
