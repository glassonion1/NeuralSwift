//
//  Operators.swift
//  SimpleNN
//
//  Created by taisuke fujita on 2017/07/19.
//  Copyright © 2017年 taisuke fujita. All rights reserved.
//

import Accelerate

func +(lhs: Vector, rhs: Vector) -> Vector {
    return lhs.add(rhs)
}

func +(lhs: Matrix, rhs: Double) -> Matrix {
    return lhs.add(rhs)
}

func +(lhs: Matrix, rhs: Matrix) -> Matrix {
    return lhs.add(rhs)
}

func -(lhs: Double, rhs: Vector) -> Vector {
    let vector = Vector(value: lhs, rows: rhs.rows)
    return vector.difference(rhs)
}

func -(lhs: Vector, rhs: Vector) -> Vector {
    return lhs.difference(rhs)
}

func -(lhs: Matrix, rhs: Double) -> Matrix {
    return lhs.difference(rhs)
}

func -(lhs: Matrix, rhs: Matrix) -> Matrix {
    return lhs.difference(rhs)
}

func *(lhs: Vector, rhs: Double) -> Vector {
    return lhs.scale(rhs)
}

func *(lhs: Double, rhs: Vector) -> Vector {
    return rhs.scale(lhs)
}

// Product the column vector and row vector
func *(lhs: Vector, rhs: Vector) -> Matrix {
    assert(lhs.cols != rhs.cols)
    return lhs.calculator * rhs.calculator
}

func *(lhs: Matrix, rhs: Double) -> Matrix {
    return lhs.scale(rhs)
}

func *(lhs: Double, rhs: Matrix) -> Matrix {
    return rhs.scale(lhs)
}

func *(lhs: Matrix, rhs: Vector) -> Vector {
    return lhs.dot(rhs)
}

func *(lhs: Matrix, rhs: Matrix) -> Matrix {
    return lhs.dot(rhs)
}

func sum(_ vector: Vector) -> Double {
    let array = vector.toArray()
    return cblas_dasum(Int32(array.count), array, 1)
}

func square(_ vector: Vector) -> Vector {
    let array = vector.toArray()
    var results = [Double](repeating: 0.0, count: array.count)
    
    vDSP_vsqD(array, 1, &results, 1, vDSP_Length(array.count))
    
    return Vector(array: results)
}

func sigmoid(_ scalar: Double) -> Double {
    return 1.0 / (1.0  + exp(-scalar))
}

func sigmoid(_ vector: Vector) -> Vector {
    let array = vector.toArray().map { sigmoid($0) }
    return Vector(array: array)
}

// y = (sigmoid(x) * (1 - sigmoid(x))
func sigmoidPrime(_ vector: Vector) -> Vector {
    
    return sigmoid(vector).multiply(Vector(value: 1.0, rows: vector.rows) - sigmoid(vector))
}

// y = (e^x / sum(e^x))
func softmax(_ vector: Vector) -> Vector {
    let exps = vector.buffer.map { exp($0) }
    let sum = exps.reduce(0.0, +)
    let y = exps.map { $0 / sum }
    
    return Vector(array: y)
}

func tanh(_ vector: Vector) -> Vector {
    let array = vector.toArray()
    var results = [Double](repeating: 0.0, count: array.count)
    vvtanh(&results, array, [Int32(array.count)])
    
    return Vector(array: results)
}

// y = (1 - tanh(x)^2)
func tanhPrime(_ vector: Vector) -> Vector {
    
    return Vector(value: 1, rows: vector.rows) - square(tanh(vector))
}

func toOneHot(value: Int, maxValue: Int) -> [Double] {
    var result = [Double](repeating: 0.0, count: maxValue + 1)
    result[value] = 1.0
    return result
}

func toOneHot(values: [Int], maxValue: Int) -> [[Double]] {
    return values.map { toOneHot(value: $0, maxValue: maxValue) }
}
