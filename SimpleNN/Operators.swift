//
//  Operators.swift
//  SimpleNN
//
//  Created by taisuke fujita on 2017/07/19.
//  Copyright © 2017年 taisuke fujita. All rights reserved.
//

import Foundation

func +(lhs: Vector, rhs: Vector) -> Vector {
    return lhs.sum(rhs)
}

func +(lhs: Matrix, rhs: Double) -> Matrix {
    return lhs.sum(rhs)
}

func +(lhs: Matrix, rhs: Matrix) -> Matrix {
    return lhs.sum(rhs)
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

func sigmoid(_ scalar: Double) -> Double {
    return 1.0 / (1.0  + exp(-scalar))
}

func sigmoid(_ vector: Vector) -> Vector {
    let exped = vector.toArray().map { sigmoid($0) }
    return Vector(array: exped)
}

func sigmoid(_ matrix: Matrix) -> Matrix {
    let exped = matrix.toArray().map { sigmoid($0) }
    return Matrix(array: exped, cols: matrix.cols)
}
