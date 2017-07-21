//
//  Vector.swift
//  Matrix
//
//  Created by taisuke fujita on 2017/07/18.
//  Copyright © 2017年 Taisuke Fujita. All rights reserved.
//

import Foundation
import Accelerate

struct Vector {
    let calculator: Matrix
    
    var rows: Int {
        return calculator.rows
    }
    var cols: Int {
        return calculator.cols
    }
    
    var buffer: [Double] {
        return calculator.buffer
    }
    
    init(array: [Double]) {
        calculator = Matrix(array: array)
    }
    
    init(value: Double, rows: Int) {
        calculator = Matrix(value: value, rows: rows, cols: 1)
    }
    
    init(calculator: Matrix) {
        self.calculator = calculator
    }
    
    subscript(i: Int) -> Double {
        return buffer[i]
    }
    
    func toArray() -> [Double] {
        return buffer
    }
    
    func filled(value: Double) -> Vector {
        return Vector(value: value, rows: rows)
    }
    
    func transpose() -> Vector {
        return Vector(calculator: calculator.transpose())
    }
    
    func scale(_ scalar: Double) -> Vector {
        return Vector(calculator: calculator.scale(scalar))
    }
    
    func sum(_ other: Double) -> Vector {
        return Vector(calculator: calculator.sum(other))
    }
    
    func sum(_ other: Vector) -> Vector {
        return Vector(calculator: calculator.sum(other.calculator))
    }
    
    func difference(_ other: Double) -> Vector {
        return Vector(calculator: calculator.difference(other))
    }
    
    func difference(_ other: Vector) -> Vector {
        return Vector(calculator: calculator.difference(other.calculator))
    }
    
    // Hadamard product
    func multiply(_ other: Vector) -> Vector {
        let matrix = calculator.multiply(other.calculator)
        return Vector(calculator: matrix)
    }
    
    // Inner product
    func dot(_ other: Vector) -> Double {
        assert(rows == other.rows && cols == other.cols)
        let la = la_inner_product(calculator.linearAlgebraObject,
                                  other.calculator.linearAlgebraObject)
        let matrix = Matrix(linearAlgebraObject: la)
        return matrix[0]
    }
}
