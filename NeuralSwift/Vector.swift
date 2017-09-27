//
//  Vector.swift
//  Matrix
//
//  Created by taisuke fujita on 2017/07/18.
//  Copyright © 2017年 Taisuke Fujita. All rights reserved.
//

import Foundation
import Accelerate

public struct Vector: Variable {
    let calculator: Matrix
    
    private var buffer: [Double] {
        return calculator.toArray()
    }
    
    public var rows: Int {
        return calculator.rows
    }
    public var cols: Int {
        return calculator.cols
    }
    
    public init(array: [Double]) {
        calculator = Matrix(array: array)
    }
    
    public init(value: Double, rows: Int) {
        calculator = Matrix(value: value, rows: rows, cols: 1)
    }
    
    public init(calculator: Matrix) {
        self.calculator = calculator
    }
    
    public subscript(i: Int) -> Double {
        return buffer[i]
    }
    
    public func toArray() -> [Double] {
        return buffer
    }
    
    public func filled(value: Double) -> Vector {
        return Vector(value: value, rows: rows)
    }
    
    public func transpose() -> Vector {
        return Vector(calculator: calculator.transpose())
    }
    
    public func scale(_ scalar: Double) -> Vector {
        return Vector(calculator: calculator.scale(scalar))
    }
    
    public func add(_ other: Double) -> Vector {
        return Vector(calculator: calculator.add(other))
    }
    
    public func add(_ other: Vector) -> Vector {
        return Vector(calculator: calculator.add(other.calculator))
    }
    
    public func difference(_ other: Double) -> Vector {
        return Vector(calculator: calculator.difference(other))
    }
    
    public func difference(_ other: Vector) -> Vector {
        return Vector(calculator: calculator.difference(other.calculator))
    }
    
    // Hadamard product
    public func multiply(_ other: Vector) -> Vector {
        let matrix = calculator.multiply(other.calculator)
        return Vector(calculator: matrix)
    }
    
    // Inner product
    public func dot(_ other: Vector) -> Double {
        assert(rows == other.rows && cols == other.cols)
        let la = la_inner_product(calculator.linearAlgebraObject,
                                  other.calculator.linearAlgebraObject)
        let matrix = Matrix(linearAlgebraObject: la)
        return matrix[0]
    }
    
    public func outer(_ other: Vector) -> Matrix {
        let new = la_outer_product(calculator.linearAlgebraObject, other.calculator.linearAlgebraObject)
        return Matrix(linearAlgebraObject: new)
    }
}
