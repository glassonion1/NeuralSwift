//
//  Matrix.swift
//  ImprovisationPiano
//
//  Created by taisuke fujita on 2017/07/12.
//  Copyright © 2017年 taisuke fujita. All rights reserved.
//

import Foundation
import Accelerate
import GameplayKit

public struct Matrix: Variable {

    fileprivate let buffer: [Double]
    let rows: Int
    let cols: Int
    
    var linearAlgebraObject: la_object_t {
        return la_matrix_from_double_buffer(buffer,
                                            la_count_t(rows), la_count_t(cols),
                                            la_count_t(cols),
                                            la_hint_t(LA_NO_HINT),
                                            la_attribute_t(LA_DEFAULT_ATTRIBUTES))
    }
    
    public init(array: [[Double]]) {
        self.rows = array.count
        self.cols = array[0].count
        self.buffer = Array(array.joined())
    }
    
    public init(array: [Double], cols: Int = 1) {
        self.rows = array.count / cols
        self.cols = cols
        self.buffer = array
    }
    
    public init(rows: Int, cols: Int) {
        self.rows = rows
        self.cols = cols
        let rand = GKGaussianDistribution(randomSource: GKMersenneTwisterRandomSource(),
                                          lowestValue: -500, highestValue: 500)
        self.buffer = [Double](repeating: 0.0, count: rows * cols).map { _ in Double(rand.nextInt()) / 1000.0 }
    }
    
    public init(value: Double, rows: Int, cols: Int) {
        self.rows = rows
        self.cols = cols
        self.buffer = [Double](repeating: value, count: rows * cols)
    }
    
    public init(linearAlgebraObject: la_object_t) {
        self.rows = Int(la_matrix_rows(linearAlgebraObject))
        self.cols = Int(la_matrix_cols(linearAlgebraObject))
        var buf = [Double](repeating: 0.0, count: rows * cols)
        la_matrix_to_double_buffer(&buf, la_count_t(self.cols), linearAlgebraObject)
        self.buffer = buf
    }
    
    public subscript(row: Int, col: Int) -> Double {
        return buffer[row * self.cols + col]
    }
    
    public subscript(i: Int) -> Double {
        return self[i, 0]
    }
    
    public func toArray() -> [Double] {
        return buffer
    }
    
    public func filled(value: Double) -> Matrix {
        return Matrix(value: value, rows: rows, cols: cols)
    }
    
    public func transpose() -> Matrix {
        let la = la_transpose(linearAlgebraObject)
        return Matrix(linearAlgebraObject: la)
    }
    
    public func scale(_ scalar: Double) -> Matrix {
        let la = la_scale_with_double(linearAlgebraObject, scalar)
        return Matrix(linearAlgebraObject: la)
    }
    
    public func add(_ other: Double) -> Matrix {
        let la = la_sum(linearAlgebraObject, filled(value: other).linearAlgebraObject)
        return Matrix(linearAlgebraObject: la)
    }
    
    public func add(_ other: Matrix) -> Matrix {
        let la = la_sum(linearAlgebraObject, other.linearAlgebraObject)
        return Matrix(linearAlgebraObject: la)
    }
    
    public func difference(_ other: Double) -> Matrix {
        let la = la_difference(linearAlgebraObject, filled(value: other).linearAlgebraObject)
        return Matrix(linearAlgebraObject: la)
    }
    
    public func difference(_ other: Matrix) -> Matrix {
        let la = la_difference(linearAlgebraObject, other.linearAlgebraObject)
        return Matrix(linearAlgebraObject: la)
    }
    
    // Hadamard product
    public func multiply(_ other: Matrix) -> Matrix {
        assert(rows == other.rows && cols == other.cols)
        let la = la_elementwise_product(linearAlgebraObject, other.linearAlgebraObject)
        return Matrix(linearAlgebraObject: la)
    }
    
    public func dot(_ other: Matrix) -> Matrix {
        let new = la_matrix_product(linearAlgebraObject, other.linearAlgebraObject)
        return Matrix(linearAlgebraObject: new)
    }
    
    public func dot(_ other: Vector) -> Vector {
        return Vector(calculator: dot(other.calculator))
    }
}

extension Matrix: CustomStringConvertible {
    public var description: String {
        let array = (0..<rows).map { (row: Int) -> [Double] in
            let from = cols * row
            let to = cols * (row + 1)
            return Array(buffer[from..<to])
        }
        return "[" + array.map({$0.description}).joined(separator: "\n") + "]"
    }
}
