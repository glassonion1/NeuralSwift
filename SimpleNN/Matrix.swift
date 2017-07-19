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

struct Matrix {
    let linearAlgebraObject: la_object_t
    
    var rows: Int {
        return Int(la_matrix_rows(linearAlgebraObject))
    }
    var cols: Int {
        return Int(la_matrix_cols(linearAlgebraObject))
    }
    
    var buffer: [Double] {
        var buf = [Double](repeating: 0.0, count: rows * cols)
        la_matrix_to_double_buffer(&buf, la_count_t(cols), linearAlgebraObject)
        return buf
    }
    
    init(array: [[Double]]) {
        let rows = array.count
        let cols = array[0].count
        let buffer = Array(array.joined())
        self.linearAlgebraObject = la_matrix_from_double_buffer(buffer,
                                                                la_count_t(rows), la_count_t(cols),
                                                                la_count_t(cols),
                                                                la_hint_t(LA_NO_HINT),
                                                                la_attribute_t(LA_DEFAULT_ATTRIBUTES))
    }
    
    init(array: [Double], cols: Int = 1) {
        let rows = array.count / cols
        self.linearAlgebraObject = la_matrix_from_double_buffer(array,
                                                                la_count_t(rows), la_count_t(cols),
                                                                la_count_t(cols),
                                                                la_hint_t(LA_NO_HINT),
                                                                la_attribute_t(LA_DEFAULT_ATTRIBUTES))
    }
    
    init(rows: Int, cols: Int) {
        let rand = GKGaussianDistribution(randomSource: GKMersenneTwisterRandomSource(),
                                          lowestValue: -500, highestValue: 500)
        let buf = [Double](repeating: 0.0, count: rows * cols).map { _ in Double(rand.nextInt()) / 1000.0 }
        self.linearAlgebraObject = la_matrix_from_double_buffer(buf,
                                                                la_count_t(rows), la_count_t(cols),
                                                                la_count_t(cols),
                                                                la_hint_t(LA_NO_HINT),
                                                                la_attribute_t(LA_DEFAULT_ATTRIBUTES))
    }
    
    init(value: Double, rows: Int, cols: Int) {
        let splat = la_splat_from_double(value, la_attribute_t(LA_DEFAULT_ATTRIBUTES))
        self.linearAlgebraObject = la_matrix_from_splat(splat, la_count_t(rows), la_count_t(cols))
    }
    
    init(linearAlgebraObject: la_object_t) {
        self.linearAlgebraObject = linearAlgebraObject
    }
    
    subscript(row: Int, col: Int) -> Double {
        return buffer[row * self.cols + col]
    }
    
    subscript(i: Int) -> Double {
        return self[i, 0]
    }
    
    func toArray() -> [Double] {
        return buffer
    }
    
    func filled(value: Double) -> Matrix {
        return Matrix(value: value, rows: rows, cols: cols)
    }
    
    func transpose() -> Matrix {
        let linearAlgebraObject = la_transpose(self.linearAlgebraObject)
        return Matrix(linearAlgebraObject: linearAlgebraObject)
    }
    
    func scale(_ scalar: Double) -> Matrix {
        let linearAlgebraObject = la_scale_with_double(self.linearAlgebraObject, scalar)
        return Matrix(linearAlgebraObject: linearAlgebraObject)
    }
    
    func sum(_ other: Double) -> Matrix {
        let linearAlgebraObject = la_sum(self.linearAlgebraObject, filled(value: other).linearAlgebraObject)
        return Matrix(linearAlgebraObject: linearAlgebraObject)
    }
    
    func sum(_ other: Matrix) -> Matrix {
        let linearAlgebraObject = la_sum(self.linearAlgebraObject, other.linearAlgebraObject)
        return Matrix(linearAlgebraObject: linearAlgebraObject)
    }
    
    func difference(_ other: Double) -> Matrix {
        let linearAlgebraObject = la_difference(self.linearAlgebraObject, filled(value: other).linearAlgebraObject)
        return Matrix(linearAlgebraObject: linearAlgebraObject)
    }
    
    func difference(_ other: Matrix) -> Matrix {
        let linearAlgebraObject = la_difference(self.linearAlgebraObject, other.linearAlgebraObject)
        return Matrix(linearAlgebraObject: linearAlgebraObject)
    }
    
    // Hadamard product
    func multiply(_ other: Matrix) -> Matrix {
        assert(rows == other.rows && cols == other.cols)
        let linearAlgebraObject = la_elementwise_product(self.linearAlgebraObject, other.linearAlgebraObject)
        return Matrix(linearAlgebraObject: linearAlgebraObject)
    }
    
    func dot(_ other: Matrix) -> Matrix {
        let new = la_matrix_product(linearAlgebraObject, other.linearAlgebraObject)
        return Matrix(linearAlgebraObject: new)
    }
    
    func dot(_ other: Vector) -> Vector {
        return Vector(calculator: dot(other.calculator))
    }
}

extension Matrix: CustomStringConvertible {
    var description: String {
        let array = (0..<rows).map { (row: Int) -> [Double] in
            let from = cols * row
            let to = cols * (row + 1)
            return Array(buffer[from..<to])
        }
        return "[" + array.map({$0.description}).joined(separator: "\n") + "]"
    }
}
