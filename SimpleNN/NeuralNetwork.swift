//
//  NeuralNetwork.swift
//  Matrix
//
//  Created by taisuke fujita on 2017/07/14.
//  Copyright © 2017年 Taisuke Fujita. All rights reserved.
//

import Foundation

public struct NeuralNetwork {
    private let inputLayerSize: Int
    private let hiddenLayerSize: Int
    private let outputLayerSize: Int
    
    private let learningRate: Double
    
    // Weight between the input layer and hidden layer
    var w1: Matrix
    // Weight between the hidden layer and output layer
    var w2: Matrix
    
    public init(inputLayerSize: Int, hiddenLayerSize: Int, outputLayerSize: Int, learningRate: Double) {
        self.inputLayerSize = inputLayerSize
        self.outputLayerSize = outputLayerSize
        self.hiddenLayerSize = hiddenLayerSize
        self.learningRate = learningRate
        w1 = Matrix(rows: hiddenLayerSize, cols: inputLayerSize)
        w2 = Matrix(rows: outputLayerSize, cols: hiddenLayerSize)
    }
    
    func forward(weight: Matrix, inputs: Vector) -> Vector {
        return sigmoid(weight.dot(inputs))
    }
    
    func backward(inputs: Vector, outputs: Vector, errors: Vector) -> Matrix {
        let columnVector = errors.multiply(outputs).multiply(1.0 - outputs)
        let rowVector = inputs.transpose()
        return learningRate * (columnVector * rowVector)
    }
    
    public func query(list: [Double]) -> [Double] {
        let inputs = Vector(array: list)
        
        let hiddenOutputs = forward(weight: w1, inputs: inputs)
        
        let finalOutputs = forward(weight: w2, inputs: hiddenOutputs)
        
        return finalOutputs.toArray()
    }
    
    public mutating func train(inputList: [Double], targetList: [Double]) {
        let inputs = Vector(array: inputList)
        
        let hiddenOutputs = forward(weight: w1, inputs: inputs)
        
        let finalOutputs = forward(weight: w2, inputs: hiddenOutputs)
        
        let targets = Vector(array: targetList)
        
        let outputErrors = targets - finalOutputs
        let hiddenErros = w2.transpose().dot(outputErrors)
        
        // Update weight2
        w2 = w2 + backward(inputs: hiddenOutputs, outputs: finalOutputs, errors: outputErrors)
        
        // Update weight1
        w1 = w1 + backward(inputs: inputs, outputs: hiddenOutputs, errors: hiddenErros)
    }
}
