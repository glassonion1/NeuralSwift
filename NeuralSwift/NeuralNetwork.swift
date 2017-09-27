//
//  NeuralNetwork.swift
//  Matrix
//
//  Created by taisuke fujita on 2017/07/14.
//  Copyright © 2017年 Taisuke Fujita. All rights reserved.
//

import Foundation

public struct NeuralNetwork {

    private let learningRate: Double
    
    // Weight between the input layer and hidden layer
    var w1: Matrix
    // Weight between the hidden layer and output layer
    var w2: Matrix
    
    var hiddenLayer: Layer
    var outputLayer: Layer
    
    public init(inputLayerSize: Int, hiddenLayerSize: Int, outputLayerSize: Int, learningRate: Double) {
        self.learningRate = learningRate
        w1 = Matrix(rows: hiddenLayerSize, cols: inputLayerSize)
        w2 = Matrix(rows: outputLayerSize, cols: hiddenLayerSize)
        
        hiddenLayer = SigmoidLayer()
        outputLayer = SigmoidLayer()
    }
    
    public func query(list: [Double]) -> [Double] {
        let inputs = Vector(array: list)
        
        let hiddenOutputs = hiddenLayer.forward(x: w1 * inputs)
        
        let finalOutputs = outputLayer.forward(x: w2 * hiddenOutputs)
        
        return finalOutputs.toArray()
    }
    
    public mutating func train(inputList: [Double], targetList: [Double]) -> Double {
        
        let inputs = Vector(array: inputList)
        
        let hiddenOutputs = hiddenLayer.forward(x: w1 * inputs)
        
        let finalOutputs = outputLayer.forward(x: w2 * hiddenOutputs)
        
        let targets = Vector(array: targetList)
        
        let outputErrors = finalOutputs - targets
        let hiddenErros = w2.transpose().dot(outputErrors)
        
        // Update weight2
        let deltaOutput = outputLayer.backward(y: finalOutputs, delta: outputErrors) * hiddenOutputs.transpose()
        w2 = w2 - deltaOutput * learningRate
        
        // Update weight1
        let deltaHidden = hiddenLayer.backward(y: hiddenOutputs, delta: hiddenErros) * inputs.transpose()
        w1 = w1 - deltaHidden * learningRate
        
        return sum(outputErrors.multiply(outputErrors) * 0.5) / Double(targetList.count)
    }
}
