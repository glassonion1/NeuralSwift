//
//  NeuralNetwork.swift
//  Matrix
//
//  Created by taisuke fujita on 2017/07/14.
//  Copyright © 2017年 Taisuke Fujita. All rights reserved.
//

import Foundation

public class NeuralNetwork {

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
        
        hiddenLayer = SigmoidLayer(value: Vector(value: 0.0, rows: w1.rows))
        outputLayer = SigmoidLayer(value: Vector(value: 0.0, rows: w2.rows))
    }
    
    public func query(list: [Double]) -> [Double] {
        let inputs = Vector(array: list)
        
        hiddenLayer.forward(x: w1 * inputs)
        
        outputLayer.forward(x: w2 * hiddenLayer.value)
        
        return outputLayer.value.toArray()
    }
    
    public func train(inputList: [Double], targetList: [Double]) -> Double {
        
        let inputs = Vector(array: inputList)
        
        hiddenLayer.forward(x: w1 * inputs)
        
        outputLayer.forward(x: w2 * hiddenLayer.value)
        
        let targets = Vector(array: targetList)
        
        let outputErrors = outputLayer.value - targets
        let hiddenErros = w2.transpose().dot(outputErrors)
        
        // Update weight2
        outputLayer.backward(delta: outputErrors)
        let deltaOutput = outputLayer.value * hiddenLayer.value.transpose()
        w2 = w2 - deltaOutput * learningRate
        
        // Update weight1
        hiddenLayer.backward(delta: hiddenErros)
        let deltaHidden =  hiddenLayer.value * inputs.transpose()
        w1 = w1 - deltaHidden * learningRate
        
        return sum(outputErrors.multiply(outputErrors) * 0.5) / Double(targetList.count)
    }
}
