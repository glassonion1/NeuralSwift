//
//  LSTMNetwork.swift
//  SimpleNN
//
//  Created by taisuke fujita on 2017/07/27.
//  Copyright © 2017年 taisuke fujita. All rights reserved.
//

import Foundation

// @see https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9
public struct LSTMNetwork {

    let sequenceSize: Int
    
    // activation input
    var wa: Matrix
    // input gate
    var wi: Matrix
    // forget gate
    var wf: Matrix
    // output gate
    var wo: Matrix
    
    // activation input
    var ra: Matrix
    // input gate
    var ri: Matrix
    // forget gate
    var rf: Matrix
    // output gate
    var ro: Matrix
    
    private let learningRate: Double
    
    var lstms = [LSTM]()
    
    public init(sequenceSize: Int, inputDataLength: Int, outputDataLength: Int, learningRate: Double) {
        self.sequenceSize = sequenceSize
        
        wf = Matrix(rows: outputDataLength, cols: inputDataLength)
        wi = Matrix(rows: outputDataLength, cols: inputDataLength)
        wo = Matrix(rows: outputDataLength, cols: inputDataLength)
        wa = Matrix(rows: outputDataLength, cols: inputDataLength)
        
        rf = Matrix(rows: outputDataLength, cols: outputDataLength)
        ri = Matrix(rows: outputDataLength, cols: outputDataLength)
        ro = Matrix(rows: outputDataLength, cols: outputDataLength)
        ra = Matrix(rows: outputDataLength, cols: outputDataLength)
        
        self.learningRate = learningRate
        
        reset()
    }
    
    mutating func reset() {
        var lstmList = [LSTM]()
        for _ in 0..<sequenceSize {
            lstmList.append(LSTM(wf: wf, wi: wi, wo: wo, wa: wa, rf: rf, ri: ri, ro: ro, ra: ra))
        }
        self.lstms = lstmList
    }
    
    // data: onehot vectors
    public mutating func query(lists: [[Double]]) -> [[Double]] {
        
        let h = Vector(value: 0, rows: wf.rows)
        let cell = Vector(value: 0, rows: wf.rows)
        var state = (h, cell)
        var outputs = [[Double]]()
        
        for i in 0..<lists.count {
            let inputs = Vector(array: lists[i])
            state = lstms[i].forward(x: inputs, state: state)
            
            let output = state.0
            //outputs.append(softmax(output).toArray())
            outputs.append(output.toArray())
        }
        
        return outputs
    }
    
    public mutating func train(inputLists: [[Double]], targetLists: [[Double]]) -> Double {
        let outputs = query(lists: inputLists)
        
        let dX = Vector(value: 0, rows: wf.rows)
        let dY = Vector(value: 0, rows: wf.rows)
        let dCell = Vector(value: 0, rows: wf.rows)
        var state = (dX, dY, dCell)
        var deltas = [[String: Vector]]()
        var totalLoss = 0.0
        
        for i in (0..<lstms.count).reversed() {
            let lstm = lstms[i]
            let output = Vector(array: outputs[i])
            let target = Vector(array: targetLists[i])
            let deltaX = output - target
            
            let l2Loss = sum(deltaX.multiply(deltaX) * 0.5)
            totalLoss += l2Loss
            
            let next = i + 1 > lstms.count - 1 ? LSTM(wf: wf, wi: wi, wo: wo, wa: wa, rf: rf, ri: ri, ro: ro, ra: ra) : lstms[i + 1]
            let result = lstm.backward(deltaX: deltaX, recurrentOut: state, nextForget: next.forgetGate)
            state.0 = result.0
            state.1 = result.1
            state.2 = result.2
            deltas.append(result.3)
        }
        deltas = deltas.reversed()
        
        var dWa = Matrix(value: 0, rows: wa.rows, cols: wa.cols)
        var dWi = Matrix(value: 0, rows: wi.rows, cols: wi.cols)
        var dWf = Matrix(value: 0, rows: wf.rows, cols: wf.cols)
        var dWo = Matrix(value: 0, rows: wo.rows, cols: wo.cols)
        for i in 0..<inputLists.count {
            let delta = deltas[i]
            let xT = Vector(array: inputLists[i]).transpose()
            
            dWa = dWa + delta["deltaA"]!.outer(xT)
            dWi = dWi + delta["deltaI"]!.outer(xT)
            dWf = dWf + delta["deltaF"]!.outer(xT)
            dWo = dWo + delta["deltaO"]!.outer(xT)
        }
        wa = wa - learningRate * dWa
        wi = wi - learningRate * dWi
        wf = wf - learningRate * dWf
        wo = wo - learningRate * dWo
        
        var dRa = Matrix(value: 0, rows: ra.rows, cols: ra.cols)
        var dRi = Matrix(value: 0, rows: ri.rows, cols: ri.cols)
        var dRf = Matrix(value: 0, rows: rf.rows, cols: rf.cols)
        var dRo = Matrix(value: 0, rows: ro.rows, cols: ro.cols)

        for i in 0..<outputs.count - 1 {
            let delta = deltas[i + 1]
            let yT = Vector(array: outputs[i]).transpose()
            
            dRa = dRa + delta["deltaA"]!.outer(yT)
            dRi = dRi + delta["deltaI"]!.outer(yT)
            dRf = dRf + delta["deltaF"]!.outer(yT)
            dRo = dRo + delta["deltaO"]!.outer(yT)
        }
        
        ra = ra - learningRate * dRa
        ri = ri - learningRate * dRi
        rf = rf - learningRate * dRf
        ro = ro - learningRate * dRo
        
        reset()
        
        return totalLoss / Double(outputs.count)
    }
}
