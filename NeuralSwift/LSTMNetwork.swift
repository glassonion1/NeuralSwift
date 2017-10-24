//
//  LSTMNetwork.swift
//  SimpleNN
//
//  Created by taisuke fujita on 2017/07/27.
//  Copyright © 2017年 taisuke fujita. All rights reserved.
//

import Foundation

extension Array {
    func chunks(_ chunkSize: Int) -> [[Element]] {
        return stride(from: 0, to: self.count, by: chunkSize).map {
            Array(self[$0..<Swift.min($0 + chunkSize, self.count)])
        }
    }
}

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
    
    // bias for forget gate
    var bf: Vector
    // bias for input gate
    var bi: Vector
    // bias for output gate
    var bo: Vector
    // bias for input activation
    var ba: Vector
    
    private let learningRate: Double
    
    var lstms = [LSTM]()
    
    public init(sequenceSize: Int, inputDataLength: Int, outputDataLength: Int, learningRate: Double) {
        self.sequenceSize = sequenceSize
        self.learningRate = learningRate
        
        wf = Matrix(rows: outputDataLength, cols: inputDataLength)
        wi = Matrix(rows: outputDataLength, cols: inputDataLength)
        wo = Matrix(rows: outputDataLength, cols: inputDataLength)
        wa = Matrix(rows: outputDataLength, cols: inputDataLength)
        
        rf = Matrix(rows: outputDataLength, cols: outputDataLength)
        ri = Matrix(rows: outputDataLength, cols: outputDataLength)
        ro = Matrix(rows: outputDataLength, cols: outputDataLength)
        ra = Matrix(rows: outputDataLength, cols: outputDataLength)
        
        bf = Vector(value: 1.0, rows: rf.rows)
        bi = Vector(value: 0.0, rows: ri.rows)
        bo = Vector(value: 0.0, rows: ro.rows)
        ba = Vector(value: 0.0, rows: ra.rows)
        
        reset()
    }
    
    mutating func reset() {
        var lstmList = [LSTM]()
        for _ in 0..<sequenceSize {
            lstmList.append(LSTM(wf: wf, wi: wi, wo: wo, wa: wa,
                                 rf: rf, ri: ri, ro: ro, ra: ra,
                                 bf: bf, bi: bi, bo: bo, ba: ba))
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
            //outputs.append(softmaxLayer.forward(x: wy * output).toArray())
            outputs.append(output.toArray())
        }
        
        return outputs
    }
    
    public mutating func train(inputLists: [[Double]], targetLists: [[Double]]) -> Double {
        let outputs = query(lists: inputLists)
        
        let dY = Vector(value: 0, rows: wf.rows)
        let dCell = Vector(value: 0, rows: wf.rows)
        var state = (dY, dCell)
        var deltas = [[String: Vector]]()
        var totalLoss = 0.0
        
        for i in (0..<lstms.count).reversed() {
            let lstm = lstms[i]
            let output = Vector(array: outputs[i])
            let target = Vector(array: targetLists[i])
            // a target is one hot vector like [0,1,0,0..0]
            let deltaX = output - target
            
            let loss = target - output
            let l2Loss = sum(loss.multiply(loss) * 0.5)
            totalLoss += l2Loss
            
            
            /*
            let loss = sum(target.multiply(log(output))) * -1
            totalLoss += loss
            */
            
            let next = i + 1 > lstms.count - 1 ? LSTM(wf: wf, wi: wi, wo: wo, wa: wa, rf: rf, ri: ri, ro: ro, ra: ra) : lstms[i + 1]
            let backward = lstm.backward(deltaX: deltaX, recurrentOut: state, nextForget: next.forgetGateValue)
            state.0 = backward.0
            state.1 = backward.1
            deltas.append(backward.2)
        }
        deltas = deltas.reversed()
        
        var dBa = Vector(value: 0, rows: ra.rows)
        var dBi = Vector(value: 0, rows: ri.rows)
        var dBf = Vector(value: 0, rows: rf.rows)
        var dBo = Vector(value: 0, rows: ro.rows)
        
        var dWa = Matrix(value: 0, rows: wa.rows, cols: wa.cols)
        var dWi = Matrix(value: 0, rows: wi.rows, cols: wi.cols)
        var dWf = Matrix(value: 0, rows: wf.rows, cols: wf.cols)
        var dWo = Matrix(value: 0, rows: wo.rows, cols: wo.cols)
        
        for i in 0..<inputLists.count {
            let delta = deltas[i]
            let xT = Vector(array: inputLists[i]).transpose()
            
            dBa = dBa + delta["deltaA"]!
            dBi = dBi + delta["deltaI"]!
            dBf = dBf + delta["deltaF"]!
            dBo = dBo + delta["deltaO"]!
            
            dWa = dWa + delta["deltaA"]!.outer(xT)
            dWi = dWi + delta["deltaI"]!.outer(xT)
            dWf = dWf + delta["deltaF"]!.outer(xT)
            dWo = dWo + delta["deltaO"]!.outer(xT)
        }
        ba = ba - learningRate * dBa
        bi = bi - learningRate * dBi
        bf = bf - learningRate * dBf
        bo = bo - learningRate * dBo
        
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
    
    public mutating func train2(inputLists: [[Double]], targetLists: [[Double]]) -> Double {
        
        let xList = inputLists.chunks(sequenceSize)
        var tList = targetLists.chunks(sequenceSize)
        assert(xList.count == tList.count)
        
        var totalLoss = 0.0
        
        for x in xList {
            var deltas = [[String: Vector]]()
            let y = query(lists: x)
            let t = tList.remove(at: 0)
            
            let dY = Vector(value: 0, rows: wf.rows)
            let dCell = Vector(value: 0, rows: wf.rows)
            var state = (dY, dCell)
            
            var sequenceLoss = 0.0
            
            for i in (0..<lstms.count).reversed() {
                // 端数対応
                if y.count <= i {
                    continue
                }
                let lstm = lstms[i]
                let output = Vector(array: y[i])
                let target = Vector(array: t[i])
                // a target is one hot vector like [0,1,0,0..0]
                let deltaX = output - target
                
                let loss = target - output
                let l2Loss = sum(loss.multiply(loss) * 0.5)
                sequenceLoss += l2Loss
                
                /*
                 let loss = sum(target.multiply(log(output))) * -1
                 totalLoss += loss
                 */
                
                let next = i + 1 == lstms.count ? LSTM(wf: wf, wi: wi, wo: wo, wa: wa, rf: rf, ri: ri, ro: ro, ra: ra) : lstms[i + 1]
                let backward = lstm.backward(deltaX: deltaX, recurrentOut: state, nextForget: next.forgetGateValue)
                state.0 = backward.0
                state.1 = backward.1
                deltas.insert(backward.2, at: 0)
            }
            totalLoss += sequenceLoss / Double(y.count)
            
            var dBa = Vector(value: 0, rows: ra.rows)
            var dBi = Vector(value: 0, rows: ri.rows)
            var dBf = Vector(value: 0, rows: rf.rows)
            var dBo = Vector(value: 0, rows: ro.rows)
            
            var dWa = Matrix(value: 0, rows: wa.rows, cols: wa.cols)
            var dWi = Matrix(value: 0, rows: wi.rows, cols: wi.cols)
            var dWf = Matrix(value: 0, rows: wf.rows, cols: wf.cols)
            var dWo = Matrix(value: 0, rows: wo.rows, cols: wo.cols)
            for i in 0..<x.count {
                let delta = deltas[i]
                
                dBa = dBa + delta["deltaA"]!
                dBi = dBi + delta["deltaI"]!
                dBf = dBf + delta["deltaF"]!
                dBo = dBo + delta["deltaO"]!
                
                let xT = Vector(array: x[i]).transpose()
                
                dWa = dWa + delta["deltaA"]!.outer(xT)
                dWi = dWi + delta["deltaI"]!.outer(xT)
                dWf = dWf + delta["deltaF"]!.outer(xT)
                dWo = dWo + delta["deltaO"]!.outer(xT)
            }
            ba = ba - learningRate * dBa
            bi = bi - learningRate * dBi
            bf = bf - learningRate * dBf
            bo = bo - learningRate * dBo
            
            wa = wa - learningRate * dWa
            wi = wi - learningRate * dWi
            wf = wf - learningRate * dWf
            wo = wo - learningRate * dWo
            
            var dRa = Matrix(value: 0, rows: ra.rows, cols: ra.cols)
            var dRi = Matrix(value: 0, rows: ri.rows, cols: ri.cols)
            var dRf = Matrix(value: 0, rows: rf.rows, cols: rf.cols)
            var dRo = Matrix(value: 0, rows: ro.rows, cols: ro.cols)
            
            for i in 0..<y.count - 1 {
                let delta = deltas[i + 1]
                let yT = Vector(array: y[i]).transpose()
                
                dRa = dRa + delta["deltaA"]!.outer(yT)
                dRi = dRi + delta["deltaI"]!.outer(yT)
                dRf = dRf + delta["deltaF"]!.outer(yT)
                dRo = dRo + delta["deltaO"]!.outer(yT)
            }
            
            ra = ra - learningRate * dRa
            ri = ri - learningRate * dRi
            rf = rf - learningRate * dRf
            ro = ro - learningRate * dRo
        }
        
        reset()
        
        return totalLoss / Double(xList.count)
    }
}
