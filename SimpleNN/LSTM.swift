//
//  LSTM.swift
//  SimpleNN
//
//  Created by taisuke fujita on 2017/08/01.
//  Copyright © 2017年 taisuke fujita. All rights reserved.
//

import Foundation

public struct LSTM {
    
    var forgetGate: Vector
    var inputGate: Vector
    var outputGate: Vector
    var inputActivation: Vector
    
    var prevCell: Vector
    var cell: Vector
    
    // weight for forget gate
    var wf: Matrix
    // weight for input gate
    var wi: Matrix
    // weight for output gate
    var wo: Matrix
    // weight for input activation
    var wa: Matrix
    
    // recurrent weight for forget gate
    var rf: Matrix
    // recurrent weight for input gate
    var ri: Matrix
    // recurrent weight for output gate
    var ro: Matrix
    // recurrent weight for input activation
    var ra: Matrix
    
    init(wf: Matrix, wi: Matrix, wo: Matrix, wa: Matrix, rf: Matrix, ri: Matrix, ro: Matrix, ra: Matrix) {
        self.wf = wf
        self.wi = wi
        self.wo = wo
        self.wa = wa
        
        self.rf = rf
        self.ri = ri
        self.ro = ro
        self.ra = ra
        
        forgetGate = Vector(value: 0.0, rows: wf.rows)
        inputGate = Vector(value: 0.0, rows: wf.rows)
        outputGate = Vector(value: 0.0, rows: wf.rows)
        inputActivation = Vector(value: 0.0, rows: wf.rows)
        
        prevCell = Vector(value: 0.0, rows: rf.cols)
        cell = Vector(value: 0.0, rows: rf.cols)
    }
    
    mutating func forward(x: Vector, state: (Vector, Vector)) -> (Vector, Vector) {
        
        assert(x.rows == wf.cols)
        assert(state.0.rows == wf.rows)
        assert(state.0.rows == rf.cols)
        
        let h = state.0
        prevCell = state.1
        
        inputActivation = tanh(wa.dot(x) + ra.dot(h))
        inputGate = sigmoid(wi.dot(x) + ri.dot(h))
        forgetGate = sigmoid(wf.dot(x) + rf.dot(h))
        outputGate = sigmoid(wo.dot(x) + ro.dot(h))
        
        let newCell = inputGate.multiply(inputActivation) + prevCell.multiply(forgetGate)
        let newH = tanh(newCell).multiply(outputGate)
        
        self.cell = newCell
        return (newH, newCell)
    }

    func backward(deltaX: Vector, recurrentOut: (Vector, Vector, Vector), nextForget: Vector) -> (Vector, Vector, Vector, [String: Vector]) {
        let recurrentDeltaY = recurrentOut.1
        let recurrentDeltaCell = recurrentOut.2
        
        let delta = deltaX + recurrentDeltaY
        let dCell = delta.multiply(outputGate).multiply(tanhPrime(cell)) + recurrentDeltaCell.multiply(nextForget)
        
        let dInputActivation = dCell.multiply(inputGate).multiply(1 - square(inputActivation))
        let dInputGate = dCell.multiply(inputActivation).multiply(inputGate.multiply(1 - inputGate))
        let dForgetGate = dCell.multiply(prevCell).multiply(forgetGate.multiply(1 - forgetGate))
        let dOutputGate = delta.multiply(tanh(cell)).multiply(outputGate.multiply(1 - outputGate))
        
        let dWInputActivation = wa.transpose() * dInputActivation
        let dWInputGate = wi.transpose() * dInputGate
        let dWForgetGate  = wf.transpose() * dForgetGate
        let dWOutputGate  = wo.transpose() * dOutputGate
        
        let dX = dWInputActivation + dWInputGate + dWForgetGate + dWOutputGate
        
        let dRInputActivation = ra.transpose() * dInputActivation
        let dRInputGate = ri.transpose() * dInputGate
        let dRForgetGate  = rf.transpose() * dForgetGate
        let dROutputGate  = ro.transpose() * dOutputGate
        
        let dY = dRInputActivation + dRInputGate + dRForgetGate + dROutputGate
        
        let dictionary = ["deltaA": dInputActivation,
                          "deltaI": dInputGate,
                          "deltaF": dForgetGate,
                          "deltaO": dOutputGate]
        
        return (dX, dY, dCell, dictionary)
    }
}
