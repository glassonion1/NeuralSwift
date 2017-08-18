//
//  LSTM.swift
//  SimpleNN
//
//  Created by taisuke fujita on 2017/08/01.
//  Copyright © 2017年 taisuke fujita. All rights reserved.
//

import Foundation

public struct LSTM {
    
    var forgetGate: Layer
    var inputGate: Layer
    var outputGate: Layer
    var inputActivation: Layer
    var cellActivation: Layer
    
    var forgetGateValue: Vector
    var inputGateValue: Vector
    var outputGateValue: Vector
    var inputActivationValue: Vector
    
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
        
        forgetGate = SigmoidLayer()
        inputGate = SigmoidLayer()
        outputGate = SigmoidLayer()
        inputActivation = TanhLayer()
        cellActivation = TanhLayer()
        
        forgetGateValue = Vector(value: 0.0, rows: wf.rows)
        inputGateValue = Vector(value: 0.0, rows: wf.rows)
        outputGateValue = Vector(value: 0.0, rows: wf.rows)
        inputActivationValue = Vector(value: 0.0, rows: wf.rows)
        
        prevCell = Vector(value: 0.0, rows: rf.cols)
        cell = Vector(value: 0.0, rows: rf.cols)
    }
    
    mutating func forward(x: Vector, state: (Vector, Vector)) -> (Vector, Vector) {
        
        assert(x.rows == wf.cols)
        assert(state.0.rows == wf.rows)
        assert(state.0.rows == rf.cols)
        
        let h = state.0
        prevCell = state.1
        
        inputActivationValue = inputActivation.forward(x: wa.dot(x) + ra.dot(h))
        inputGateValue = inputGate.forward(x: wi.dot(x) + ri.dot(h))
        forgetGateValue = forgetGate.forward(x: wf.dot(x) + rf.dot(h))
        outputGateValue = outputGate.forward(x: wo.dot(x) + ro.dot(h))
        
        let newCell = inputGateValue.multiply(inputActivationValue) + prevCell.multiply(forgetGateValue)
        let newH = cellActivation.forward(x: newCell).multiply(outputGateValue)
        
        self.cell = newCell
        return (newH, newCell)
    }
    
    func backward(deltaX: Vector, recurrentOut: (Vector, Vector, Vector), nextForget: Vector) -> (Vector, Vector, Vector, [String: Vector]) {
        let recurrentDeltaY = recurrentOut.1
        let recurrentDeltaCell = recurrentOut.2
        
        let delta = deltaX + recurrentDeltaY
        let dCell = cellActivation.backward(y: tanh(cell), delta: delta.multiply(outputGateValue))
            + recurrentDeltaCell.multiply(nextForget)
        
        let dInputActivation = inputActivation.backward(y: inputActivationValue, delta: dCell.multiply(inputGateValue))
        let dInputGate = inputGate.backward(y: inputGateValue, delta: dCell.multiply(inputActivationValue))
        let dForgetGate = forgetGate.backward(y: forgetGateValue, delta: dCell.multiply(prevCell))
        let dOutputGate = outputGate.backward(y: outputGateValue, delta: delta.multiply(tanh(cell)))
        
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
