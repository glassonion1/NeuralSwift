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
    
    // bias for forget gate
    var bf: Vector
    // bias for input gate
    var bi: Vector
    // bias for output gate
    var bo: Vector
    // bias for input activation
    var ba: Vector
    
    init(wf: Matrix, wi: Matrix, wo: Matrix, wa: Matrix, rf: Matrix, ri: Matrix, ro: Matrix, ra: Matrix,
         bf: Vector, bi: Vector, bo: Vector, ba: Vector) {
        self.wf = wf
        self.wi = wi
        self.wo = wo
        self.wa = wa
        
        self.rf = rf
        self.ri = ri
        self.ro = ro
        self.ra = ra
        
        self.bf = bf
        self.bi = bi
        self.bo = bo
        self.ba = ba
        
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
    
    init(wf: Matrix, wi: Matrix, wo: Matrix, wa: Matrix, rf: Matrix, ri: Matrix, ro: Matrix, ra: Matrix) {
        let bf = Vector(value: 1.0, rows: rf.rows)
        let bi = Vector(value: 0.0, rows: ri.rows)
        let bo = Vector(value: 0.0, rows: ro.rows)
        let ba = Vector(value: 0.0, rows: ra.rows)
        self.init(wf: wf, wi: wi, wo: wo, wa: wa, rf: rf, ri: ri, ro: ro, ra: ra, bf: bf, bi: bi, bo: bo, ba: ba)
    }
    
    mutating func forward(x: Vector, state: (Vector, Vector)) -> (Vector, Vector) {
        
        assert(x.rows == wf.cols)
        assert(state.0.rows == wf.rows)
        assert(state.0.rows == rf.cols)
        
        let prevY = state.0
        prevCell = state.1
        
        inputActivationValue = inputActivation.forward(x: wa.dot(x) + ra.dot(prevY) + ba)
        inputGateValue = inputGate.forward(x: wi.dot(x) + ri.dot(prevY) + bi)
        forgetGateValue = forgetGate.forward(x: wf.dot(x) + rf.dot(prevY) + bf)
        outputGateValue = outputGate.forward(x: wo.dot(x) + ro.dot(prevY) + bo)
        
        let newCell = inputGateValue.multiply(inputActivationValue) + prevCell.multiply(forgetGateValue)
        let y = cellActivation.forward(x: newCell).multiply(outputGateValue)
        
        self.cell = newCell
        return (y, newCell)
    }
    
    func backward(deltaX: Vector, recurrentOut: (Vector, Vector), nextForget: Vector) -> (Vector, Vector, [String: Vector]) {
        let recurrentDeltaY = recurrentOut.0
        let recurrentDeltaCell = recurrentOut.1
        
        let delta = deltaX + recurrentDeltaY
        let dCell = cellActivation.backward(y: tanh(cell), delta: delta.multiply(outputGateValue))
            + recurrentDeltaCell.multiply(nextForget)
        
        let dInputActivation = inputActivation.backward(y: inputActivationValue, delta: dCell.multiply(inputGateValue))
        let dInputGate = inputGate.backward(y: inputGateValue, delta: dCell.multiply(inputActivationValue))
        let dForgetGate = forgetGate.backward(y: forgetGateValue, delta: dCell.multiply(prevCell))
        let dOutputGate = outputGate.backward(y: outputGateValue, delta: delta.multiply(tanh(cell)))
        
        /* for debugging
        let dWInputActivation = wa.transpose() * dInputActivation
        let dWInputGate = wi.transpose() * dInputGate
        let dWForgetGate  = wf.transpose() * dForgetGate
        let dWOutputGate  = wo.transpose() * dOutputGate
        
        let dX = dWInputActivation + dWInputGate + dWForgetGate + dWOutputGate
        */
        
        let dRInputActivation = ra.transpose() * dInputActivation
        let dRInputGate = ri.transpose() * dInputGate
        let dRForgetGate  = rf.transpose() * dForgetGate
        let dROutputGate  = ro.transpose() * dOutputGate
        
        let dY = dRInputActivation + dRInputGate + dRForgetGate + dROutputGate
        
        let dictionary = ["deltaA": dInputActivation,
                          "deltaI": dInputGate,
                          "deltaF": dForgetGate,
                          "deltaO": dOutputGate]
        
        return (dY, dCell, dictionary)
    }
}
