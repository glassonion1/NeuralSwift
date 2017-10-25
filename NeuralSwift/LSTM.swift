//
//  LSTM.swift
//  SimpleNN
//
//  Created by taisuke fujita on 2017/08/01.
//  Copyright © 2017年 taisuke fujita. All rights reserved.
//

import Foundation

public class LSTM {
    
    var forgetGate: Layer
    var inputGate: Layer
    var outputGate: Layer
    var inputActivation: Layer
    var cellActivation: Layer
    
    var forgetGateValue: Vector
    var prevCell: Vector
    
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
        
        forgetGate = SigmoidLayer(value: Vector(value: 0.0, rows: wf.rows))
        inputGate = SigmoidLayer(value: Vector(value: 0.0, rows: wf.rows))
        outputGate = SigmoidLayer(value: Vector(value: 0.0, rows: wf.rows))
        inputActivation = TanhLayer(value: Vector(value: 0.0, rows: wf.rows))
        cellActivation = TanhLayer(value: Vector(value: 0.0, rows: wf.rows))
        
        forgetGateValue = Vector(value: 0.0, rows: wf.rows)
        prevCell = Vector(value: 0.0, rows: rf.cols)
    }
    
    convenience init(wf: Matrix, wi: Matrix, wo: Matrix, wa: Matrix, rf: Matrix, ri: Matrix, ro: Matrix, ra: Matrix) {
        let bf = Vector(value: 1.0, rows: rf.rows)
        let bi = Vector(value: 0.0, rows: ri.rows)
        let bo = Vector(value: 0.0, rows: ro.rows)
        let ba = Vector(value: 0.0, rows: ra.rows)
        self.init(wf: wf, wi: wi, wo: wo, wa: wa, rf: rf, ri: ri, ro: ro, ra: ra, bf: bf, bi: bi, bo: bo, ba: ba)
    }
    
    func forward(x: Vector, state: (Vector, Vector)) -> (Vector, Vector) {
        
        assert(x.rows == wf.cols)
        assert(state.0.rows == wf.rows)
        assert(state.0.rows == rf.cols)
        
        let prevY = state.0
        prevCell = state.1
        
        inputActivation.forward(x: wa.dot(x) + ra.dot(prevY) + ba)
        inputGate.forward(x: wi.dot(x) + ri.dot(prevY) + bi)
        forgetGate.forward(x: wf.dot(x) + rf.dot(prevY) + bf)
        outputGate.forward(x: wo.dot(x) + ro.dot(prevY) + bo)
        
        let cell = inputGate.value.multiply(inputActivation.value) + prevCell.multiply(forgetGate.value)
        cellActivation.forward(x: cell)
        let y = cellActivation.value.multiply(outputGate.value)
        
        forgetGateValue = forgetGate.value
        return (y, cell)
    }
    
    func backward(deltaX: Vector, recurrentOut: (Vector, Vector), nextForget: Vector) -> (Vector, Vector) {
        let recurrentDeltaY = recurrentOut.0
        let recurrentDeltaCell = recurrentOut.1
        
        let delta = deltaX + recurrentDeltaY
        
        // it caches value before layers backward
        let cellActivationValue = cellActivation.value
        let inputActivationValue = inputActivation.value
        let inputGateValue = inputGate.value
        
        cellActivation.backward(delta: delta.multiply(outputGate.value))
        let dCell = cellActivation.value + recurrentDeltaCell.multiply(nextForget)
        
        inputActivation.backward(delta: dCell.multiply(inputGateValue))
        inputGate.backward(delta: dCell.multiply(inputActivationValue))
        forgetGate.backward(delta: dCell.multiply(prevCell))
        outputGate.backward(delta: delta.multiply(cellActivationValue))
        
        /* for debugging
        let dWInputActivation = wa.transpose() * inputActivation.value
        let dWInputGate = wi.transpose() * inputGate.value
        let dWForgetGate  = wf.transpose() * forgetGate.value
        let dWOutputGate  = wo.transpose() * outputGate.value
        
        let dX = dWInputActivation + dWInputGate + dWForgetGate + dWOutputGate
        */
        
        let dRInputActivation = ra.transpose() * inputActivation.value
        let dRInputGate = ri.transpose() * inputGate.value
        let dRForgetGate  = rf.transpose() * forgetGate.value
        let dROutputGate  = ro.transpose() * outputGate.value
        
        let dY = dRInputActivation + dRInputGate + dRForgetGate + dROutputGate
        
        return (dY, dCell)
    }
}
